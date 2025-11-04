# =================================================================
# runners/diffusion.py (FINAL MERGED VERSION)
# Combines LPIPS/Validation fixes + LR Scheduler
# =================================================================

import logging
import time
import glob
import numpy as np
import tqdm
import torch.utils.data as data
import torchvision.utils as tvu
from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import data_transform, inverse_data_transform
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import os, torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
from datasets import get_dataset, data_transform
from functions.denoising import generalized_steps
from torch.cuda.amp import autocast, GradScaler

# --- [FIX 1/3] Imports: Add LPIPS ---
try:
    import lpips
except ImportError:
    lpips = None
# --- End FIX ---

# --- [SCHEDULER FIX 1/5] Imports: Add Schedulers ---
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


# --- End SCHEDULER FIX ---


# ==== Peak memory helpers ====
def _reset_peak_all_gpus():
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.reset_peak_memory_stats()


def _log_peak_all_gpus(step: int, micro_bs: int, accum: int):
    if not torch.cuda.is_available():
        return
    lines = [f"[mem] step={step} (batch={micro_bs} x accum={accum} -> effective={micro_bs * accum})"]
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            peak_alloc = torch.cuda.max_memory_allocated() / 1024 ** 2  # MiB
            peak_resvd = torch.cuda.max_memory_reserved() / 1024 ** 2  # MiB
            cur_alloc = torch.cuda.memory_allocated() / 1024 ** 2
            cur_resvd = torch.cuda.memory_reserved() / 1024 ** 2
        lines.append(f"  cuda:{i}  peak_alloc={peak_alloc:.0f} MiB, peak_reserved={peak_resvd:.0f} MiB | "
                     f"now_alloc={cur_alloc:.0f} MiB, now_reserved={cur_resvd:.0f} MiB")
    logging.info("\n".join(lines))


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5,
                             num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end,
                            num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


# ---------- 可复现：DataLoader worker 子种子 ----------
def _make_dataloader_seed_fn(base_seed: int):
    """
    返回给 DataLoader 的 worker_init_fn。每个 worker 会用稳定子种子初始化
    numpy/random，从而保证 shuffle、增强等完全可复现。
    """

    def _init_fn(worker_id: int):
        worker_seed = (base_seed + worker_id) % (2 ** 32)
        np.random.seed(worker_seed)
        import random as _py_random
        _py_random.seed(worker_seed)

    return _init_fn


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        # --- [FIX 2/3] Initialize LPIPS model ---
        self.lpips_fn = None
        if lpips is not None:
            logging.info("Initializing LPIPS metric (AlexNet)...")
            self.lpips_fn = lpips.LPIPS(net='alex').to(self.device).eval()
        else:
            logging.warning(
                "LPIPS not installed. Skipping LPIPS validation. "
                "Run `pip install lpips` to enable."
            )
        # --- End FIX ---

    # ------------------------------- TRAINING -------------------------------
    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger

        # ====== 数据集 & 可复现的 DataLoader 配置 ======
        dataset, test_dataset = get_dataset(args, config)

        g_train = torch.Generator()
        g_train.manual_seed(int(args.seed))
        worker_init = _make_dataloader_seed_fn(int(args.seed))

        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True,
            generator=g_train,
            worker_init_fn=worker_init,
            drop_last=True,
        )

        model = Model(config).to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        # --- [SCHEDULER FIX 2/5] Define Scheduler (after optimizer) ---
        try:
            total_steps = self.config.optim.total_steps
            warmup_steps = self.config.optim.warmup_steps
        except AttributeError:
            raise ValueError("请在 .yml 文件的 'optim' 部分添加 'total_steps' 和 'warmup_steps'")

        # 读取最小学习率，如果没有定义，默认为 1e-6
        min_lr = getattr(self.config.optim, 'min_lr', 1e-6)

        logging.info(
            f"[Scheduler] Initializing: Warmup {warmup_steps} steps, then Cosine Decay {total_steps - warmup_steps} steps (min_lr={min_lr}).")

        # 1. 定义 Warmup (从 1% -> 100% 的最大学习率)
        scheduler_warmup = LinearLR(optimizer,
                                    start_factor=0.01,
                                    end_factor=1.0,
                                    total_iters=warmup_steps)

        # 2. 定义 Decay (从 100% -> min_lr)
        decay_iters = total_steps - warmup_steps
        scheduler_decay = CosineAnnealingLR(optimizer,
                                            T_max=decay_iters,
                                            eta_min=min_lr)

        # 3. 组合
        scheduler = SequentialLR(optimizer,
                                 schedulers=[scheduler_warmup, scheduler_decay],
                                 milestones=[warmup_steps])  # 在 warmup_steps 步时切换
        # --- End SCHEDULER FIX ---

        ema_helper = None
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

            # 统一超参类型，避免 yaml 读成字符串
            for pg in optimizer.param_groups:
                pg["lr"] = float(self.config.optim.lr)
                pg["eps"] = float(self.config.optim.eps)
                pg["weight_decay"] = float(self.config.optim.weight_decay)
                beta1 = float(self.config.optim.beta1)
                pg["betas"] = (beta1, 0.999)

            start_epoch = states[2]
            step = states[3]

            sch_idx = -1  # 寻找 scheduler state 的索引
            if self.config.model.ema:
                if len(states) > 4:
                    ema_helper.load_state_dict(states[4])
                if len(states) > 5:  # [model, optim, epoch, step, ema, scheduler]
                    sch_idx = 5
            else:
                if len(states) > 4:  # [model, optim, epoch, step, scheduler]
                    sch_idx = 4

            # --- [SCHEDULER FIX 3/5] Load Scheduler State ---
            if sch_idx != -1 and states[sch_idx] is not None:
                logging.info(f"Loading scheduler state from checkpoint (index {sch_idx})...")
                scheduler.load_state_dict(states[sch_idx])
            else:
                logging.warning("Scheduler state not found in checkpoint or is None. Initializing new scheduler.")
                # (如果没找到，会使用上面新定义的 scheduler)
            # --- End SCHEDULER FIX ---

        # ====== 梯度累积 & AMP ======
        accum = int(getattr(self.config.training, "accum_steps", 1))
        assert accum >= 1
        use_amp = bool(getattr(self.config.training, "amp", True))
        scaler = GradScaler(enabled=use_amp)

        optimizer.zero_grad(set_to_none=True)

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            micro = 0  # micro step 计数（用于累积）

            for i, batch in enumerate(train_loader):
                # 在一个“有效大batch”的窗口开始时重置峰值统计
                if micro % accum == 0:
                    _reset_peak_all_gpus()
                # 兼容 (cond, x0) 或 (cond, x0, meta) / dict
                if isinstance(batch, (list, tuple)):
                    cond, x0 = batch[:2]
                elif isinstance(batch, dict):
                    cond, x0 = batch["cond"], batch["gt"]
                else:
                    raise TypeError(f"Unexpected batch type: {type(batch)}")

                n = x0.size(0)
                data_time += time.time() - data_start
                model.train()

                cond = cond.to(self.device, non_blocking=True).float()
                x0 = x0.to(self.device, non_blocking=True).float()

                # *** NOTE: data_transform is applied HERE ***
                cond = data_transform(self.config, cond)
                x0 = data_transform(self.config, x0)
                # *** cond and x0 are now in [-1, 1] range ***

                # 采样噪声与时间步（注意：batch 建议设为偶数以配合对称 t）
                e = torch.randn_like(x0)
                b = self.betas
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,),
                    device=self.device
                )
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                # -------- 前向 + 累积反传（AMP） --------
                with autocast(enabled=use_amp):
                    loss = loss_registry["simple_cond"](model, x0, cond, t, e, b)

                # 保留最近一次loss用于日志（按优化步记录）
                last_loss = float(loss.detach().item())

                # 累积反传
                scaler.scale(loss / accum).backward()
                micro += 1

                # 仅在累积满时做一次优化器 step（= 有效大batch）
                if micro % accum == 0:
                    # 先反缩放梯度方便裁剪
                    scaler.unscale_(optimizer)
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.optim.grad_clip
                        )
                    except Exception:
                        pass

                    # AMP step
                    scaler.step(optimizer)
                    scaler.update()

                    # --- [SCHEDULER FIX 4/5] Step the Scheduler (after optimizer step) ---
                    scheduler.step()
                    # --- End SCHEDULER FIX ---

                    optimizer.zero_grad(set_to_none=True)

                    # EMA 更新
                    if ema_helper is not None:
                        ema_helper.update(model)

                    # 只在“优化步”上递增 global step / 记录 / 存档 / 预览
                    step += 1

                    # 日志（以优化步为横轴）
                    # 额外记录学习率
                    current_lr = optimizer.param_groups[0]['lr']
                    tb_logger.add_scalar("loss", last_loss, global_step=step)
                    tb_logger.add_scalar("lr", current_lr, global_step=step)

                    logging.info(
                        f"step: {step}, loss: {last_loss:.6f}, lr: {current_lr:.2e}, data time: {data_time / (i + 1):.4f}"
                    )

                    # ---------- 存 ckpt ----------
                    # *** NOTE: config.training.snapshot_freq (2000) is used here ***
                    if step % self.config.training.snapshot_freq == 0 or step == 1:

                        # --- [SCHEDULER FIX 5/5] Save Scheduler State ---
                        states = [
                            model.state_dict(),
                            optimizer.state_dict(),
                            epoch,
                            step
                        ]
                        if self.config.model.ema:
                            states.append(ema_helper.state_dict())

                        states.append(scheduler.state_dict())  # 添加 scheduler 状态
                        # --- End SCHEDULER FIX ---

                        torch.save(states, os.path.join(self.args.log_path, f"ckpt_{step}.pth"))
                        torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                    # ---------- 预览 x0（每500优化步） ----------
                    # (This block is fine, it uses the [-1, 1] `cond` and `x0` from the train loop)
                    if step % 500 == 0:
                        model.eval()
                        with torch.no_grad():
                            sample_cond, sample_target = cond[:1], x0[:1]
                            t_sample = torch.randint(0, self.num_timesteps, (1,), device=self.device)
                            e_sample = torch.randn_like(sample_target)
                            a = (1 - self.betas).cumprod(dim=0).index_select(0, t_sample).view(-1, 1, 1, 1)
                            x_t = sample_target * a.sqrt() + e_sample * (1 - a).sqrt()
                            eps_pred = model(torch.cat([x_t, sample_cond], dim=1), t_sample.float())
                            x0_hat = (x_t - (1 - a).sqrt() * eps_pred) / a.sqrt()
                            x0_hat = x0_hat.clamp(-1, 1)

                            cond_vis = ((sample_cond.clamp(-1, 1) + 1) / 2).detach().cpu()
                            pred_vis = ((x0_hat + 1) / 2).detach().cpu()
                            target_vis = ((sample_target.clamp(-1, 1) + 1) / 2).detach().cpu()

                            comp = torch.cat(
                                [cond_vis[:, :3],
                                 pred_vis.repeat(1, 3, 1, 1),
                                 target_vis.repeat(1, 3, 1, 1)],
                                dim=3
                            )
                            save_path = os.path.join(self.args.exp, f"preview_{step:06d}.png")
                            tvu.save_image(comp, save_path)
                            logging.info(f"[Preview/x0] saved {save_path}")
                        model.train()

                    # --- [FIX 3/3] CORRECTED VALIDATION (PSNR/SSIM/LPIPS) ---
                    # Use validation_freq from config, and run on step > 0
                    if step % config.training.validation_freq == 0 and step > 0:

                        # (1) DDIM 10-step qualitative preview (Fast)
                        # (This block is fine, it uses the [-1, 1] `cond` from the train loop)
                        model.eval()
                        with torch.no_grad():
                            sample_cond = cond[:1]
                            nC, C, H, W = 1, self.config.data.channels, self.config.data.image_size, self.config.data.image_size
                            x = torch.randn(nC, C, H, W, device=self.device)
                            T = self.num_timesteps
                            seq = [int(s) for s in (np.linspace(0, np.sqrt(T * 0.8), 10) ** 2)]
                            xs = generalized_steps(x, seq, model, self.betas, eta=0.0, cond=sample_cond)
                            x_gen = xs[0][-1].clamp(-1, 1)

                            cond_vis = ((sample_cond.clamp(-1, 1) + 1) / 2).detach().cpu()
                            gen_vis = ((x_gen + 1) / 2).detach().cpu()
                            comp = torch.cat([cond_vis[:, :3], gen_vis.repeat(1, 3, 1, 1)], dim=3)
                            save_path = os.path.join(self.args.exp, f"preview_ddim10_{step:06d}.png")
                            tvu.save_image(comp, save_path)
                            logging.info(f"[Preview/DDIM10] saved {save_path}")
                        model.train()  # Back to train mode

                        # (2) Full quantitative validation (PSNR/SSIM/LPIPS)
                        # This runs a full DDIM sample (e.g., 50 steps) for N validation images
                        logging.info(f"Running full validation for step {step}...")
                        model.eval()
                        g_val = torch.Generator();
                        g_val.manual_seed(int(self.args.seed) + 12345)
                        val_loader = data.DataLoader(
                            test_dataset, batch_size=2, shuffle=False,
                            num_workers=config.data.num_workers, pin_memory=True,
                            generator=g_val,
                            worker_init_fn=_make_dataloader_seed_fn(int(self.args.seed) + 12345),
                            drop_last=False,
                        )

                        psnr_list, ssim_list, lpips_list = [], [], []
                        H = W = int(self.config.data.image_size)

                        # Use sampling settings from config for validation
                        val_timesteps = int(getattr(self.config.sampling, "timesteps", 50))
                        val_eta = float(getattr(self.config.sampling, "eta", 0.0))
                        skip = max(1, self.num_timesteps // val_timesteps)
                        seq = list(range(0, self.num_timesteps, skip))

                        with torch.no_grad():
                            for vb in val_loader:
                                if isinstance(vb, (list, tuple)):
                                    val_cond, val_target = vb[:2]
                                elif isinstance(vb, dict):
                                    val_cond, val_target = vb["cond"], vb["gt"]
                                else:
                                    raise TypeError(f"Unexpected val batch type: {type(vb)}")

                                # *** CRITICAL FIX: Apply data_transform to val data ***
                                # val_cond/val_target are [0, 1], model needs [-1, 1]
                                val_cond_n = data_transform(self.config, val_cond.to(self.device))
                                val_target_n = data_transform(self.config, val_target.to(self.device))
                                # *** val_cond_n and val_target_n are now [-1, 1] ***

                                n_val = val_cond_n.size(0)
                                x = torch.randn(
                                    n_val,
                                    self.config.data.channels,
                                    H, W, device=self.device
                                )

                                # Run full DDIM sampling
                                xs = generalized_steps(
                                    x, seq, model, self.betas,
                                    eta=val_eta, cond=val_cond_n
                                )
                                x0_hat = xs[0][-1].clamp(-1, 1)  # Final prediction, [-1, 1]

                                # Calc LPIPS (on [-1, 1] tensors)
                                if self.lpips_fn is not None:
                                    lpips_score = self.lpips_fn(x0_hat, val_target_n).mean()
                                    lpips_list.append(lpips_score.item())

                                # Calc PSNR/SSIM (on uint8 numpy arrays [0, 255])
                                for k in range(n_val):
                                    # Get single image, channel 0 (assuming grayscale FFA)
                                    pred_k = x0_hat[k, 0]
                                    gt_k = val_target_n[k, 0]

                                    # Convert from [-1, 1] tensor to [0, 255] numpy
                                    pred_np = ((pred_k.detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
                                    gt_np = ((gt_k.detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)

                                    psnr_list.append(psnr(gt_np, pred_np, data_range=255))
                                    ssim_list.append(ssim(gt_np, pred_np, data_range=255))

                                # Limit validation to ~10 images (5 batches of size 2)
                                if len(psnr_list) >= 10:
                                    break

                        avg_psnr = float(np.mean(psnr_list))
                        avg_ssim = float(np.mean(ssim_list))

                        tb_logger.add_scalar("val/psnr", avg_psnr, step)
                        tb_logger.add_scalar("val/ssim", avg_ssim, step)
                        log_msg = f"[Val] Step {step}: PSNR={avg_psnr:.2f} SSIM={avg_ssim:.3f}"

                        if self.lpips_fn is not None and lpips_list:
                            avg_lpips = float(np.mean(lpips_list))
                            tb_logger.add_scalar("val/lpips", avg_lpips, step)
                            log_msg += f" LPIPS={avg_lpips:.4f}"

                        logging.info(log_msg)
                        self.config.tb_logger.flush()

                        model.train()  # IMPORTANT: Set model back to train mode
                    # --- End FIX ---

                # 继续下一 micro step
                data_start = time.time()

            # -------- 处理 epoch 末尾“不满accum”的残留梯度 --------
            if micro % accum != 0:
                scaler.unscale_(optimizer)
                try:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
                except Exception:
                    pass
                scaler.step(optimizer)
                scaler.update()

                # --- [SCHEDULER FIX 5/5] Step the Scheduler (for epoch-end stragglers) ---
                scheduler.step()
                # --- End SCHEDULER FIX ---

                optimizer.zero_grad(set_to_none=True)
                if ema_helper is not None:
                    ema_helper.update(model)
                step += 1
                # 打印本次“有效大batch”的峰值显存
                _log_peak_all_gpus(step, config.training.batch_size, accum)

                current_lr = optimizer.param_groups[0]['lr']
                tb_logger.add_scalar("loss", last_loss, global_step=step)
                tb_logger.add_scalar("lr", current_lr, global_step=step)

    # =================================================================
    #
    # SAMPLING / EVALUATION (No changes needed below this line)
    #
    # =================================================================

    # ------------------------------- SAMPLING -------------------------------
    def sample(self):
        model = Model(self.config)
        ckpt_file = (
            os.path.join(self.args.log_path, "ckpt.pth")
            if getattr(self.config.sampling, "ckpt_id", None) is None
            else os.path.join(
                self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
            )
        )
        states = torch.load(ckpt_file, map_location=self.device)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        ema_helper = None
        # 检查 EMA state，同时兼容新的 scheduler state
        ema_idx = -1
        if self.config.model.ema:
            if len(states) > 4:  # [model, optim, epoch, step, ema, ...]
                ema_idx = 4

        if ema_idx != -1 and states[ema_idx] is not None:
            logging.info(f"Loading EMA state from checkpoint (index {ema_idx})...")
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[ema_idx])
            ema_helper.ema(model)
        else:
            logging.info("No EMA state found or EMA disabled. Using non-EMA weights.")

        model.eval()

        if getattr(self.args, "paired", False):
            self.sample_paired(model)
        elif getattr(self.args, "fid", False):
            self.sample_fid(model)
        elif getattr(self.args, "sequence", False):
            self.sample_sequence(model)
        else:
            self.sample_paired_triptych(model)

    def sample_image(self, x, model, last=True, **kwargs):
        if self.args.sample_type == "generalized":
            from functions.denoising import generalized_steps
            skip = self.num_timesteps // self.args.timesteps
            seq = range(0, self.num_timesteps, skip)
            xs = generalized_steps(
                x, seq, model, self.betas, eta=self.args.eta, cond=kwargs.get("cond", None)
            )
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            from functions.denoising import ddpm_steps
            skip = self.num_timesteps // self.args.timesteps
            seq = range(0, self.num_timesteps, skip)
            x = ddpm_steps(x, seq, model, self.betas, cond=kwargs.get("cond", None))
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def sample_paired(self, model):
        config = self.config
        _, test_dataset = get_dataset(self.args, self.config)

        g_eval = torch.Generator()
        g_eval.manual_seed(int(self.args.seed) + 2222)

        loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=True,
            generator=g_eval,
            worker_init_fn=_make_dataloader_seed_fn(int(self.args.seed) + 2222),
        )
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        model.eval()
        with torch.no_grad():
            for cond, _ in tqdm.tqdm(loader, desc="Paired sampling (CFP→FFA)"):
                cond = data_transform(config, cond.to(self.device))
                n = cond.size(0)
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                xs = self.sample_image(x, model, last=True, cond=cond)
                x = inverse_data_transform(config, xs)
                for i in range(n):
                    tvu.save_image(
                        x[i],
                        os.path.join(self.args.image_folder, f"{img_id}.png"),
                    )
                    img_id += 1

    def sample_paired_triptych(self, model):
        """
        保存三联图：左=数据集原始CFP，中=模型预测，右=数据集原始FFA
        仅做尺寸对齐，不做其他处理。
        需要 Dataset __getitem__ 返回 (cond_t, gt_t, meta)，其中：
          meta = {"cond_path": <CFP原图绝对路径>, "gt_path": <FFA原图绝对路径>, "id": <样本ID>}
        """

        os.makedirs(self.args.image_folder, exist_ok=True)

        train_dataset, test_dataset = get_dataset(self.args, self.config)

        g_eval = torch.Generator()
        g_eval.manual_seed(int(self.args.seed) + 3333)

        loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=getattr(self.config.data, "num_workers", 0),
            pin_memory=True,
            generator=g_eval,
            worker_init_fn=_make_dataloader_seed_fn(int(self.args.seed) + 3333),
        )

        timesteps = getattr(self.args, "timesteps", None)
        skip_type = getattr(self.args, "skip_type", "uniform")

        if timesteps is None or timesteps <= 0 or timesteps > self.num_timesteps:
            seq = list(range(self.num_timesteps))
        else:
            if skip_type == "uniform":
                skip = max(1, self.num_timesteps // timesteps)
                seq = list(range(0, self.num_timesteps, skip))[:timesteps]
            elif skip_type == "quad":
                import numpy as np
                seq = (np.linspace(0, (self.num_timesteps * 0.8) ** 0.5, timesteps) ** 2).astype(int).tolist()
            else:
                raise NotImplementedError(f"Unknown skip_type: {skip_type}")

        def _first(x):
            return x[0] if isinstance(x, (list, tuple)) else x

        model.eval()
        H = W = int(self.config.data.image_size)
        eta = float(getattr(self.args, "eta", 0.0))

        with torch.no_grad():
            for idx, batch in enumerate(tqdm.tqdm(loader, desc="Triptych sampling")):
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    cond, gt, meta = batch
                else:
                    cond, gt = batch
                    meta = {}

                # cond is [0, 1] from dataloader
                cond = cond.to(self.device)

                x = torch.randn(
                    1,
                    self.config.data.channels,
                    self.config.data.image_size,
                    self.config.data.image_size,
                    device=self.device,
                )

                # model needs cond in [-1, 1]
                cond_n = data_transform(self.config, cond)
                xs = generalized_steps(x, seq, model, self.betas, eta=eta, cond=cond_n)

                pred = xs[1][-1].clamp(-1, 1)  # [1,C,H,W], in [-1, 1] range

                cond_path = _first(meta.get("cond_path", ""))
                gt_path = _first(meta.get("gt_path", ""))
                name_id = _first(meta.get("id", f"{idx:06d}"))

                # Open raw images for triptych
                cfp_raw = Image.open(cond_path).convert("RGB")
                ffa_raw = Image.open(gt_path)
                if ffa_raw.mode != "RGB":
                    ffa_raw = ffa_raw.convert("RGB")

                # Convert prediction from [-1, 1] tensor to PIL image
                pred_vis = ((pred + 1) / 2).clamp(0, 1).squeeze(0).cpu()
                if pred_vis.shape[0] == 1:
                    pred_vis = pred_vis.repeat(3, 1, 1)
                pred_pil = TF.to_pil_image(pred_vis)

                # Resize all to match config size
                cfp_img = cfp_raw.resize((W, H), Image.BILINEAR)
                pred_img = pred_pil.resize((W, H), Image.BILINEAR)
                ffa_img = ffa_raw.resize((W, H), Image.BILINEAR)

                triptych = Image.new("RGB", (W * 3, H), (0, 0, 0))
                triptych.paste(cfp_img, (0, 0))
                triptych.paste(pred_img, (W, 0))
                triptych.paste(ffa_img, (2 * W, 0))

                rel = str(name_id).replace("\\", "/")
                if not rel.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
                    rel += ".png"

                out_path_trip = os.path.join(self.args.image_folder, rel)
                out_path_pred = os.path.join(self.args.image_folder, "pred", rel)

                os.makedirs(os.path.dirname(out_path_trip), exist_ok=True)
                os.makedirs(os.path.dirname(out_path_pred), exist_ok=True)

                triptych.save(out_path_trip)
                pred_img.save(out_path_pred)

                if idx % 50 == 0:
                    logging.info(f"[Eval] saved triptych={out_path_trip}, pred={out_path_pred}")

    def sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        total_n_samples = 50000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size
        with torch.no_grad():
            for _ in tqdm.tqdm(
                    range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                # NOTE: FID sampling should be UNCONDITIONAL. 
                # This code assumes conditional sampling if cond is not None.
                # For a true FID score (unconditional), you might need to pass cond=None
                # or have a separate model config. 
                # Assuming here the model can handle cond=None if needed.
                cond = None
                # If your model is *always* conditional, you can't compute a standard FID.
                # You would compute a "conditional FID" (cIFID) or "Frechet Paired Distance" (FPD)
                # which requires a different sampling setup (e.g., using test set conditions).
                # The current `sample_image` call (without `cond`) will likely fail 
                # if `generalized_steps` expects `cond`.
                #
                # *** Assuming unconditional generation for FID as is standard ***
                # If your `generalized_steps` fails with cond=None, this must be adjusted.

                x = self.sample_image(x, model, cond=cond)
                x = inverse_data_transform(config, x)
                for i in range(n):
                    tvu.save_image(
                        x[i],
                        os.path.join(self.args.image_folder, f"{img_id}.png"),
                    )
                    img_id += 1

    def sample_sequence(self, model):
        config = self.config
        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False, cond=None)  # Assuming unconditional
        x = [inverse_data_transform(config, y) for y in x]
        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(x[i][j],
                               os.path.join(self.args.image_folder, f"{j}_{i}.png"))

    def test(self):
        pass