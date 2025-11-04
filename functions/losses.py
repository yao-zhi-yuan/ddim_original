# functions/losses.py
import torch
import torch.nn.functional as F

def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor,
                          keepdim=False):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    out = model(x, t.float())
    err = (e - out).pow(2).mean(dim=(1, 2, 3))     # ✅ mean 而非 sum
    return err if keepdim else err.mean()

def noise_estimation_loss_cond(model,
                               x0: torch.Tensor,      # target (FFA)
                               cond: torch.Tensor,    # condition (CFP)
                               t: torch.LongTensor,
                               e: torch.Tensor,
                               b: torch.Tensor,
                               keepdim=False):
    # 1) 构造 x_t
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x_t = x0 * a.sqrt() + e * (1.0 - a).sqrt()

    # 2) 拼接条件并前向
    model_in = torch.cat([x_t, cond], dim=1)
    out = model(model_in, t.float())                # 预测噪声 (same shape as e)

    # 3) 数值检查（调试期保留，稳定后可注释）
    if not torch.isfinite(out).all():
        bad = (~torch.isfinite(out)).float().mean().item()
        raise RuntimeError(f"Model output contains non-finite values (ratio={bad:.6f}). "
                           f"Check attn_resolutions / channels / LR.")

    # 4) MSE (mean) 而非 sum
    err = (e - out).pow(2).mean(dim=(1, 2, 3))
    return err if keepdim else err.mean()

loss_registry = {
    'simple': noise_estimation_loss,
    'simple_cond': noise_estimation_loss_cond,
}
