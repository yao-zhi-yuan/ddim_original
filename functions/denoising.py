import torch


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


# 在文件顶部保持 compute_alpha 不变

@torch.no_grad()
def generalized_steps(x, seq, model, betas, eta=0.0, cond=None):
    """
    x:     初始噪声 [B,C,H,W]
    seq:   时间步列表
    model: 预测噪声ε的模型
    betas: 噪声调度
    eta:   DDIM 退火系数
    cond:  条件图 [B,3,H,W]（已归一化到[-1,1]）
    """
    device = x.device
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(0)
    sqrt_alphas_cumprod = alphas_cumprod.sqrt()
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()

    xs, x0s = [x], []
    for i in range(len(seq) - 1, -1, -1):
        t = torch.full((x.size(0),), fill_value=seq[i], device=device, dtype=torch.long)
        a_t = alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_at = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_at = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        # ✅ 加条件输入：把 x_t 和 cond 拼在通道维
        if cond is None:
            eps = model(x, t.float())
        else:
            eps = model(torch.cat([x, cond], dim=1), t.float())

        # 预测 x0
        x0_hat = (x - sqrt_one_minus_at * eps) / sqrt_at
        x0_hat = x0_hat.clamp(-1, 1)
        x0s.append(x0_hat)

        # 计算下一个时间步
        if i == 0:
            x = x0_hat
            xs.append(x)
            break

        t_prev = torch.full((x.size(0),), fill_value=seq[i-1], device=device, dtype=torch.long)
        a_prev = alphas_cumprod[t_prev].view(-1, 1, 1, 1)
        sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
        dir_xt = torch.sqrt(1 - a_prev - sigma**2) * eps
        noise = sigma * torch.randn_like(x)
        x = torch.sqrt(a_prev) * x0_hat + dir_xt + noise
        xs.append(x)

    return xs, x0s   # ✅ 返回 (x_t 序列, x0_hat 序列)


def ddpm_steps(x, seq, model, b, **kwargs):
    cond = kwargs.get("cond", None)
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs, x0_preds = [x], []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')
            # ↓ 关键：拼接条件
            x_in = torch.cat([x, cond.to(x.device)], dim=1) if cond is not None else x
            e = model(x_in, t.float())
            x0_from_e = (1.0/at).sqrt()*x - (1.0/at - 1).sqrt()*e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = ((atm1.sqrt()*beta_t)*x0_from_e + ((1 - beta_t).sqrt()*(1 - atm1))*x) / (1.0 - at)
            noise = torch.randn_like(x)
            mask = (1 - (t == 0).float()).view(-1,1,1,1)
            logvar = beta_t.log()
            sample = mean_eps + mask * torch.exp(0.5*logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds
