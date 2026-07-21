"""DDIM sampler (Song et al. 2021) — works with any model trained via ddpm_mnist.py.

Key idea: reuse the same epsilon-prediction network, but replace the Markovian
ancestral sampling with a (optionally deterministic) non-Markovian update that
can skip timesteps. No retraining required.
"""

import torch


@torch.no_grad()
def ddim_sample(model, diffusion, shape, device, num_steps=50, eta=0.0):
    """DDIM sampling.

    Args:
        model: epsilon-prediction network, called as model(x, t)
        diffusion: Diffusion object from ddpm_mnist (provides alpha_bars, T)
        shape: output tensor shape, e.g. (B, 1, 32, 32)
        num_steps: number of sampling steps (uniform subsequence of [0, T-1])
        eta: 0 = fully deterministic; 1 = stochasticity equivalent to DDPM
    """
    ts = torch.linspace(diffusion.T - 1, 0, num_steps, dtype=torch.long)

    x = torch.randn(shape, device=device)
    for i in range(len(ts)):
        t = ts[i].item()
        t_prev = ts[i + 1].item() if i + 1 < len(ts) else -1
        ab_t = diffusion.alpha_bars[t]
        ab_prev = (diffusion.alpha_bars[t_prev] if t_prev >= 0
                   else torch.tensor(1.0, device=device))

        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps = model(x, t_batch)

        # 1. Estimate x0 from current x_t and predicted noise
        x0_pred = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)  # stability trick

        # 2. sigma controls stochasticity (0 when eta=0)
        sigma = eta * ((1 - ab_prev) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_prev).sqrt()

        # 3. Jump to t_prev
        dir_xt = (1 - ab_prev - sigma**2).sqrt() * eps
        x = ab_prev.sqrt() * x0_pred + dir_xt
        if eta > 0 and t_prev >= 0:
            x = x + sigma * torch.randn_like(x)
    return x
