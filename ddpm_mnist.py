"""
DDPM from scratch on MNIST — Week 1 skeleton for Colab
=======================================================
設計目標：
  1. 從零實作 DDPM（Ho et al. 2020）：linear noise schedule + epsilon-prediction loss
  2. 精簡但完整的 U-Net（time embedding、residual block、attention）
  3. Colab 實務：checkpoint 存 Google Drive、EMA、fp16 混合精度、斷點續訓

Colab 使用方式：
  # Cell 1: 掛載 Drive
  from google.colab import drive
  drive.mount('/content/drive')

  # Cell 2: 貼上本檔全部內容（或 %run ddpm_mnist.py）
  # 然後呼叫：
  train()                    # 開始/繼續訓練
  sample_and_show(n=64)      # 從最新 checkpoint 生成圖片
  visualize_denoising()      # 去噪過程視覺化（README 素材）

T4 約 1-2 小時可看到清楚的數字；A100 更快。
"""

import os, math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

# ----------------------------- Config -----------------------------
class CFG:
    image_size   = 32          # MNIST 原生 28x28，pad 成 32 方便 U-Net 下採樣
    channels     = 1
    T            = 1000        # diffusion timesteps
    beta_start   = 1e-4
    beta_end     = 0.02
    base_ch      = 64          # U-Net 基礎通道數
    ch_mults     = (1, 2, 2)   # 32 -> 16 -> 8
    batch_size   = 128
    lr           = 2e-4
    total_steps  = 30_000      # T4 約 1.5 hr；先跑 5000 步就能看出雛形
    ema_decay    = 0.999
    log_every    = 200
    ckpt_every   = 2_000
    # Colab: 存到 Drive 才不會斷線就消失；本機測試則存當前目錄
    ckpt_dir     = ("/content/drive/MyDrive/ddpm_mnist"
                    if os.path.exists("/content/drive/MyDrive")
                    else "./ddpm_mnist_ckpt")
    device       = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CFG.ckpt_dir, exist_ok=True)

# ------------------------ Diffusion schedule ------------------------
class Diffusion:
    """封裝 forward process 的所有預計算量。
    q(x_t | x_0) = N( sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I )
    """
    def __init__(self, cfg: CFG):
        self.T = cfg.T
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(cfg.device)
        self.alphas = alphas.to(cfg.device)
        self.alpha_bars = alpha_bars.to(cfg.device)
        self.sqrt_ab = alpha_bars.sqrt().to(cfg.device)
        self.sqrt_1mab = (1 - alpha_bars).sqrt().to(cfg.device)

    def q_sample(self, x0, t, noise):
        """一步加噪到任意 t（訓練用）。t: (B,) long tensor"""
        sab = self.sqrt_ab[t].view(-1, 1, 1, 1)
        s1mab = self.sqrt_1mab[t].view(-1, 1, 1, 1)
        return sab * x0 + s1mab * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, device, return_trajectory=False):
        """DDPM ancestral sampling（Algorithm 2）。"""
        x = torch.randn(shape, device=device)
        traj = []
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            eps = model(x, t)
            beta = self.betas[i]
            alpha = self.alphas[i]
            ab = self.alpha_bars[i]
            # 均值：1/sqrt(alpha) * (x - beta/sqrt(1-ab) * eps)
            mean = (x - beta / (1 - ab).sqrt() * eps) / alpha.sqrt()
            if i > 0:
                x = mean + beta.sqrt() * torch.randn_like(x)
            else:
                x = mean
            if return_trajectory and i % (self.T // 10) == 0:
                traj.append(x.clone())
        return (x, traj) if return_trajectory else x

# ----------------------------- U-Net -----------------------------
def timestep_embedding(t, dim):
    """Sinusoidal embedding（同 Transformer 位置編碼）。t: (B,)"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class AttnBlock(nn.Module):
    """簡單 self-attention，只放在最低解析度（8x8）以控制成本。"""
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(B, C, H * W).transpose(1, 2)   # (B, HW, C)
        k = k.reshape(B, C, H * W)                   # (B, C, HW)
        v = v.reshape(B, C, H * W).transpose(1, 2)
        attn = torch.softmax(q @ k / math.sqrt(C), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj(out)

class UNet(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        ch = cfg.base_ch
        time_dim = ch * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(ch, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.base_ch = ch

        self.in_conv = nn.Conv2d(cfg.channels, ch, 3, padding=1)

        # Down path: 32(ch) -> 16(2ch) -> 8(2ch)
        chs = [ch * m for m in cfg.ch_mults]           # [64, 128, 128]
        self.down1 = ResBlock(chs[0], chs[0], time_dim)
        self.pool1 = nn.Conv2d(chs[0], chs[0], 3, stride=2, padding=1)
        self.down2 = ResBlock(chs[0], chs[1], time_dim)
        self.pool2 = nn.Conv2d(chs[1], chs[1], 3, stride=2, padding=1)
        self.down3 = ResBlock(chs[1], chs[2], time_dim)

        # Bottleneck with attention at 8x8
        self.mid1 = ResBlock(chs[2], chs[2], time_dim)
        self.attn = AttnBlock(chs[2])
        self.mid2 = ResBlock(chs[2], chs[2], time_dim)

        # Up path（skip connection 用 concat）
        self.up1 = ResBlock(chs[2] + chs[2], chs[1], time_dim)
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up2 = ResBlock(chs[1] + chs[1], chs[0], time_dim)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up3 = ResBlock(chs[0] + chs[0], chs[0], time_dim)

        self.out_norm = nn.GroupNorm(8, chs[0])
        self.out_conv = nn.Conv2d(chs[0], cfg.channels, 3, padding=1)

    def forward(self, x, t):
        temb = self.time_mlp(timestep_embedding(t, self.base_ch))
        h1 = self.down1(self.in_conv(x), temb)   # 32
        h2 = self.down2(self.pool1(h1), temb)    # 16
        h3 = self.down3(self.pool2(h2), temb)    # 8

        m = self.mid2(self.attn(self.mid1(h3, temb)), temb)

        u = self.up1(torch.cat([m, h3], dim=1), temb)
        u = self.up2(torch.cat([self.upsample1(u), h2], dim=1), temb)
        u = self.up3(torch.cat([self.upsample2(u), h1], dim=1), temb)
        return self.out_conv(F.silu(self.out_norm(u)))

# ----------------------------- EMA -----------------------------
class EMA:
    """Exponential Moving Average of weights — 取樣品質的關鍵，別省。"""
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1 - self.decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)

# --------------------------- Data ---------------------------
def get_loader(cfg: CFG):
    tf = transforms.Compose([
        transforms.Pad(2),                       # 28 -> 32
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),          # [-1, 1]
    ])
    ds = datasets.MNIST("./data", train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                      num_workers=2, pin_memory=True, drop_last=True)

# --------------------------- Train ---------------------------
def train(cfg: CFG = CFG):
    device = cfg.device
    diffusion = Diffusion(cfg)
    model = UNet(cfg).to(device)
    ema = EMA(model, cfg.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # ---- 斷點續訓 ----
    step = 0
    latest = os.path.join(cfg.ckpt_dir, "latest.pt")
    if os.path.exists(latest):
        ck = torch.load(latest, map_location=device)
        model.load_state_dict(ck["model"])
        ema.shadow.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"Resumed from step {step}")

    loader = get_loader(cfg)
    data_iter = iter(loader)
    model.train()
    running = 0.0

    while step < cfg.total_steps:
        try:
            x0, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x0, _ = next(data_iter)
        x0 = x0.to(device, non_blocking=True)

        t = torch.randint(0, cfg.T, (x0.size(0),), device=device)
        noise = torch.randn_like(x0)
        xt = diffusion.q_sample(x0, t, noise)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            pred = model(xt, t)
            loss = F.mse_loss(pred, noise)       # simplified loss（論文 Eq.14）

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        ema.update(model)

        running += loss.item()
        step += 1

        if step % cfg.log_every == 0:
            print(f"step {step:>6d} | loss {running / cfg.log_every:.4f}")
            running = 0.0

        if step % cfg.ckpt_every == 0 or step == cfg.total_steps:
            torch.save({"model": model.state_dict(),
                        "ema": ema.shadow.state_dict(),
                        "opt": opt.state_dict(),
                        "step": step}, latest)
            # 順手存一張樣本圖，肉眼追蹤訓練進度
            grid_path = os.path.join(cfg.ckpt_dir, f"sample_{step}.png")
            _save_samples(ema.shadow, diffusion, cfg, grid_path, n=16)
            model.train()
            print(f"  checkpoint + samples saved @ step {step}")

    print("Training done.")

@torch.no_grad()
def _save_samples(model, diffusion, cfg, path, n=16):
    model.eval()
    x = diffusion.p_sample_loop(model, (n, cfg.channels, cfg.image_size, cfg.image_size), cfg.device)
    x = (x.clamp(-1, 1) + 1) / 2
    save_image(make_grid(x, nrow=int(math.sqrt(n))), path)

# ------------------------ Inference utils ------------------------
@torch.no_grad()
def sample_and_show(n=64, cfg: CFG = CFG):
    """從最新 EMA checkpoint 生成並顯示（Colab 內用）。"""
    import matplotlib.pyplot as plt
    diffusion = Diffusion(cfg)
    model = UNet(cfg).to(cfg.device)
    ck = torch.load(os.path.join(cfg.ckpt_dir, "latest.pt"), map_location=cfg.device)
    model.load_state_dict(ck["ema"])
    model.eval()
    x = diffusion.p_sample_loop(model, (n, cfg.channels, cfg.image_size, cfg.image_size), cfg.device)
    x = (x.clamp(-1, 1) + 1) / 2
    grid = make_grid(x, nrow=int(math.sqrt(n))).cpu().permute(1, 2, 0)
    plt.figure(figsize=(8, 8)); plt.imshow(grid, cmap="gray"); plt.axis("off"); plt.show()

@torch.no_grad()
def visualize_denoising(n=8, cfg: CFG = CFG):
    """去噪軌跡視覺化：README / 簡報的招牌素材。"""
    import matplotlib.pyplot as plt
    diffusion = Diffusion(cfg)
    model = UNet(cfg).to(cfg.device)
    ck = torch.load(os.path.join(cfg.ckpt_dir, "latest.pt"), map_location=cfg.device)
    model.load_state_dict(ck["ema"])
    model.eval()
    _, traj = diffusion.p_sample_loop(
        model, (n, cfg.channels, cfg.image_size, cfg.image_size),
        cfg.device, return_trajectory=True)
    fig, axes = plt.subplots(n, len(traj), figsize=(len(traj) * 1.2, n * 1.2))
    for r in range(n):
        for c, snap in enumerate(traj):
            img = ((snap[r].clamp(-1, 1) + 1) / 2).cpu().squeeze()
            axes[r][c].imshow(img, cmap="gray"); axes[r][c].axis("off")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    train()
