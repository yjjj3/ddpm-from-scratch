"""FID evaluation for the DDIM step-count experiment.

Usage (Colab or local, after training via ddpm_mnist.py):
    !pip install clean-fid

    from fid_eval import prepare_real_images, run_fid_experiment, run_seed_check
    prepare_real_images()                 # once
    results = run_fid_experiment()        # main sweep
    results = run_seed_check(seed=42)     # robustness check (optional)

Time estimates (A100, batch 500, 10k images per setting):
    ~1 min per 10-step config; ~8 min for 100 steps; ~80 min for 1000 steps.
"""

import os
import json
import shutil

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from cleanfid import fid

from ddpm_mnist import CFG, UNet, Diffusion
from ddim import ddim_sample

# ----------------------------- config -----------------------------
STEP_LIST = [200, 100, 50, 20, 10, 5, 3, 2]   # append 1000 when budget allows
NUM_GEN = 10_000        # images per setting (>=10k recommended for FID)
GEN_BATCH = 500
REAL_DIR = "/content/fid_real"      # local SSD: much faster than Drive
FAKE_ROOT = "/content/fid_fake"
RESULTS_JSON = os.path.join(CFG.ckpt_dir, "fid_results.json")


def _load_ema_model():
    diffusion = Diffusion(CFG)
    model = UNet(CFG).to(CFG.device)
    ck = torch.load(os.path.join(CFG.ckpt_dir, "latest.pt"), map_location=CFG.device)
    model.load_state_dict(ck["ema"])
    model.eval()
    return model, diffusion


def _load_results():
    return json.load(open(RESULTS_JSON)) if os.path.exists(RESULTS_JSON) else {}


def _save_results(results):
    json.dump(results, open(RESULTS_JSON, "w"), indent=2)


# ------------------------ real image prep ------------------------
def prepare_real_images():
    """Save the 10k MNIST test images as 32x32 PNGs (same format as samples)."""
    if os.path.exists(REAL_DIR) and len(os.listdir(REAL_DIR)) >= 10_000:
        print("real images already prepared")
        return
    os.makedirs(REAL_DIR, exist_ok=True)
    tf = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])
    ds = datasets.MNIST("./data", train=False, download=True, transform=tf)
    for i in range(len(ds)):
        x, _ = ds[i]
        save_image(x, os.path.join(REAL_DIR, f"{i:05d}.png"))
        if (i + 1) % 2000 == 0:
            print(f"  saved {i + 1}/10000")
    print("real images done")


# ------------------------ generation + FID ------------------------
@torch.no_grad()
def _generate_images(model, diffusion, num_steps, out_dir, seed, tag):
    os.makedirs(out_dir, exist_ok=True)
    if len(os.listdir(out_dir)) >= NUM_GEN:
        print(f"  [{tag}] already generated, skip")
        return
    torch.manual_seed(seed)  # shared noise sequence across settings
    idx = 0
    while idx < NUM_GEN:
        n = min(GEN_BATCH, NUM_GEN - idx)
        x = ddim_sample(model, diffusion,
                        (n, CFG.channels, CFG.image_size, CFG.image_size),
                        CFG.device, num_steps=num_steps, eta=0.0)
        x = (x.clamp(-1, 1) + 1) / 2
        for j in range(n):
            save_image(x[j], os.path.join(out_dir, f"{idx + j:05d}.png"))
        idx += n
        print(f"  [{tag}] {idx}/{NUM_GEN}")


def _eval_one(model, diffusion, num_steps, seed, key, results):
    """Generate + score one setting; results saved to Drive immediately."""
    out_dir = os.path.join(FAKE_ROOT, f"steps_{key}")
    print(f"=== {key} ===")
    _generate_images(model, diffusion, num_steps, out_dir, seed, key)
    score = fid.compute_fid(REAL_DIR, out_dir)
    results[key] = score
    _save_results(results)
    print(f"[{key}] FID = {score:.2f}\n")
    shutil.rmtree(out_dir)  # free disk; score is saved


def run_fid_experiment():
    """Main sweep over STEP_LIST with the default seed (0).

    Resume-safe: completed settings are skipped on re-run.
    """
    model, diffusion = _load_ema_model()
    results = _load_results()
    for n_steps in STEP_LIST:
        key = str(n_steps)
        if key in results:
            print(f"[{key}] FID already computed: {results[key]:.2f}, skip")
            continue
        _eval_one(model, diffusion, n_steps, seed=0, key=key, results=results)
    print("all done:", results)
    return results


def run_seed_check(seed, step_list=(200, 20)):
    """Re-run selected settings with a different seed to verify FID ordering."""
    model, diffusion = _load_ema_model()
    results = _load_results()
    for n_steps in step_list:
        key = f"{n_steps}_seed{seed}"
        if key in results:
            print(f"[{key}] already computed: {results[key]:.2f}, skip")
            continue
        _eval_one(model, diffusion, n_steps, seed=seed, key=key, results=results)
    return results
