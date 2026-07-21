"""Plotting utilities for the FID step-count experiment."""

import os
import json

import numpy as np
import matplotlib.pyplot as plt

from ddpm_mnist import CFG
from fid_eval import RESULTS_JSON


def plot_fid_curve(out_name="fid_curve_polished.png"):
    """Presentation-grade FID curve.

    - Main curve over all non-seed-suffixed settings
    - Error bars (mean +/- 1 std) where multiple seeds exist
    - Best point highlighted, annotation placed below the curve
    - Caption records experimental conditions so the figure is self-contained
    """
    all_results = json.load(open(RESULTS_JSON))

    # main curve: keys without a seed suffix
    main = {int(k): v for k, v in all_results.items() if "_seed" not in k}
    steps = sorted(main.keys())
    fids = [main[s] for s in steps]

    # aggregate multi-seed stats: e.g. "20", "20_seed42", "20_seed123"
    multi = {}
    for s in steps:
        vals = [v for k, v in all_results.items()
                if k == str(s) or k.startswith(f"{s}_seed")]
        if len(vals) > 1:
            multi[s] = (np.mean(vals), np.std(vals))

    best_step = min(main, key=main.get)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(steps, fids, marker="o", linewidth=2, color="#1f77b4",
            markersize=7, zorder=3)

    for s, (m, sd) in multi.items():
        ax.errorbar(s, m, yerr=sd, fmt="none", ecolor="#1f77b4",
                    capsize=4, zorder=4)

    ax.scatter([best_step], [main[best_step]], s=180, facecolors="none",
               edgecolors="#d62728", linewidths=2.5, zorder=5)
    ax.axvline(best_step, ls="--", color="#d62728", alpha=0.35, zorder=1)
    ax.annotate(f"best: {best_step} steps\nFID {main[best_step]:.1f}",
                (best_step, main[best_step]),
                textcoords="offset points", xytext=(20, 8),
                fontsize=10, color="#d62728", fontweight="bold")

    for s, f in zip(steps, fids):
        if s == best_step:
            continue
        ax.annotate(f"{f:.1f}", (s, f), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("DDIM sampling steps (log scale)", fontsize=11)
    ax.set_ylabel("FID (log scale, lower is better)", fontsize=11)
    ax.set_title("Quality-efficiency trade-off of DDIM sampling on MNIST",
                 fontsize=13)
    ax.grid(True, which="both", alpha=0.3)

    fig.text(0.5, -0.03,
             "Error bars: \u00b11 std over 3 random seeds (often smaller than "
             "markers). \u03b7 = 0 (deterministic DDIM); FID computed on 10k "
             "samples vs MNIST test set.",
             ha="center", fontsize=8.5, style="italic", color="0.35")

    out = os.path.join(CFG.ckpt_dir, out_name)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.show()
    print("saved to", out)
