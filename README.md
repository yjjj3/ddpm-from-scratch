# DDPM from Scratch + DDIM Clipping Ablation

From-scratch PyTorch DDPM on MNIST, with an empirical study of how clipping
and noise recomputation affect DDIM's sampling-step/quality relationship.

![Clipping ablation: FID and sampling time](assets/clipping_ablation.svg)

## Main findings

Across three **sampling seeds (0, 42, 123) using one EMA checkpoint**:

- Original clipping has its lowest measured FID at 20 steps.
- No intermediate clipping and clipping with noise recomputation both have
  their lowest measured FID at 50 steps.
- At 200 steps, recomputing noise after clipping reduces mean FID from
  **17.471 to 5.680**. Removing intermediate clipping gives **5.447**.
- Both alternative modes retain a smaller FID increase from 50 to 200 steps
  in every seed. Clipping treatment does not explain the entire non-monotonic curve.

These controlled comparisons support sensitivity to clipping treatment for
this checkpoint. They do not establish a universally optimal step count,
a high-frequency error mechanism, or robustness across independently trained models.

## Results: clipping ablation

Mean ± **sample standard deviation (ddof=1)** over three sampling seeds.
Every cell has n=3; each run generates 10,000 images. Lower FID is better.

| Steps | No intermediate clipping | Original clipping | Clip + recompute noise |
|---|---|---|---|
| 10 | 10.486 ± 0.080 | 11.039 ± 0.070 | 10.832 ± 0.063 |
| 20 | 5.405 ± 0.058 | 6.369 ± 0.047 | 5.639 ± 0.066 |
| 50 | 4.878 ± 0.061 | 7.707 ± 0.073 | 5.121 ± 0.052 |
| 200 | 5.447 ± 0.082 | 17.471 ± 0.099 | 5.680 ± 0.069 |


The compared update rules are:

| Mode | Intermediate clean-image estimate | Direction noise |
|---|---|---|
| no_clip | Unclipped | Original predicted noise |
| clip_original | Clipped to [-1, 1] | Original predicted noise |
| clip_recompute | Clipped to [-1, 1] | Recomputed to agree with clipped estimate |

**All modes clip the final image to [-1, 1] before conversion to PNG and FID.**
“No clipping” therefore refers only to intermediate sampling updates.
The no_clip out-of-range fraction is recorded before final clipping; it does
not measure clipping magnitude or independently establish image quality.

### Protocol and provenance

- One 30,000-step EMA checkpoint, as reported by the Colab checkpoint inspection.
- SHA-256: `c4145894a7dfd44a695396070fccd142c53fff3c06c46e4d04a66a4177b3adc6`.
- MNIST, padded to 32×32; T=1000; linear beta from 0.0001 to 0.02; eta=0.
- Batch size 100, NVIDIA A100-SXM4-40GB, recorded PyTorch 2.11.0+cu128.
- Matched initial-noise sequences across modes and step counts within each seed.
- Colab used clean-fid in clean mode against 10,000 padded MNIST test images.
  The exact clean-fid version was not recorded.
- These are user-executed Colab measurements archived without changing scores;
  this update recomputes statistics and plots, not generation or FID.
- Raw files: [seed 0](results/clipping_ablation/seed_0.json),
  [seed 42](results/clipping_ablation/seed_42.json),
  [seed 123](results/clipping_ablation/seed_123.json).
- [Summary CSV](results/clipping_ablation/summary.csv) includes mean sampling times.

Mean sampling time for 10,000 images is approximately 10–11 seconds at 20 steps,
25–26 seconds at 50 steps, and 101–103 seconds at 200 steps.
Timing covers synchronized sampler calls after warm-up; it excludes initial-noise
creation, PNG saving, preview generation, and FID evaluation. These are observed
times from the runs, not a separate randomized latency benchmark.

The weights are not included in this repository. The supplied preview PNGs are archived in [assets/clipping_previews](assets/clipping_previews).
The ablation was run using the Colab cell supplied for this study; the existing
`ddim.py` still implements the original clipped mode. The legacy FID runner
does **not** reproduce all three modes. The script below reproduces the
summary and figure from the archived measurements only.

## Rebuild the published statistics and figure

From the repository root:

```bash
pip install matplotlib
python scripts/summarize_ablation.py
```

No GPU, model weights, or MNIST download is needed. The script validates seed
identities, matching recorded configurations, and all 36 result entries before
writing the summary CSV, Markdown table, and SVG/PNG figures. It uses Python's
sample standard deviation and plots means with corresponding error bars.

## Fixed-noise preview comparison

![Matched preview comparison](assets/clipping_preview_comparison.png)

Rows: no intermediate clipping, original clipping, and clipping with recomputed
noise. Columns: 20, 50, and 200 steps. Each panel shows the same 16 initial
latents (preview seed 2026 in the experiment cell). All modes receive the same
final display clipping. The supplied archive's results.json exactly matches
the archived sampling-seed-0 experiment; preview seed and evaluation seed are
different.

### Visual observations and limits

- Corresponding positions largely preserve recognizable digit structures across
  modes and step counts in these previews.
- Original clipping at 200 steps shows visible isolated background marks,
  including near the top-right sample and beside the second-row left sample.
  Corresponding regions look cleaner in the two alternative 200-step panels.
- This is consistent with an observable effect of clipping treatment, but does
  not establish that these marks explain the full FID gap or prove a
  high-frequency accumulation mechanism.
- The alternative 50- and 200-step panels are visually similar at this scale.
  These 16 examples do not reliably resolve their smaller FID differences.

The previews are illustrative, not the 10,000-image FID sets. Their fixed
seed is shared across runs, so repeated copies from different evaluation-seed
folders are not independent visual evidence. No class labels or image-wide
quality ratings are inferred from this small grid.

Rebuild the comparison from the nine original PNGs with
`python scripts/build_preview_figure.py` (requires matplotlib). The script only
arranges existing images using nearest-neighbor display; it does not regenerate
or retouch digits.

## Implementation and derivations

The existing training configuration uses 30,000 steps, batch size 128, AdamW
at 2e-4, EMA decay 0.999, and CUDA mixed precision. Checkpoints restore model,
EMA, optimizer, and step, but not all RNG or GradScaler state for exact continuation.

[Derivation notes](notes/derivation.md) cover forward marginals, ELBO telescoping,
Gaussian KL, simplified noise MSE, DDIM's non-Markovian construction, and code
mappings. In particular, eta=1 is not identical to this repository's DDPM
sampler, which uses beta rather than posterior beta-tilde variance.

## Historical exploration

The earlier original-clipping sweep reported:

| Steps | 2 | 3 | 5 | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|---|
| Single-seed FID | 320.9 | 129.5 | 41.7 | 10.6 | 6.2 | 7.6 | 9.9 | 17.4 |

These historical values are **not pooled with the new ablation**. The old runner
uses batch size 500 and a different RNG setup; an identical numeric seed need
not give identical initial samples across the two implementations.
The previous “20 steps best” claim is now scoped to original clipping and tested
settings. Low-step degradation alone does not establish a phase transition.
The old plot and plotting utility remain historical artifacts; use
`scripts/summarize_ablation.py` for the current figure.

## Limitations and next steps

**Completed:** three clipping modes × four step counts × three sampling seeds,
with raw scores, sampling-time measurements, and consistent mean/sample-SD plots.

**Completed visual follow-up:** nine preview grids have been archived and combined
above. The next research gate is independent training checkpoints, rather than
additional copies of the same fixed-noise previews.

**Further experiments:**

1. Train independently seeded checkpoints in separate directories and repeat
   the key comparisons to assess training variability.
2. Add domain-relevant quality/diversity metrics and inspect final clipping
   magnitudes. Inception features and finite-sample FID have limitations.
3. Densify step counts around the observed minimum using a validation split
   for selection, then evaluate on held-out data. The current minima were
   selected using test-set FID and are exploratory.
4. Extend datasets, noise schedules, and eta; add mechanistic diagnostics before
   attributing the remaining increase to accumulated high-frequency errors.

The three sampling seeds do not quantify training variability, metric bias,
or uncertainty from resampling the real-image reference set.

## References

- [Ho et al., DDPM](https://arxiv.org/abs/2006.11239)
- [Song et al., DDIM](https://arxiv.org/abs/2010.02502)
- [Chong & Forsyth, finite-sample FID bias](https://arxiv.org/abs/1911.07023)
