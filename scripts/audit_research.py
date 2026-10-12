"""CPU-only, standard-library audit of archived evidence; never runs a model."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/clipping_ablation'
SEEDS = (0, 42, 123)
MODES = ('no_clip', 'clip_original', 'clip_recompute')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_runs(folder, steps):
    runs, sources = [], []
    for seed in SEEDS:
        path = folder / f'seed_{seed}.json'
        raw = path.read_bytes()
        run = json.loads(raw)
        config = run['config']
        require(config['seed'] == seed, f'{path}: wrong seed')
        require(config['steps'] == steps and config['modes'] == list(MODES), f'{path}: grid mismatch')
        expected = {f'{mode}_{step}' for mode in MODES for step in steps}
        require(set(run['results']) == expected, f'{path}: incomplete grid')
        for key, value in run['results'].items():
            require(key == f"{value['mode']}_{value['steps']}", f'{path}: identity mismatch')
            require(math.isfinite(value['fid']) and value['fid'] >= 0, f'{path}: invalid FID')
        cfg = {k: v for k, v in config.items() if k != 'seed'}
        if runs:
            require(cfg == {k: v for k, v in runs[0]['config'].items() if k != 'seed'}, f'{path}: config mismatch')
        runs.append(run)
        sources.append((str(path.relative_to(ROOT)), hashlib.sha256(raw).hexdigest()))
    return runs, sources


def algebra_check():
    # Synthetic scalar identities, including active/inactive clipping and endpoint.
    # This checks the derivation, not PyTorch, checkpoint quality, or FID.
    count = 0
    for a in (0.00004, 0.1, 0.5, 0.9999):
        for b in (a, (a + 1) / 2, 1.0):
            for x0 in (-2., -1., 0.25, 1., 2.):
                eps = 0.37
                x = math.sqrt(a) * x0 + math.sqrt(1-a) * eps
                clipped = min(1., max(-1., x0))
                delta = clipped - x0
                corrected = (x - math.sqrt(a)*clipped) / math.sqrt(1-a)
                residual = x - (math.sqrt(a)*clipped + math.sqrt(1-a)*eps)
                require(math.isclose(residual, -math.sqrt(a)*delta, abs_tol=1e-12), 'residual identity')
                require(math.isclose(x, math.sqrt(a)*clipped + math.sqrt(1-a)*corrected, abs_tol=1e-12), 'reconstruction identity')
                difference = math.sqrt(1-b)*(corrected-eps)
                require(math.isclose(difference, -math.sqrt((1-b)*a/(1-a))*delta, abs_tol=1e-12), 'update identity')
                no_clip = math.sqrt(b)*x0 + math.sqrt(1-b)*eps
                original = math.sqrt(b)*clipped + math.sqrt(1-b)*eps
                recompute = math.sqrt(b)*clipped + math.sqrt(1-b)*corrected
                require(math.isclose(original-no_clip, math.sqrt(b)*delta, abs_tol=1e-12), 'original update delta')
                expected = (math.sqrt(b)-math.sqrt((1-b)*a/(1-a)))*delta
                require(math.isclose(recompute-no_clip, expected, abs_tol=1e-12), 'recompute update delta')
                if delta == 0:
                    require(math.isclose(original, recompute, abs_tol=1e-12) and original == no_clip, 'inactive clipping')
                if b == a:
                    require(math.isclose(no_clip, x, abs_tol=1e-12) and math.isclose(recompute, x, abs_tol=1e-12), 'zero-step identity')
                if b == 1:
                    require(original == recompute == clipped, 'clean endpoint')
                count += 1
    return count


def main():
    groups = [('original', DATA, [10, 20, 50, 200]),
              ('training_seed_2026', DATA/'train_seed_2026', [20, 50, 200])]
    paired, summaries, configs, sources = [], [], [], []
    total = 0
    for label, folder, steps in groups:
        runs, provenance = load_runs(folder, steps)
        sources.extend(provenance)
        configs.append(runs[0]['config'])
        total += sum(len(r['results']) for r in runs)
        contrasts = [(f'{mode}:200-50', f'{mode}_200', f'{mode}_50') for mode in MODES]
        contrasts += [(f'clip_recompute-no_clip:{step}', f'clip_recompute_{step}', f'no_clip_{step}') for step in (20, 50, 200)]
        contrasts += [('clip_original-clip_recompute:200', 'clip_original_200', 'clip_recompute_200')]
        for name, left, right in contrasts:
            differences = []
            for seed, run in zip(SEEDS, runs):
                difference = run['results'][left]['fid'] - run['results'][right]['fid']
                differences.append(difference)
                paired.append(dict(checkpoint=label, checkpoint_sha256=run['config']['checkpoint_sha256'], sampling_seed=seed, contrast=name, left_fid=run['results'][left]['fid'], right_fid=run['results'][right]['fid'], fid_difference=difference))
            summaries.append(f'| {label} | {name} | {st.mean(differences):.6f} | {st.stdev(differences):.6f} | {sum(x > 0 for x in differences)}/3 |')
    exclude = {'seed', 'steps', 'checkpoint_sha256'}
    require({k:v for k,v in configs[0].items() if k not in exclude} == {k:v for k,v in configs[1].items() if k not in exclude}, 'cross-checkpoint protocol mismatch')
    require(configs[0]['checkpoint_sha256'] != configs[1]['checkpoint_sha256'], 'checkpoint hashes must differ')
    metadata = json.loads((DATA/'train_seed_2026/checkpoint_metadata.json').read_text())
    require(metadata['checkpoint_sha256'] == configs[1]['checkpoint_sha256'] and metadata['completed_steps'] == 30000 and metadata['has_ema'] is True, 'metadata mismatch')
    require(total == 63, 'unexpected archive count')
    count = algebra_check()
    with (DATA/'paired_differences.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    text = ['# Archived evidence audit', '', 'Generated by `python3 scripts/audit_research.py`; no model execution or FID recomputation.', '', f'Validated {total} measurements; 54 on the shared 20/50/200 grid. Synthetic scalar algebra checks: {count}.', '', 'Post-hoc exploratory contrasts, paired within checkpoint and sampling seed. SD is the sample SD of three paired differences (ddof=1), not a confidence interval or training uncertainty. Positive means left-hand FID is worse. The repeated contrasts share runs and are not independent replications.', '', '| Checkpoint | Contrast (left minus right) | Mean ΔFID | Sample SD of Δ | Positive |', '|---|---|---:|---:|---:|', *summaries, '', '## Exact raw sources', '', 'SHA-256 below identifies the JSON bytes read, not independently inspected checkpoint weights.', '']
    text += [f'- `{path}`: `{digest}`' for path, digest in sources]
    (DATA/'evidence_audit.md').write_text('\n'.join(text)+'\n')
    print('\n'.join(text[:23]))


if __name__ == '__main__':
    main()
