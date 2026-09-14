"""Rebuild tables and figure from archived Colab results; no GPU required."""
from pathlib import Path
import csv
import json
import math
import statistics as st
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/clipping_ablation'

def main():
    runs = [json.loads((DATA / f'seed_{s}.json').read_text()) for s in (0,42,123)]
    config = {k:v for k,v in runs[0]['config'].items() if k != 'seed'}
    for seed, run in zip((0,42,123), runs):
        assert run['config']['seed'] == seed
        assert {k:v for k,v in run['config'].items() if k != 'seed'} == config
        expected = {f'{m}_{s}' for m in config['modes'] for s in config['steps']}
        assert set(run['results']) == expected
        for key, row in run['results'].items():
            assert key == f"{row['mode']}_{row['steps']}"
            assert math.isfinite(row['fid']) and row['fid'] >= 0
            assert row['generation_seconds'] > 0
    rows = []
    for mode in config['modes']:
        for steps in config['steps']:
            values = [r['results'][f'{mode}_{steps}'] for r in runs]
            rows.append(dict(mode=mode, steps=steps, n=3,
                fid_mean=st.mean(v['fid'] for v in values),
                fid_sample_sd=st.stdev(v['fid'] for v in values),
                generation_seconds_mean=st.mean(v['generation_seconds'] for v in values)))
    with (DATA / 'summary.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    table=['| Steps | No intermediate clipping | Original clipping | Clip + recompute noise |',
           '|---|---|---|---|']
    for steps in config['steps']:
        vals=[next(r for r in rows if r['mode']==m and r['steps']==steps) for m in config['modes']]
        table.append('| '+str(steps)+' | '+' | '.join(f"{v['fid_mean']:.3f} ± {v['fid_sample_sd']:.3f}" for v in vals)+' |')
    (DATA/'table.md').write_text('\n'.join(table)+'\n')
    fig, axes=plt.subplots(1,2,figsize=(11,4.8))
    labels=['No intermediate clipping','Original clipping','Clip + recompute noise']
    for mode,label,color in zip(config['modes'],labels,['#237a57','#bd443e','#366fc0']):
        group=[r for r in rows if r['mode']==mode]
        y=[r['fid_mean'] for r in group];err=[r['fid_sample_sd'] for r in group]
        axes[0].errorbar([r['steps'] for r in group],y,yerr=err,marker='o',capsize=4,label=label,color=color)
        axes[1].errorbar([r['generation_seconds_mean'] for r in group],y,yerr=err,marker='o',capsize=4,color=color)
    axes[0].set(xscale='log',xlabel='DDIM sampling steps',ylabel='FID (lower is better)',title='Clipping changes the step-quality relationship')
    axes[0].set_xticks(config['steps'],[str(s) for s in config['steps']]);axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Mean sampling seconds / 10,000 images',ylabel='FID (lower is better)',title='Observed quality and sampling time')
    for ax in axes: ax.grid(alpha=.2);ax.set_ylim(bottom=0)
    fig.text(.5,.015,'Mean ± sample SD; 3 sampling seeds; one EMA checkpoint; MNIST; eta=0.\nA100, batch 100. Timing excludes image saving and FID. All outputs clipped before FID.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.09,1,1))
    (ROOT/'assets').mkdir(exist_ok=True)
    fig.savefig(ROOT/'assets/clipping_ablation.svg')
    fig.savefig(ROOT/'assets/clipping_ablation.png',dpi=180)
    plt.close(fig)
    print('\n'.join(table))

if __name__ == '__main__':
    main()
