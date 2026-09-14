"""Summarize each checkpoint separately from archived measurements; no inference."""
from pathlib import Path
import csv,json,math,statistics as st
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype']='none'
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/clipping_ablation'
MODES=['no_clip','clip_original','clip_recompute']
STEPS=[20,50,200]

def main():
    groups=[('Original checkpoint',DATA),('Training seed 2026',DATA/'train_seed_2026')]
    rows=[];configs=[]
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),sharey=True)
    tables=[]
    for ax,(label,folder) in zip(axes,groups):
        runs=[json.loads((folder/f'seed_{s}.json').read_text()) for s in [0,42,123]]
        cfg={k:v for k,v in runs[0]['config'].items() if k!='seed'}
        configs.append(cfg)
        for seed,r in zip([0,42,123],runs):
            assert r['config']['seed']==seed
            assert {k:v for k,v in r['config'].items() if k!='seed'}==cfg
            assert set(r['results'])=={f'{m}_{s}' for m in cfg['modes'] for s in cfg['steps']}
            for key,v in r['results'].items():
                assert key==f"{v['mode']}_{v['steps']}"
                assert math.isfinite(v['fid']) and v['fid']>=0
        for mode,color in zip(MODES,['#237a57','#bd443e','#366fc0']):
            selected=[]
            for step in STEPS:
                vals=[r['results'][f'{mode}_{step}'] for r in runs]
                row=dict(checkpoint=label,checkpoint_sha256=cfg['checkpoint_sha256'],mode=mode,steps=step,n_sampling_seeds=3,fid_mean=st.mean(v['fid'] for v in vals),fid_sample_sd=st.stdev(v['fid'] for v in vals),sampling_seconds_mean=st.mean(v['generation_seconds'] for v in vals))
                rows.append(row);selected.append(row)
            ax.errorbar(STEPS,[v['fid_mean'] for v in selected],yerr=[v['fid_sample_sd'] for v in selected],marker='o',capsize=4,label=mode,color=color)
        ax.set(xscale='log',xlabel='DDIM steps',title=label,ylim=(0,23))
        ax.set_xticks(STEPS,[str(s) for s in STEPS]);ax.grid(alpha=.2)
        table=['| Steps | No intermediate clipping | Original clipping | Clip + recompute |','|---|---|---|---|']
        for s in STEPS:
            vals=[next(r for r in rows if r['checkpoint']==label and r['steps']==s and r['mode']==m) for m in MODES]
            table.append('| '+str(s)+' | '+' | '.join(f"{v['fid_mean']:.3f} ± {v['fid_sample_sd']:.3f}" for v in vals)+' |')
        tables.append('### '+label+'\n\n'+'\n'.join(table))
    assert configs[0]['checkpoint_sha256']!=configs[1]['checkpoint_sha256']
    assert {k:v for k,v in configs[0].items() if k not in ['checkpoint_sha256','steps']}=={k:v for k,v in configs[1].items() if k not in ['checkpoint_sha256','steps']}
    training=json.loads((DATA/'train_seed_2026/training_config.json').read_text())
    assert training['training_seed']==2026 and training['total_steps']==30000
    for key in ['T','beta_start','beta_end','image_size','channels']: assert training[key]==configs[1][key]
    with (DATA/'checkpoint_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (DATA/'checkpoint_tables.md').write_text('\n\n'.join(tables)+'\n')
    axes[0].set_ylabel('FID (lower is better)');axes[0].legend(fontsize=8)
    fig.text(.5,.02,'Within-checkpoint mean ± sample SD (ddof=1), 3 sampling seeds each; no pooling.\n10,000 images/run, eta=0, batch 100. Original 10-step runs excluded from this matched comparison.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.10,1,1))
    fig.savefig(ROOT/'assets/checkpoint_comparison.svg');fig.savefig(ROOT/'assets/checkpoint_comparison.png',dpi=150)
    plt.close(fig)
    print('\n\n'.join(tables))
if __name__=='__main__':main()
