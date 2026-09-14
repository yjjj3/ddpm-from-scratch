"""Compose existing PNGs without altering sample pixels; no model inference."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def build(source, output):
    fig,axes=plt.subplots(3,3,figsize=(9,9.6))
    modes=['no_clip','clip_original','clip_recompute']
    labels=['No intermediate clipping','Original clipping','Clip + recompute noise']
    for row,(mode,label) in enumerate(zip(modes,labels)):
        for col,steps in enumerate([20,50,200]):
            ax=axes[row,col]
            ax.imshow(mpimg.imread(source/f'{mode}_{steps}.png'),interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            if row==0: ax.set_title(f'{steps} steps',fontsize=13)
            if col==0: ax.set_ylabel(label,fontsize=10)
    fig.suptitle('DDIM clipping ablation: matched preview noise',fontsize=16)
    fig.text(.5,.015,'Same 16 initial latents (preview seed 2026), one EMA checkpoint.\nFinal display clipping is applied to every mode. Illustrative previews, not the FID sample set.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.95))
    fig.savefig(output,dpi=160)
    plt.close(fig)

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    build(root/'assets/clipping_previews',root/'assets/clipping_preview_comparison.png')
