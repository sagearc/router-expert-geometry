"""Qwen and DeepSeek-V4 expert-rank response curves."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.artifacts import RankCurve


def plot(result: RankCurve, out: Path) -> None:
    rank, top_k = np.asarray(result['expert_rank']), result['top_k']
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharex=True)
    for ax, mean, name in zip(axes, result['response_mean'], ['gate', 'up'], strict=True):
        ax.axvspan(.5, top_k + .5, color='#dceafa', label=f'Native top {top_k}')
        ax.axvline(top_k + .5, color='#7c9aba', linewidth=.8, linestyle='--')
        ax.plot(rank, mean, color='#2577b6', linewidth=1.6)
        ax.set(xlabel='Expert router-score rank (1 = highest)',
               ylabel=rf'Mean $\|W_{{\mathrm{{{name}}}}}h\|_2$',
               xlim=(.5, len(rank) + .5), ylim=(0, None))
        ax.set_xticks([1, len(rank) // 2, len(rank)])
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(result['model'].split('/')[-1])
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
