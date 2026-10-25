"""Original 1B score-response dynamics and final-checkpoint controls."""

import json
from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np

from src.artifacts import DynamicsResult
from src.constants import FIGURE2_STEPS


def read_result(path: Path) -> tuple[DynamicsResult, list[float]]:
    result = cast(DynamicsResult, json.loads(path.read_text()))
    if result.get('metric') != 'w1_top10_l2':
        raise ValueError('expected signed top-10 gate-preactivation L2, not full L2 or SiLU')
    rows = result['dynamics']
    if [row['step'] for row in rows] != list(FIGURE2_STEPS):
        raise ValueError('expected the six original checkpoints in order')
    values = []
    for row in rows:
        for group in ('routed', 'nonrouted'):
            interval = row['routed_ci'] if group == 'routed' else row['nonrouted_ci']
            if len(interval) != 2 or interval[0] > interval[1]:
                raise ValueError('invalid bootstrap interval')
            values.extend([row[group], *interval])
        values.append(row['identity_permuted_routed'])
    final = result['final']
    values.extend(final[key] for key in ('routed', 'nonrouted', 'identity_permuted_routed', 'token_shuffled_routed'))
    if not np.isfinite(values).all() or (np.abs(values) > 1).any():
        raise ValueError('correlations must be finite and lie in [-1, 1]')
    if any(final[group] != rows[-1][group] for group in ('routed', 'nonrouted')):
        raise ValueError('final checkpoint disagrees with the dynamics')
    return result, values


def plot(result: DynamicsResult, values: list[float], out: Path) -> None:
    rows, final = result['dynamics'], result['final']
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), sharey=True)
    colors = ['#0072B2', '#D55E00', '#777777', '#CC79A7']
    positions = np.arange(6)
    for group, label, color in zip(cast(tuple[Literal['routed', 'nonrouted'], ...], ('routed', 'nonrouted')), ('Routed experts', 'Non-routed experts'), colors[:2], strict=True):
        axes[0].plot(positions, [r[group] for r in rows], 'o-', color=color, markersize=4, label=label)
        intervals = np.asarray([r['routed_ci'] if group == 'routed' else r['nonrouted_ci'] for r in rows])
        axes[0].fill_between(positions, intervals[:, 0], intervals[:, 1], color=color, alpha=.14)
    axes[0].plot(positions, [r['identity_permuted_routed'] for r in rows], '--',
                 color=colors[2], label='Permuted expert identity')
    axes[0].set(xticks=positions, xticklabels=['0', '1k', '5k', '10k', '20k', '39.75k'],
                xlabel='Training step (checkpoints equally spaced)', title='Training dynamics')
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    keys: tuple[Literal['routed', 'nonrouted', 'identity_permuted_routed', 'token_shuffled_routed'], ...] = ('routed', 'nonrouted', 'identity_permuted_routed', 'token_shuffled_routed')
    bars = [final[key] for key in keys]
    axes[1].bar(np.arange(4), bars, color=colors, width=.7)
    axes[1].set(xticks=np.arange(4), xticklabels=['Routed', 'Non-routed', 'Expert-\npermuted', 'Token-\nshuffled'],
                title='Controls at step 39,750')
    for i, value in enumerate(bars):
        label = f'{value:.2g}' if 0 < abs(value) < .001 else f'{value:.3f}'
        axes[1].annotate(label, (i, value), xytext=(0, 5 if value >= 0 else -12),
                         textcoords='offset points', ha='center', fontsize=9)
    for axis in axes:
        axis.axhline(0, color='.65', linewidth=.7)
        axis.spines[['top', 'right']].set_visible(False)
        axis.set_ylim(min(-.04, min(values) - .04), max(.60, max(values) + .12))
        axis.set_axisbelow(True)
        axis.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Score–response Spearman ρ\n(within-expert standardized, pooled)')
    fig.text(.5, .01, 'Response: L2 of 10 largest signed gate preactivations · Shading: 95% title-bootstrap intervals',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(out, dpi=180)
    plt.close(fig)
