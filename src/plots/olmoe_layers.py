"""OLMoE score-gate-response Spearman correlation against layer depth."""

import json
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from src.artifacts import CorrelationGroup, LayerCorrelations
from src.constants import OLMOE_LAYERS, OLMOE_TOP_K


def read_result(path: Path) -> LayerCorrelations:
    result = cast(LayerCorrelations, json.loads(path.read_text()))
    if (result.get('response') != 'l2_norm_w_gate_h'
            or result.get('aggregation') != 'mean_of_per_title_spearman'
            or result.get('layers') != list(range(1, OLMOE_LAYERS + 1))):
        raise ValueError('expected the public OLMoE layer-correlation reduction')
    for name in ('routed_top8', 'bottom8', 'bottom_half'):
        group = result['groups'][name]
        mean, std = np.asarray(group['mean']), np.asarray(group['title_std'])
        valid, undefined = np.asarray(group['valid_titles']), np.asarray(group['undefined_titles'])
        if any(value.shape != (OLMOE_LAYERS,) for value in (mean, std, valid, undefined)):
            raise ValueError('expected one value per layer')
        if (not np.isfinite(mean).all() or not np.isfinite(std).all()
                or (abs(mean) > 1).any() or (std < 0).any()
                or (valid < 1).any() or (undefined < 0).any()
                or not np.all(valid + undefined == result['n_titles'])):
            raise ValueError('invalid correlations, spread or title counts')
    return result


def plot(results: list[tuple[str, LayerCorrelations]], out: Path, show_spread: bool) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4), sharex=True, sharey=True)
    groups: list[tuple[CorrelationGroup, str]] = [('routed_top8', 'Routed top 8'), ('bottom8', 'Bottom 8'), ('bottom_half', 'Bottom half')]
    for ax, (group, title) in zip(axes, groups, strict=True):
        for label, result in results:
            mean = np.asarray(result['groups'][group]['mean'])
            std = np.asarray(result['groups'][group]['title_std'])
            line, = ax.plot(result['layers'], mean, marker='o', markersize=3, linewidth=1.5, label=label)
            if show_spread:
                ax.fill_between(result['layers'], mean - std, mean + std, color=line.get_color(), alpha=.12)
        ax.axhline(0, color='.65', linewidth=.7, linestyle='--')
        ax.set(title=title, xlabel='MoE layer (1-indexed)', xlim=(1, OLMOE_LAYERS), ylim=(-1, 1),
               xticks=[1, 4, OLMOE_TOP_K, 12, OLMOE_LAYERS], yticks=[-1, 0, 1])
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Score–gate-response Spearman ρ')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
               ncol=min(5, len(results)), frameon=False)
    if show_spread:
        fig.text(.5, .01, 'Shading: ±1 standard deviation across titles (not confidence intervals)',
                 ha='center', fontsize=OLMOE_TOP_K)
    fig.tight_layout(rect=(0, .05, 1, .88))
    fig.savefig(out, dpi=180)
    plt.close(fig)
