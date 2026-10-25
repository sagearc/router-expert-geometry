"""Reduce an existing 1B capture with the exact pooled Figure 2 statistic.

uv run figure2 capture.npz --response w1_top10_l2
"""

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np

from src.analysis.measurements import (
    pooled_correlation,
    selected_mask,
)


class Args:
    capture: Path
    response: Literal['w1_top10_l2', 'w1_silu_mean']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--response', required=True, choices=['w1_top10_l2', 'w1_silu_mean'])
    args = parser.parse_args(namespace=Args())
    with np.load(args.capture, allow_pickle=False) as data:
        scores = data['router_scores'].astype(np.float32)
        response = data[args.response].astype(np.float32)
        selected = data['selected_experts']
    mask = selected_mask(selected, scores.shape[-1])
    print(json.dumps({
        'response': args.response,
        'reduction': 'within_layer_expert_standardized_pooled_spearman',
        'shape': list(scores.shape),
        'routed': pooled_correlation(scores, response, mask),
        'nonrouted': pooled_correlation(scores, response, ~mask),
        'routed_by_layer': [pooled_correlation(
            scores[:, i:i+1], response[:, i:i+1], mask[:, i:i+1]
        ) for i in range(scores.shape[1])],
    }, indent=2, allow_nan=False))
