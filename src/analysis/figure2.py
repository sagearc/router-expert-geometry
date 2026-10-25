"""Original 1B Figure 2: the pooled statistic, its dynamics and its controls."""

import hashlib
from pathlib import Path

import numpy as np

from src.analysis.measurements import pooled_correlation, selected_mask
from src.artifacts import DynamicsFinal, DynamicsPoint, DynamicsResult
from src.constants import (
    CI95_QUANTILES,
    FIGURE2_BOOTSTRAP_DRAWS,
    FIGURE2_BOOTSTRAP_SEED,
    FIGURE2_FINAL_IDENTITY_DRAWS,
    FIGURE2_IDENTITY_DRAWS,
    FIGURE2_SEED,
    FIGURE2_SHUFFLE_DRAWS,
    FIGURE2_STEPS,
    FIGURE2_TITLE_COUNT,
    PAPER_EXPERTS,
    PAPER_LAYERS,
    PAPER_TOP_K,
)
from src.types import BoolArray, FloatArray


def load_capture(path: Path) -> tuple[FloatArray, FloatArray, BoolArray]:
    with np.load(path, allow_pickle=False) as data:
        scores = data['router_scores'].astype(np.float32)
        response = data['w1_top10_l2'].astype(np.float32)
        selected = data['selected_experts']
    if scores.shape != (FIGURE2_TITLE_COUNT, PAPER_LAYERS, PAPER_EXPERTS) or response.shape != scores.shape or selected.shape != (FIGURE2_TITLE_COUNT, PAPER_LAYERS, PAPER_TOP_K):
        raise ValueError('expected 1,024 titles, nine layers, 64 experts and top-6 routing')
    return scores, response, selected_mask(selected, PAPER_EXPERTS)


def identity_controls(
    scores: FloatArray,
    response: FloatArray,
    mask: BoolArray,
    rng: np.random.Generator,
    repeats: int,
) -> list[float]:
    return [pooled_correlation(scores, response, mask,
                              np.stack([rng.permutation(PAPER_EXPERTS) for _ in range(PAPER_LAYERS)]))
            for _ in range(repeats)]


def bootstrap_interval(
    scores: FloatArray, response: FloatArray, mask: BoolArray, rng: np.random.Generator,
) -> list[float]:
    values = []
    for _ in range(FIGURE2_BOOTSTRAP_DRAWS):
        indices = rng.integers(0, len(scores), size=len(scores))
        values.append(pooled_correlation(scores[indices], response[indices], mask[indices]))
    return [float(value) for value in np.quantile(values, CI95_QUANTILES)]


def by_layer(scores: FloatArray, response: FloatArray, mask: BoolArray) -> list[float]:
    return [pooled_correlation(scores[:, i:i + 1], response[:, i:i + 1], mask[:, i:i + 1])
            for i in range(scores.shape[1])]


def capture_paths(directory: Path) -> dict[int, Path]:
    """The six retained ``step{step}-top10-random1024.npz`` captures, in step order."""
    paths = {step: directory / f'step{step}-top10-random1024.npz' for step in FIGURE2_STEPS}
    if any(not path.is_file() for path in paths.values()):
        raise ValueError('requires all six step{step}-top10-random1024.npz captures')
    return paths


def dynamics(paths: dict[int, Path]) -> DynamicsResult:
    """Dynamics and controls, consuming the RNGs in the retained script's order."""
    rng, bootstrap_rng = np.random.default_rng(FIGURE2_SEED), np.random.default_rng(FIGURE2_BOOTSTRAP_SEED)
    points: list[DynamicsPoint] = []
    for step, path in paths.items():
        scores, response, routed = load_capture(path)
        controls = identity_controls(scores, response, routed, rng, FIGURE2_IDENTITY_DRAWS)
        points.append({
            'step': step,
            'routed': pooled_correlation(scores, response, routed),
            'routed_ci': bootstrap_interval(scores, response, routed, bootstrap_rng),
            'nonrouted': pooled_correlation(scores, response, ~routed),
            'nonrouted_ci': bootstrap_interval(scores, response, ~routed, bootstrap_rng),
            'identity_permuted_routed': float(np.mean(controls)),
        })
        print(f'reduced step {step}', flush=True)
    # The final checkpoint remains in memory after the ordered sweep.
    final_identity = identity_controls(scores, response, routed, rng, FIGURE2_FINAL_IDENTITY_DRAWS)
    shuffled_controls = []
    for _ in range(FIGURE2_SHUFFLE_DRAWS):
        shuffled = response.copy()
        for layer in range(PAPER_LAYERS):
            for expert in range(PAPER_EXPERTS):
                tokens = np.flatnonzero(routed[:, layer, expert])
                shuffled[tokens, layer, expert] = response[rng.permutation(tokens), layer, expert]
        shuffled_controls.append(pooled_correlation(scores, shuffled, routed))
    final: DynamicsFinal = {
        'routed': points[-1]['routed'], 'nonrouted': points[-1]['nonrouted'],
        'identity_permuted_routed': float(np.mean(final_identity)),
        'identity_permuted_routed_range': [min(final_identity), max(final_identity)],
        'token_shuffled_routed': float(np.mean(shuffled_controls)),
        'routed_by_layer': by_layer(scores, response, routed),
        'nonrouted_by_layer': by_layer(scores, response, ~routed),
    }
    return {
        'metric': 'w1_top10_l2', 'dynamics': points, 'final': final,
        'protocol': {'seed': FIGURE2_SEED, 'bootstrap_seed': FIGURE2_BOOTSTRAP_SEED, 'title_bootstrap_repeats': FIGURE2_BOOTSTRAP_DRAWS,
                     'identity_repeats_per_checkpoint': FIGURE2_IDENTITY_DRAWS, 'final_identity_repeats': FIGURE2_FINAL_IDENTITY_DRAWS,
                     'final_token_shuffle_repeats': FIGURE2_SHUFFLE_DRAWS},
        'capture_sha256': {str(step): hashlib.sha256(path.read_bytes()).hexdigest()
                           for step, path in paths.items()},
    }
