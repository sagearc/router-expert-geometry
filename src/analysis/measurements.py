"""The paper's score/response and router-geometry reductions.

Capture axes are (tokens, layers, experts). Response definitions are selected
by the caller, never inferred from a filename or substituted between models.
"""

from collections.abc import Sequence

import numpy as np
from scipy.stats import spearmanr

from src.types import BoolArray, Float64Array, FloatArray, IntArray


def selected_mask(selected: IntArray, experts: int) -> BoolArray:
    if selected.ndim != 3 or not np.issubdtype(selected.dtype, np.integer):
        raise ValueError('selected must be integer (tokens, layers, top_k) IDs')
    if experts < 1 or selected.shape[-1] < 1 or selected.shape[-1] > experts:
        raise ValueError('invalid expert count or top_k')
    if np.any(selected < 0) or np.any(selected >= experts):
        raise ValueError('expert ID outside capture range')
    if np.any(np.diff(np.sort(selected, axis=-1), axis=-1) == 0):
        raise ValueError('duplicate selected expert')
    mask = np.zeros((*selected.shape[:2], experts), dtype=bool)
    np.put_along_axis(mask, selected, True, axis=-1)
    return mask


def _validate(scores: FloatArray, responses: FloatArray, mask: BoolArray) -> None:
    if scores.ndim != 3 or scores.shape != responses.shape or scores.shape != mask.shape:
        raise ValueError('scores, responses and mask must share (tokens, layers, experts) shape')
    if mask.dtype != np.bool_:
        raise ValueError('mask must be boolean')
    if not np.isfinite(scores).all() or not np.isfinite(responses).all():
        raise ValueError('capture contains non-finite scores or responses')


def pooled_correlation(
    scores: FloatArray, responses: FloatArray, mask: BoolArray,
    expert_permutations: IntArray | Sequence[Sequence[int]] | None = None,
) -> float:
    """Original 1B reduction: within-layer/expert z-scores, then pooled Spearman.

    Groups with fewer than ten tokens are omitted, as in the retained analysis.
    For the identity control, masks and scores stay with the source router;
    only the response expert changes. This is NOT averaged per-token Spearman.
    """
    _validate(scores, responses, mask)
    layers, experts = scores.shape[1:]
    permutations = np.tile(np.arange(experts), (layers, 1))
    if expert_permutations is not None:
        permutations = np.asarray(expert_permutations)
        if not np.issubdtype(permutations.dtype, np.integer) or permutations.shape != (layers, experts) or not np.all(
            np.sort(permutations, axis=1) == np.arange(experts)
        ):
            raise ValueError('each layer needs a permutation of expert IDs')
    xs, ys = [], []
    for layer in range(layers):
        for expert in range(experts):
            keep = mask[:, layer, expert]
            x = scores[keep, layer, expert]
            y = responses[keep, layer, permutations[layer, expert]]
            if len(x) < 10:
                continue
            xs.append((x - x.mean()) / max(x.std(), 1e-8))
            ys.append((y - y.mean()) / max(y.std(), 1e-8))
    if not xs:
        raise ValueError('no layer/expert group has at least ten tokens')
    rho = float(spearmanr(np.concatenate(xs), np.concatenate(ys)).statistic)
    if not np.isfinite(rho):
        raise ValueError('pooled correlation is undefined for constant ranks')
    return rho


def per_token_correlations(
    scores: FloatArray, responses: FloatArray, mask: BoolArray,
) -> Float64Array:
    """Public OLMoE reduction: Spearman across experts for each token/layer.

    Returns (tokens, layers), retaining undefined correlations as NaN rather
    than replacing them with zero. Caller must report valid sample counts.
    """
    _validate(scores, responses, mask)
    result = np.full(scores.shape[:2], np.nan)
    for token, layer in np.ndindex(result.shape):
        keep = mask[token, layer]
        x, y = scores[token, layer, keep], responses[token, layer, keep]
        if len(x) >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0:
            result[token, layer] = spearmanr(x, y).statistic
    return result


def router_cosines(router: FloatArray | IntArray | Sequence[Sequence[float]]) -> Float64Array:
    """All unique unordered pairs of distinct router rows (Figure 3)."""
    matrix = np.asarray(router, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 2 or not np.isfinite(matrix).all():
        raise ValueError('router must be a finite (experts >= 2, hidden) matrix')
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('cosine similarity is undefined for a zero router row')
    unit = matrix / norms
    return (unit @ unit.T)[np.triu_indices(len(matrix), k=1)]
