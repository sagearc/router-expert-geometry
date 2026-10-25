"""Occurrence-weighted loss gaps for the paper's rare-token/boundary analysis."""

import json
from collections.abc import Mapping
from typing import NotRequired, TypedDict, cast

import numpy as np

from src.constants import (
    CI95_PERCENTILES,
    ERROR_BOOTSTRAP_DRAWS,
    ERROR_BOOTSTRAP_SEED,
    PAPER_LAYERS,
    PAPER_VOCAB_SIZE,
)
from src.types import Float64Array, FloatArray, IntArray


class DecileGap(TypedDict):
    bin: int
    count: int
    mean_nll_gap: float | None
    ci95: list[float] | None
    valid_bootstraps: int


class ErrorBootstrap(TypedDict):
    unit: str
    resamples: int
    seed_per_measurement: int


class ErrorAnalysis(TypedDict):
    n_sequences: int
    n_tokens: int
    mean_nll_gap: float
    frequency_deciles: list[DecileGap]
    boundary_deciles: list[DecileGap]
    bootstrap: NotRequired[ErrorBootstrap]
    sequence_hash_verified: NotRequired[bool]
    margin_alignment: NotRequired[str]


def boundary_score(margins: FloatArray) -> Float64Array:
    """Z-score the top-6/top-7 gap per layer, then average across nine layers."""
    margins = np.asarray(margins)
    if margins.ndim != 3 or margins.shape[0] != PAPER_LAYERS or not np.isfinite(margins).all():
        raise ValueError('expected finite (9, sequences, positions) margins')
    result = np.zeros(margins.shape[1:], dtype=np.float64)
    for layer in margins:
        values = layer.astype(np.float64)
        result += (values - values.mean()) / (values.std() + 1e-12)
    return result / PAPER_LAYERS


def decile_gaps(
    score: FloatArray, delta: FloatArray, *, seed: int = ERROR_BOOTSTRAP_SEED, n_boot: int = ERROR_BOOTSTRAP_DRAWS,
) -> list[DecileGap]:
    """Quantile bins preserve score ties; bootstrap whole sequences, not tokens.

    Ties can produce unequal or empty bins. Report counts and null empty-bin
    estimates instead of describing every bin as exactly 10% of occurrences.
    """
    score, delta = np.asarray(score), np.asarray(delta, dtype=np.float64)
    if delta.ndim != 2 or score.shape != delta.shape or not delta.size:
        raise ValueError('score and loss gap must have matching nonempty sequence/position axes')
    if not np.isfinite(score).all() or not np.isfinite(delta).all() or n_boot < 1:
        raise ValueError('finite inputs and positive bootstrap count required')
    edges = np.quantile(score, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    bins = np.clip(np.searchsorted(edges, score, side='right') - 1, 0, PAPER_LAYERS)
    n_seqs = len(delta)
    flat = (np.arange(n_seqs)[:, None] * 10 + bins).ravel()
    sums = np.bincount(flat, weights=delta.ravel(), minlength=n_seqs * 10).reshape(n_seqs, 10)
    counts = np.bincount(flat, minlength=n_seqs * 10).reshape(n_seqs, 10)
    rng = np.random.default_rng(seed)
    boots = np.full((n_boot, 10), np.nan)
    for i in range(n_boot):
        weights = rng.multinomial(n_seqs, np.full(n_seqs, 1 / n_seqs))
        denominator = weights @ counts
        np.divide(weights @ sums, denominator, out=boots[i], where=denominator > 0)
    result: list[DecileGap] = []
    for b, count in enumerate(counts.sum(0)):
        valid = boots[:, b][np.isfinite(boots[:, b])]
        result.append({
            'bin': b, 'count': int(count),
            'mean_nll_gap': float(sums[:, b].sum() / count) if count else None,
            'ci95': np.percentile(valid, CI95_PERCENTILES).tolist() if len(valid) else None,
            'valid_bootstraps': len(valid),
        })
    return result


def analyze(
    kmeans_nll: FloatArray, reference_nll: FloatArray, targets: IntArray, margins: FloatArray,
) -> ErrorAnalysis:
    kmeans_nll = np.asarray(kmeans_nll)
    reference_nll = np.asarray(reference_nll)
    targets = np.asarray(targets)
    if kmeans_nll.shape != reference_nll.shape or targets.shape != kmeans_nll.shape:
        raise ValueError('NLL and targets must have identical shapes')
    if not np.issubdtype(targets.dtype, np.integer) or (targets < 0).any() or (targets >= PAPER_VOCAB_SIZE).any():
        raise ValueError('targets must be paper-vocabulary token IDs')
    delta = kmeans_nll.astype(np.float64) - reference_nll.astype(np.float64)
    counts = np.bincount(targets.ravel().astype(np.int64), minlength=PAPER_VOCAB_SIZE)
    log_frequency = np.log10(counts[targets])
    score = boundary_score(margins)
    return {'n_sequences': len(targets), 'n_tokens': targets.size,
            'mean_nll_gap': float(delta.mean()),
            'frequency_deciles': decile_gaps(log_frequency, delta),
            'boundary_deciles': decile_gaps(score, delta)}


def paired_error_analysis(km: Mapping[str, FloatArray | IntArray], ref: Mapping[str, FloatArray | IntArray]) -> ErrorAnalysis:
    """Reduce a K-means capture against its matched loss-free reference.

    Both NPZ files must agree on targets and domains; when they record input
    hashes, the hashes and the declared routings must match as well.
    """
    if 'margin6' not in km:
        raise ValueError('K-means capture must contain both nll and margin6')
    for key in ('targets', 'domain_idx', 'domain_names'):
        if not np.array_equal(km[key], ref[key]):
            raise ValueError(f'captures disagree on {key}; paired analysis is invalid')
    metadata = [json.loads(str(data['metadata_json'])) if 'metadata_json' in data else {}
                for data in (km, ref)]
    hashes = [item.get('sequences_sha256') for item in metadata]
    if any(hashes) and (not all(hashes) or len(set(hashes)) != 1):
        raise ValueError('sequence hashes must agree across NLL and margin captures')
    if all(hashes):
        for item, routing in zip(metadata, ('kmeans', 'loss_free'), strict=True):
            if item.get('routing') != routing:
                raise ValueError(f'expected {routing} capture, got {item.get("routing")}')
    result = analyze(cast(FloatArray, km['nll']), cast(FloatArray, ref['nll']),
                     cast(IntArray, km['targets']), cast(FloatArray, km['margin6']))
    result['bootstrap'] = {'unit': 'sequence', 'resamples': ERROR_BOOTSTRAP_DRAWS, 'seed_per_measurement': ERROR_BOOTSTRAP_SEED}
    result['sequence_hash_verified'] = bool(all(hashes))
    result['margin_alignment'] = ('shared input hash' if all(hashes) else
                                  'shared K-means capture; reference matched by targets and domains')
    return result
