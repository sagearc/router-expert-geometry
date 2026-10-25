"""Specialist-ablation reductions. Losses are NLL, not perplexity."""

import hashlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TypedDict, cast

import numpy as np

from src.constants import (
    CI95_PERCENTILES,
    PAPER_EXPERTS,
    SPECIALIST_CODE_DOMAIN,
    SPECIALIST_CONTROL_EXPERTS,
    SPECIALIST_CONTROL_SEED,
)
from src.types import BalancedRouting, FloatArray, IntArray


class DamageRatio(TypedDict):
    paired_sequences: int
    resamples: int
    seed: int
    raw_ratio: float
    raw_ci95: list[float]
    weight_mass_corrected_ratio: float
    weight_mass_corrected_ci95: list[float]
    aux_weight_share: float
    loss_free_weight_share: float


def specialist_weight_share(counts: IntArray, weights: FloatArray, r: int) -> float:
    """Weight mass of count-selected experts, averaged equally across layers.

    counts and weights refer to the same target domain, shape (layers, experts).
    Do not independently rank weights: that would normalize different experts
    from those actually ablated.
    """
    counts, weights = np.asarray(counts), np.asarray(weights)
    if counts.ndim != 2 or counts.shape != weights.shape or not counts.size:
        raise ValueError('counts and weights must share nonempty (layers, experts) shape')
    if not 1 <= r <= counts.shape[1]:
        raise ValueError('invalid number of specialist experts')
    if not np.isfinite(counts).all() or not np.isfinite(weights).all():
        raise ValueError('routing mass must be finite')
    if (counts < 0).any() or (weights < 0).any() or (weights.sum(axis=1) <= 0).any():
        raise ValueError('routing mass must be nonnegative with positive layer totals')
    # Same ordering as the retained ablation selection, including its tie rule.
    selected = np.argsort(-counts, axis=1)[:, :r]
    selected_weights = np.take_along_axis(weights, selected, axis=1).sum(axis=1)
    return float(np.mean(selected_weights / weights.sum(axis=1)))


def paired_damage_ratio(
    aux_damage: FloatArray | Sequence[float], loss_free_damage: FloatArray | Sequence[float],
    aux_share: float, loss_free_share: float, *, resamples: int = 10_000, seed: int = 0,
) -> DamageRatio:
    """Paired sequence bootstrap with fixed routing-mass normalization.

    Damage is ablated NLL minus clean NLL on identical sequences. The interval
    conditions on the measured mass shares; it does not bootstrap their error.
    """
    aux = np.asarray(aux_damage, dtype=np.float64)
    lfb = np.asarray(loss_free_damage, dtype=np.float64)
    if aux.ndim != 1 or aux.shape != lfb.shape or len(aux) < 2:
        raise ValueError('damage arrays must contain the same paired sequences')
    if not np.isfinite(aux).all() or not np.isfinite(lfb).all():
        raise ValueError('damage must be finite')
    if not 0 < aux_share <= 1 or not 0 < loss_free_share <= 1 or resamples < 1:
        raise ValueError('invalid mass shares or resample count')
    if aux.mean() <= 0:
        raise ValueError('auxiliary-loss mean damage must be positive')
    correction = aux_share / loss_free_share
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples)
    for i in range(resamples):
        indices = rng.integers(0, len(aux), len(aux))
        denominator = aux[indices].mean()
        if denominator <= 0:
            raise ValueError('bootstrap denominator is nonpositive; ratio interval is unstable')
        samples[i] = lfb[indices].mean() / denominator
    raw = float(lfb.mean() / aux.mean())
    return {
        'paired_sequences': len(aux), 'resamples': resamples, 'seed': seed,
        'raw_ratio': raw,
        'raw_ci95': np.percentile(samples, CI95_PERCENTILES).tolist(),
        'weight_mass_corrected_ratio': raw * correction,
        'weight_mass_corrected_ci95': np.percentile(samples * correction, CI95_PERCENTILES).tolist(),
        'aux_weight_share': aux_share, 'loss_free_weight_share': loss_free_share,
    }


def counts_routing(counts_path: Path, sequences_path: Path) -> BalancedRouting:
    """The routing a domain-counts capture was taken with, after checking its inputs.

    Ablating experts chosen from another model's counts, or counted on other
    sequences, would be invalid, so both are read from the capture's metadata.
    """
    with np.load(counts_path, allow_pickle=False) as data:
        if 'metadata_json' not in data:
            raise ValueError('domain counts must come from capture-domain-counts, with metadata')
        metadata = json.loads(str(data['metadata_json']))
    routing = metadata.get('routing')
    if routing not in ('aux_loss', 'loss_free'):
        raise ValueError(f'domain counts record unsupported routing {routing!r}')
    with sequences_path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != metadata.get('sequences_sha256'):
            raise ValueError('domain counts were captured on different sequences')
    return cast(BalancedRouting, routing)


def load_damage_and_share(
    loss_path: Path, routing_path: Path, domain: str, r: int, routing: BalancedRouting,
) -> tuple[FloatArray, float]:
    """Per-sequence ablation damage and the ablated experts' weight share.

    The ablation capture records the hash of the routing file whose counts chose
    the ablated experts; a different routing file is rejected.
    """
    with np.load(loss_path, allow_pickle=False) as data:
        damage = data[f'r{r}'] - data['clean']
        metadata = json.loads(str(data['metadata_json'])) if 'metadata_json' in data else {}
    if metadata.get('domain', domain) != domain:
        raise ValueError('requested domain differs from the ablation capture')
    if metadata.get('routing', routing) != routing:
        raise ValueError(f'expected a {routing} ablation capture, got {metadata["routing"]}')
    if 'counts_sha256' in metadata:
        with Path(routing_path).open('rb') as stream:
            routing_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        if routing_hash != metadata['counts_sha256']:
            raise ValueError('routing statistics differ from those used to select the ablated experts')
    with np.load(routing_path, allow_pickle=False) as data:
        index = list(data['domains']).index(domain)
        share = specialist_weight_share(data['counts'][:, :, index], data['counts_w'][:, :, index], r)
    return cast(FloatArray, damage), share


def expert_sets(
    counts: IntArray, domains: Sequence[str],
) -> Iterator[tuple[str, IntArray, IntArray]]:
    """Retained target order and RNG consumption; random excludes top four."""
    rng = np.random.default_rng(SPECIALIST_CONTROL_SEED)
    for target in (SPECIALIST_CODE_DOMAIN, 'dolma_pes2o'):
        selected = np.argsort(-counts[:, :, list(domains).index(target)], axis=1)[:, :SPECIALIST_CONTROL_EXPERTS]
        random = np.stack([rng.choice([e for e in range(PAPER_EXPERTS) if e not in row], size=SPECIALIST_CONTROL_EXPERTS, replace=False)
                           for row in selected])
        yield target, selected, random
