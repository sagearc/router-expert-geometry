"""Rank all four fixed interventions at every checkpoint-setting MoE layer."""

import numpy as np

from src.artifacts import (
    CausalSummary,
    CausalSummaryLayer,
    PairedSummary,
    PublicCausalResult,
)
from src.constants import (
    CAUSAL_BOOTSTRAP_DRAWS,
    CAUSAL_BOOTSTRAP_SEED,
    CAUSAL_CANDIDATES,
    CAUSAL_INTERVENTIONS,
    CAUSAL_LAYERS,
    CAUSAL_PREDICTIONS,
    CAUSAL_SAMPLE_COUNT,
    CI95_QUANTILES,
)
from src.types import Float64Array, FloatArray


def paired_summary(values: FloatArray) -> PairedSummary:
    rng = np.random.default_rng(CAUSAL_BOOTSTRAP_SEED)
    means = values[rng.integers(0, len(values), size=(CAUSAL_BOOTSTRAP_DRAWS, len(values)))].mean(-1)
    return {'mean': float(values.mean()), 'ci95': np.quantile(means, CI95_QUANTILES).tolist()}


def summarize(data: PublicCausalResult) -> CausalSummary:
    protocol = data['protocol']
    if (protocol['candidate_count'] != CAUSAL_CANDIDATES or protocol['predictions_per_sequence'] != CAUSAL_PREDICTIONS
            or protocol['bootstrap_draws'] != CAUSAL_BOOTSTRAP_DRAWS or protocol['bootstrap_seed'] != CAUSAL_BOOTSTRAP_SEED
            or protocol['interventions'] != list(CAUSAL_INTERVENTIONS)
            or set(data['layers']) != {str(layer) for layer in CAUSAL_LAYERS}
            or not data['complete'] or data['backend'] != 'checkpoint_eager_fp32'):
        raise ValueError('expected complete 16-layer, four-intervention, FP32 protocol')
    ids = data['sample_ids']
    if len(ids) != CAUSAL_SAMPLE_COUNT or len(set(ids)) != CAUSAL_SAMPLE_COUNT:
        raise ValueError('expected 176 distinct evaluation sample IDs')

    def losses(values: list[float]) -> Float64Array:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (CAUSAL_SAMPLE_COUNT,) or not np.isfinite(array).all():
            raise ValueError('expected 176 finite sequence losses')
        return array

    baseline = losses(data['baseline_sequence_ce'])
    rows: list[CausalSummaryLayer] = []
    names = [f'candidate{i:02d}' for i in range(CAUSAL_CANDIDATES)]
    for layer in sorted(data['layers'], key=int):
        measured = data['layers'][layer]
        if not 0 <= measured['cached_baseline_max_error'] <= 2e-5 or set(measured['interventions']) != set(CAUSAL_INTERVENTIONS):
            raise ValueError('missing intervention or failed cached baseline')
        for setting in CAUSAL_INTERVENTIONS:
            intervention = measured['interventions'][setting]
            conditions = intervention['conditions']
            if set(conditions) != {'router', *names}:
                raise ValueError('missing or unexpected intervention conditions')
            if not 0 <= intervention['cached_full_max_error'] <= 2e-5 or not all(v['target_routes_unchanged'] for v in conditions.values()):
                raise ValueError('failed cached intervention or changed target routing')
            router = losses(conditions['router']['sequence_ce'])
            controls = np.stack([losses(conditions[name]['sequence_ce']) for name in names])
            means = controls.mean(axis=1)
            strongest = int(np.argmax(means))
            delta = router - controls[strongest]
            rows.append({'layer_zero_based': int(layer), 'intervention': setting,
                     'router_delta_ce': float((router - baseline).mean()),
                     'router_rank_descending': 1 + int((means > router.mean()).sum()),
                     'controls_tied_with_router': int((means == router.mean()).sum()),
                     'strongest_control': names[strongest],
                     'router_minus_strongest_control_ce': float(delta.mean()),
                     'paired_ci95': paired_summary(delta)['ci95'],
                     'relative_ppl_change': float(np.expm1(delta.mean()))})
    return {'revision': data['revision'], 'baseline_ce': float(baseline.mean()),
            'baseline_ppl': float(np.exp(baseline.mean())), 'layers': rows,
            'interval_scope': 'Paired sequence bootstrap, conditional on the selected strongest '
                              'control; not multiplicity-adjusted or stratified by domain.'}
