import numpy as np
import pytest

from src.analysis.causal_summary import summarize
from src.constants import CAUSAL_INTERVENTIONS


def capture():
    base = 2 + ((np.arange(176) % 8) - 3.5) / 128
    noise = np.tile(np.array([-2., -1., 1., 2.]) / 512, 44)
    layers = {}
    for layer in range(16):
        interventions = {}
        for index, setting in enumerate(CAUSAL_INTERVENTIONS):
            conditions = {f'candidate{i:02d}': {'sequence_ce': (base + i / 128 + noise).tolist(),
                                              'target_routes_unchanged': True} for i in range(32)}
            conditions['router'] = {'sequence_ce': (base + (25 + index + layer % 2) / 128 + 2 * noise).tolist(),
                                    'target_routes_unchanged': True}
            interventions[setting] = {'conditions': conditions, 'cached_full_max_error': 0.}
        layers[str(layer)] = {'interventions': interventions, 'cached_baseline_max_error': 0.}
    return {'revision': 'test', 'complete': True, 'backend': 'checkpoint_eager_fp32',
            'sample_ids': [str(i) for i in range(176)], 'baseline_sequence_ce': base.tolist(),
            'protocol': {'candidate_count': 32, 'predictions_per_sequence': 512,
                         'bootstrap_draws': 10000, 'bootstrap_seed': 19871,
                         'interventions': list(CAUSAL_INTERVENTIONS)}, 'layers': layers}


def test_all_layer_method_ranks_and_nonconstant_paired_uncertainty():
    result = summarize(capture())
    assert result['baseline_ppl'] == np.exp(2.) and len(result['layers']) == 64
    for row in result['layers']:
        index = list(CAUSAL_INTERVENTIONS).index(row['intervention'])
        offset = index + row['layer_zero_based'] % 2
        delta = (-6 + offset) / 128
        assert row['strongest_control'] == 'candidate31'
        assert row['router_rank_descending'] == 7 - offset
        assert row['controls_tied_with_router'] == 1
        low, high = row['paired_ci95']
        assert delta - .004 < low < delta < high < delta + .004
        assert row['relative_ppl_change'] == pytest.approx(np.expm1(delta))


@pytest.mark.parametrize('failure', ['method', 'condition', 'nonfinite', 'routing', 'cache'])
def test_summary_rejects_incomplete_or_invalid_measurements(failure):
    data = capture()
    intervention = data['layers']['15']['interventions']['relative-0.1']
    if failure == 'method':
        del data['layers']['15']['interventions']['rotation-0.1']
    elif failure == 'condition':
        del intervention['conditions']['candidate31']
    elif failure == 'nonfinite':
        intervention['conditions']['router']['sequence_ce'][0] = float('nan')
    elif failure == 'routing':
        intervention['conditions']['router']['target_routes_unchanged'] = False
    else:
        intervention['cached_full_max_error'] = float('nan')
    with pytest.raises(ValueError):
        summarize(data)
