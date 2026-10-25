import json

import pytest

pytest.importorskip('matplotlib')
from src.plots.figure2_dynamics import read_result


def test_dynamics_plot_rejects_wrong_metric_and_inconsistent_final_checkpoint(tmp_path):
    rows = [{'step': step, 'routed': .4, 'nonrouted': .02,
             'routed_ci': [.3, .5], 'nonrouted_ci': [.01, .03],
             'identity_permuted_routed': .1}
            for step in (0, 1000, 5000, 10000, 20000, 39750)]
    result = {'metric': 'w1_top10_l2', 'dynamics': rows,
              'final': {'routed': .4, 'nonrouted': .02,
                        'identity_permuted_routed': .1, 'token_shuffled_routed': .00015}}
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(result))
    assert read_result(path)[0] == result
    result['metric'] = 'w1_l2'
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='signed top-10'):
        read_result(path)
    result['metric'] = 'w1_top10_l2'
    result['final']['routed'] = .5
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='disagrees'):
        read_result(path)
