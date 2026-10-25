import json
from pathlib import Path

import pytest

pytest.importorskip('matplotlib')
from src.plots.olmoe_layers import read_result

ROOT = Path(__file__).resolve().parents[1]


def test_layer_plot_requires_correlations_and_consistent_title_counts(tmp_path):
    verified = json.loads((ROOT / 'validation/olmoe-5k-layer-result.json').read_text())
    result = verified['RECOMPUTED_LAYER_TABLE']
    path = tmp_path / 'layers.json'
    path.write_text(json.dumps(result))
    assert read_result(path)['n_titles'] == 256
    result['response'] = 'mean_raw_l2'
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='correlation'):
        read_result(path)
    result['response'] = 'l2_norm_w_gate_h'
    result['groups']['bottom8']['valid_titles'][0] = 255
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='title counts'):
        read_result(path)
