import json
import sys

import numpy as np
import pytest

pytest.importorskip('matplotlib')
from src.analysis.rank_curve import rank_curve
from src.cli import plot_public_rank

runner = vars(plot_public_rank)


def test_rank_plot_cli_infers_qwen_from_metadata(tmp_path, monkeypatch):
    capture, output = tmp_path / 'qwen.npz', tmp_path / 'rank.svg'
    mean = np.arange(256, dtype=float).reshape(2, 128)
    np.savez(capture, mean=mean, expert_rank=np.arange(1, 129), top_k=8,
             metric_names=['w1_l2', 'w3_l2'],
             metadata_json=json.dumps({'model': 'Qwen/Qwen3-30B-A3B-Base'}))
    monkeypatch.setattr(sys, 'argv', ['plot_public_rank', str(capture), '--out', str(output)])
    runner['main']()
    assert output.is_file()
    result = json.loads(output.with_suffix('.json').read_text())
    assert result == rank_curve(capture)
    assert result['model'] == 'Qwen/Qwen3-30B-A3B-Base'
    np.testing.assert_array_equal(result['response_mean'], mean)


def test_retained_qwen_compact_schema_identifies_model_without_metadata(tmp_path):
    capture = tmp_path / 'legacy.npz'
    np.savez(capture, mean=np.ones((2, 128)), expert_rank=np.arange(1, 129), top_k=8,
             metric_names=['w1_l2', 'w3_l2'])
    assert rank_curve(capture)['top_k'] == 8


def test_rank_plot_rejects_unknown_model_metadata(tmp_path):
    capture = tmp_path / 'invalid.npz'
    np.savez(capture, metadata_json=json.dumps({'model': 'unknown-model'}))
    with pytest.raises(ValueError, match='unsupported capture model'):
        rank_curve(capture)
