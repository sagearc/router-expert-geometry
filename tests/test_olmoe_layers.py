import numpy as np
import pytest

from src.analysis.olmoe_layers import layer_correlations


def test_layer_correlations_keep_groups_and_titles_separate():
    scores = np.broadcast_to(np.arange(64, dtype=float), (2, 16, 64)).copy()
    response = scores.copy()
    response[1] *= -1
    # Deliberately select a non-top-score group: use actual IDs, not inferred top-k.
    selected = np.broadcast_to(np.arange(16, 24), (2, 16, 8))
    response[:, :, 16:24] = scores[:, :, 16:24]
    result = layer_correlations(scores, scores, response, selected)
    assert result['groups']['routed_top8']['mean'] == pytest.approx([1] * 16)
    assert result['groups']['bottom8']['mean'] == pytest.approx([0] * 16)
    assert result['groups']['bottom8']['title_std'] == pytest.approx([1] * 16)
    assert result['groups']['bottom8']['valid_titles'] == [2] * 16


def test_undefined_correlations_are_counted_not_zero_filled():
    scores = np.broadcast_to(np.arange(64, dtype=float), (2, 16, 64)).copy()
    selected = np.broadcast_to(np.arange(56, 64), (2, 16, 8))
    response = scores.copy()
    response[0] = 1
    result = layer_correlations(scores, scores, response, selected)
    for group in result['groups'].values():
        assert group['mean'] == pytest.approx([1] * 16)
        assert group['undefined_titles'] == [1] * 16
