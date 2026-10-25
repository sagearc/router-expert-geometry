import numpy as np
import pytest

from src.analysis.measurements import (
    per_token_correlations,
    pooled_correlation,
    router_cosines,
    selected_mask,
)


def test_pairing_control_preserves_source_selection():
    x = np.linspace(-1, 1, 20)
    scores = np.stack([x, -x], axis=-1)[:, None, :]
    mask = selected_mask(np.zeros((20, 1, 1), dtype=int), 2)
    assert pooled_correlation(scores, scores, mask) == pytest.approx(1)
    assert pooled_correlation(scores, scores, mask, [[1, 0]]) == pytest.approx(-1)


def test_public_reduction_uses_matching_experts_and_handles_ties():
    scores = np.array([[[1., 2., 2., 4.]], [[0., 0., 0., 0.]]])
    responses = -scores
    result = per_token_correlations(scores, responses, np.ones_like(scores, dtype=bool))
    assert result[0, 0] == pytest.approx(-1)
    assert np.isnan(result[1, 0])


def test_small_groups_are_not_silently_reported_as_zero():
    scores = np.ones((9, 1, 2))
    with pytest.raises(ValueError, match='ten tokens'):
        pooled_correlation(scores, scores, np.ones_like(scores, dtype=bool))


def test_selected_ids_reject_duplicates_and_out_of_range():
    for ids in ([[[0, 0]]], [[[0, 3]]]):
        with pytest.raises(ValueError):
            selected_mask(np.array(ids), 3)


def test_router_geometry_excludes_diagonal_and_double_counting():
    pairs = router_cosines([[1, 0], [0, 1], [-1, 0]])
    np.testing.assert_allclose(pairs, [0, -1, 0])
    with pytest.raises(ValueError, match='zero router row'):
        router_cosines([[0, 0], [1, 0]])


def test_nonfinite_capture_is_rejected():
    scores = np.ones((10, 1, 2))
    scores[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        pooled_correlation(scores, scores, np.ones_like(scores, dtype=bool))
