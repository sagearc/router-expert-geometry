import numpy as np
import pytest

from src.analysis.kmeans_errors import boundary_score, decile_gaps


def test_boundary_standardizes_each_layer_before_averaging():
    base = np.arange(20, dtype=float).reshape(2, 10)
    margins = np.stack([base * (i + 1) + i * 100 for i in range(9)])
    np.testing.assert_allclose(boundary_score(margins), (base - base.mean()) / base.std())
    with pytest.raises(ValueError, match='9, sequences'):
        boundary_score(margins[:8])


def test_deciles_preserve_ties_and_bootstrap_sequences():
    # All tied values belong to one bin; two sequences have constant gaps 1/3.
    score = np.ones((2, 20))
    delta = np.broadcast_to(np.array([[1.], [3.]]), score.shape)
    bins = decile_gaps(score, delta)
    assert all(b['count'] == 0 and b['mean_nll_gap'] is None for b in bins[:9])
    assert bins[9]['count'] == 40 and bins[9]['mean_nll_gap'] == 2
    assert bins[9]['ci95'] == [1., 3.]
    assert bins[9]['valid_bootstraps'] == 1000
