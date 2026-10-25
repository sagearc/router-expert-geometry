import numpy as np
import pytest

from src.analysis.specialization import (
    paired_damage_ratio,
    specialist_weight_share,
)


def test_normalization_uses_ablated_experts_not_largest_weights():
    counts = np.array([[9, 1, 0], [1, 9, 0]])
    weights = np.array([[1., 8., 1.], [7., 2., 1.]])
    assert specialist_weight_share(counts, weights, 1) == pytest.approx(.15)


def test_paired_bootstrap_preserves_exact_sequence_relation():
    aux = np.array([.01, .1, 1., 10.])
    result = paired_damage_ratio(aux, 3 * aux, .2, .4, resamples=100)
    assert result['raw_ratio'] == pytest.approx(3)
    assert result['raw_ci95'] == pytest.approx([3, 3])
    assert result['weight_mass_corrected_ci95'] == pytest.approx([1.5, 1.5])


def test_undefined_damage_ratio_is_not_reported():
    with pytest.raises(ValueError, match='positive'):
        paired_damage_ratio([0, 0], [1, 1], .2, .3)
