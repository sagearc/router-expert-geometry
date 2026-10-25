import numpy as np

from src.analysis.figure2 import bootstrap_interval


def test_title_bootstrap_matches_retained_source_and_consumes_shared_rng():
    rng = np.random.default_rng(72)
    scores = rng.normal(size=(32, 2, 4)).astype(np.float32)
    responses = rng.normal(size=scores.shape).astype(np.float32)
    mask = np.ones(scores.shape, dtype=bool)
    bootstrap_rng = np.random.default_rng(10000)
    interval = bootstrap_interval(scores, responses, mask, bootstrap_rng)
    # Independently evaluated with retained title_bootstrap_ci, 30 resamples.
    np.testing.assert_allclose(interval, [-0.13307051628453512, 0.021463555638181434],
                               rtol=0, atol=1e-14)
    expected_rng = np.random.default_rng(10000)
    expected_rng.integers(0, 32, size=(30, 32))
    assert bootstrap_rng.bit_generator.state == expected_rng.bit_generator.state
