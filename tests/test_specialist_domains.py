import numpy as np

from src.analysis.specialization import expert_sets


def test_specialist_random_controls_exclude_count_selected_experts_and_share_rng():
    counts = np.broadcast_to(np.arange(64)[None, :, None], (9, 64, 2))
    actual = list(expert_sets(counts, ['dolma_stack', 'dolma_pes2o']))
    rng = np.random.default_rng(0)
    for target, top, random in actual:
        np.testing.assert_array_equal(top, np.tile([63, 62, 61, 60], (9, 1)))
        expected = np.stack([rng.choice(list(range(60)), size=4, replace=False) for _ in range(9)])
        np.testing.assert_array_equal(random, expected)
    assert [x[0] for x in actual] == ['dolma_stack', 'dolma_pes2o']
