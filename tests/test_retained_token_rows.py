import numpy as np
import pytest

from src.sequences import final_token_rows


def test_final_token_selection_uses_positions_and_preserves_title_order():
    data = {'router_inputs': np.zeros((768, 1)),
            'sample_title_index': np.tile(np.arange(256), 3),
            'sample_token_index': np.repeat([2, 0, 1], 256)}
    np.testing.assert_array_equal(final_token_rows(data), np.arange(256))
    data.pop('sample_token_index')
    with pytest.raises(ValueError, match='both title and token'):
        final_token_rows(data)


def test_legacy_last_token_capture_requires_256_rows():
    np.testing.assert_array_equal(final_token_rows({'router_inputs': np.zeros((256, 1))}),
                                  np.arange(256))
    with pytest.raises(ValueError, match='256 last-token'):
        final_token_rows({'router_inputs': np.zeros((512, 1))})
