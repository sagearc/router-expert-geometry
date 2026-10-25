import numpy as np
import pytest

from src.sequences import code_sequences


def test_code_sequences_preserve_first_240_domain_rows(tmp_path):
    path = tmp_path / 'sequences.npz'
    rows = np.broadcast_to(np.arange(482)[:, None], (482, 2048))
    domains = np.tile([0, 1], 241)
    np.savez(path, sequences=rows, domain_idx=domains, domain_names=['dolma_wiki', 'dolma_stack'])
    np.testing.assert_array_equal(code_sequences(path), rows[1:480:2])
    np.savez(path, sequences=rows[:20], domain_idx=domains[:20], domain_names=['dolma_wiki', 'dolma_stack'])
    with pytest.raises(ValueError, match='240 full'):
        code_sequences(path)
