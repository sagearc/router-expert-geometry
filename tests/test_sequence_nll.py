import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('olmo_core')

from src.capture.ablation import sequence_nll


def test_sequence_nll_shifts_targets_and_is_batch_size_independent():
    class PositionLogits(torch.nn.Module):
        def forward(self, tokens):
            return torch.tensor([[0., 1., 2.], [0., 2., 4.], [0., 3., 6.]]).expand(len(tokens), -1, -1)

    sequences = np.array([[0, 1, 2], [2, 0, 0], [1, 2, 1]])
    log_normalizers = np.log(np.exp([[0., 1., 2.], [0., 2., 4.]]).sum(axis=1))
    expected = log_normalizers.mean() - np.array([2.5, 0., 2.])
    for batch_size in (1, 2, 3):
        np.testing.assert_allclose(sequence_nll(PositionLogits(), sequences, batch_size, 'cpu'),
                                   expected, rtol=1e-6)
