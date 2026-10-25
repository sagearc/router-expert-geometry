import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('olmo_core')
from olmo_core.nn.moe.router import MoELinearRouter

from src.capture.token_losses import (
    capture_token_losses,
    selection_margins,
)
from src.sequences import domain_batches


def test_capture_batches_restart_at_domain_boundaries_without_dropping_rows():
    batches = domain_batches
    domains = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2])
    rows = list(batches(domains, 4))
    assert [(r.start, r.stop) for r in rows] == [(0, 3), (3, 7), (7, 8), (8, 9)]


def test_native_biased_boundary_keeps_outputs_and_restores_method_on_failure():
    router = MoELinearRouter(d_model=2, num_experts=8, top_k=6, bias_gamma=0.01).eval()
    scores = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    router.score_bias[0] = 20
    before = router.get_top_k(scores)
    with pytest.raises(RuntimeError, match='test interruption'), selection_margins([router]) as records:
        after = router.get_top_k(scores)
        torch.testing.assert_close(after[0], before[0], rtol=0, atol=0)
        torch.testing.assert_close(after[1], before[1], rtol=0, atol=0)
        assert records[0].item() == 1  # ranked scores 20,7,6,5,4,3 | 2,1
        raise RuntimeError('test interruption')
    assert 'get_top_k' not in router.__dict__
    torch.testing.assert_close(router.get_top_k(scores)[1], before[1])


def test_losses_and_all_layer_margins_align_with_next_token_targets():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleDict()
            for layer in range(2):
                block = torch.nn.Module()
                block.feed_forward_moe = torch.nn.Module()
                block.feed_forward_moe.router = MoELinearRouter(
                    d_model=2, num_experts=8, top_k=6,
                )
                self.blocks[str(layer)] = block

        def forward(self, tokens):
            for layer, block in enumerate(self.blocks.values()):
                # Distinct gaps at every token, batch row and layer expose
                # accidentally shifted or flattened margin arrays.
                scores = (tokens[..., None] + layer + 1).float() * torch.arange(8.)
                block.feed_forward_moe.router.get_top_k(scores)
            return torch.nn.functional.one_hot(tokens, num_classes=8).float() * 3

    tokens = torch.tensor([[1, 1, 2, 7], [3, 0, 0, 5]])
    result = capture_token_losses(Model().eval(), tokens)
    torch.testing.assert_close(result['targets'], tokens[:, 1:])
    expected_nll = torch.nn.functional.cross_entropy(
        (torch.nn.functional.one_hot(tokens[:, :-1], 8).float() * 3).reshape(-1, 8),
        tokens[:, 1:].reshape(-1), reduction='none',
    ).reshape(2, 3)
    torch.testing.assert_close(result['nll'], expected_nll)
    assert set(result) == {'nll', 'targets', 'margin6'}
    torch.testing.assert_close(result['margin6'], torch.stack([
        tokens[:, :-1].float() + 1, tokens[:, :-1].float() + 2,
    ]))
    broken = Model().eval()
    broken.register_forward_hook(lambda module, args, output: output * float('nan'))
    with pytest.raises(ValueError, match='nonfinite per-token loss'):
        capture_token_losses(broken, tokens)
    assert all('get_top_k' not in block.feed_forward_moe.router.__dict__
               for block in broken.blocks.values())
