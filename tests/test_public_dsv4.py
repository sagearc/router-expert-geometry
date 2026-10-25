from types import SimpleNamespace

import pytest
import torch

from src.hf.dsv4 import unpack_responses


def test_packed_gate_up_responses_restore_token_expert_order():
    # Native kernel output is expert-grouped, not token-major.
    original = torch.tensor([[3., 4., 0., 2.], [0., 6., 8., 0.],
                             [5., 12., 0., 9.], [0., 8., 15., 0.]])
    packed = original[[2, 0, 3, 1]].unsqueeze(0).bfloat16()
    destination = torch.tensor([1, 3, 0, 2])
    actual = unpack_responses(packed, destination, rows=2, experts_per_token=2)
    torch.testing.assert_close(actual, torch.tensor([[[5., 2.], [6., 8.]],
                                                     [[13., 9.], [8., 15.]]]))
    with pytest.raises(ValueError, match='dimensions'):
        unpack_responses(packed[..., :3], destination, 2, 2)


def test_capture_excludes_hash_layers_checks_coverage_and_removes_hooks(monkeypatch):
    from src.hf import dsv4 as capture

    class Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.e_score_correction_bias = torch.arange(256).float()

        def forward(self, hidden):
            return torch.zeros((len(hidden), 256)), None

    layers = [SimpleNamespace(ffn=SimpleNamespace(
        gate=Gate(), n_routed_experts=256,
        experts=SimpleNamespace(routed_experts=None))) for _ in range(43)]
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))

    def responses(routed, hidden, ids):
        torch.testing.assert_close(ids[0], torch.arange(255, -1, -1))
        return hidden[:, :1, None].expand(-1, 256, 2)

    monkeypatch.setattr(capture, 'expert_rank_responses', responses)
    capture.install_capture(model, 3)
    assert all(not layer.ffn.gate._forward_hooks for layer in layers[:3])
    for layer in layers:
        layer.ffn.gate(torch.tensor([[1.], [2.]]))
        layer.ffn.gate(torch.tensor([[3.]]))
    sums, counts = capture.finish_capture(model)
    assert sums.shape == (40, 256, 2) and (sums == 6).all()
    assert (counts == 3).all()
    assert all(not layer.ffn.gate._forward_hooks for layer in layers)
    capture.install_capture(model, 3)
    with pytest.raises(ValueError, match='incomplete'):
        capture.finish_capture(model)
    assert not hasattr(model, '_rank_capture')
    assert all(not layer.ffn.gate._forward_hooks for layer in layers)
