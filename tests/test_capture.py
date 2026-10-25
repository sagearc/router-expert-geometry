import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('olmo_core')

from olmo_core.nn.moe.mlp import DroplessMoEMLP
from olmo_core.nn.transformer import TransformerConfig

from src.capture.last_token import capture_last_tokens, expert_responses
from src.capture.token_losses import capture_token_losses


def test_gate_and_up_response_layout_and_signed_top10():
    mlp = DroplessMoEMLP(d_model=2, hidden_size=12, num_experts=2)
    with torch.no_grad():
        mlp.w1.copy_(torch.arange(-24, 24).reshape(24, 2))
        mlp.w3.copy_(-mlp.w1 * 2)
    h = torch.tensor([[1., 0.], [0., 1.]])
    result = expert_responses(h, mlp)
    gate = torch.stack([h @ w.T for w in mlp.w1.view(2, 12, 2)], dim=1)
    torch.testing.assert_close(result['w1_l2'], gate.norm(dim=-1))
    torch.testing.assert_close(result['w3_l2'], 2 * gate.norm(dim=-1))
    torch.testing.assert_close(result['w1_top10_l2'], gate.sort(dim=-1).values[..., -10:].norm(dim=-1))
    assert not torch.equal(result['w1_top10_l2'], gate.abs().topk(10).values.norm(dim=-1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='dropless dispatch requires CUDA kernels')
def test_last_token_capture_matches_direct_expert_responses_without_changing_model():
    config = TransformerConfig.llama_like_moe(
        d_model=16, vocab_size=32, n_layers=1, n_heads=2,
        num_experts=4, top_k=2, expert_hidden_size=12,
        dropless=True, reordered_norm=True,
    )
    model = config.build(init_device='cuda')
    model.init_weights(max_seq_len=4)
    model.eval()
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]], device='cuda')
    lengths = torch.tensor([4, 2], device='cuda')
    moe = model.blocks['0'].feed_forward_moe
    hidden = []
    handle = moe.router.register_forward_pre_hook(lambda module, args: hidden.append(args[0].detach()))
    with torch.inference_mode():
        before = model(ids)
        result = capture_last_tokens(model, ids, lengths)
        after = model(ids)
    handle.remove()
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    h = hidden[0][torch.arange(2, device='cuda'), lengths - 1].float()
    mlp = moe.experts.mlp
    for name, weights in [('w1_l2', mlp.w1), ('w3_l2', mlp.w3)]:
        expected = torch.stack([
            (h @ w.T.float()).norm(dim=-1)
            for w in weights.detach().view(4, 12, 16)
        ], dim=-1)
        torch.testing.assert_close(result[name][:, 0], expected.cpu())
    gate = torch.einsum('bd,ehd->beh', h, mlp.w1.detach().view(4, 12, 16).float())
    torch.testing.assert_close(result['w1_top10_l2'][:, 0], gate.topk(10).values.norm(dim=-1).cpu())
    assert result['selected_experts'].shape == (2, 1, 2)
    assert not moe.router._forward_hooks


@pytest.mark.skipif(not torch.cuda.is_available(), reason='dropless dispatch requires CUDA kernels')
def test_token_loss_capture_preserves_forward_and_next_token_alignment():
    config = TransformerConfig.llama_like_moe(
        d_model=16, vocab_size=32, n_layers=1, n_heads=2,
        num_experts=8, top_k=6, expert_hidden_size=12,
        dropless=True, reordered_norm=True,
    )
    model = config.build(init_device='cuda')
    model.init_weights(max_seq_len=4)
    model.eval()
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device='cuda')
    with torch.inference_mode():
        before = model(tokens)
        captured = capture_token_losses(model, tokens)
        after = model(tokens)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    expected = -before[:, :-1].float().log_softmax(-1).gather(-1, tokens[:, 1:, None]).squeeze(-1)
    torch.testing.assert_close(captured['nll'], expected.cpu())
    torch.testing.assert_close(captured['targets'], tokens[:, 1:].cpu())
    assert captured['margin6'].shape == (1, 2, 3)
    assert (captured['margin6'] >= 0).all()
    assert 'get_top_k' not in model.blocks['0'].feed_forward_moe.router.__dict__
