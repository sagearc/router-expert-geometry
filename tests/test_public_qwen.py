import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from src.hf.qwen import rank_responses


def test_qwen_rank_response_uses_valid_tokens_packed_inputs_and_stable_ranks(monkeypatch):
    config = Qwen3MoeConfig(hidden_size=8, intermediate_size=16, moe_intermediate_size=8,
                           num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                           num_experts=20, num_experts_per_tok=2, vocab_size=32,
                           attn_implementation='eager', experts_implementation='eager')
    model = Qwen3MoeForCausalLM(config).eval()
    moe = model.model.layers[0].mlp
    # Equal probabilities must produce expert-ID order in the measurement,
    # independently of the native top-k tie-breaking used for dispatch.
    with torch.no_grad():
        moe.gate.weight.zero_()
    seen = []
    hook = moe.gate.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    tokens = torch.tensor([[1, 2, 3], [4, 5, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    chunks = []
    original_einsum = torch.einsum

    def record_projection(equation, *operands):
        if equation == 'ehd,bd->beh':
            chunks.append(operands[0].shape[0])
        return original_einsum(equation, *operands)

    monkeypatch.setattr(torch, 'einsum', record_projection)
    with torch.inference_mode():
        before = model(tokens, attention_mask=mask, use_cache=False).logits
        sums, count, differences = rank_responses(model, tokens, mask)
        after = model(tokens, attention_mask=mask, use_cache=False).logits
    hook.remove()
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert count == 5 and differences == 0
    assert chunks == [16, 16, 4, 4]
    hidden = seen[0][mask.bool().flatten()]
    expected = []
    for weights in moe.experts.gate_up_proj.detach().chunk(2, dim=1):
        expected.append(np.array([(hidden @ w.T).norm(dim=-1).double().sum().item() for w in weights]))
    np.testing.assert_allclose(sums, expected, rtol=1e-6, atol=1e-7)
    assert not moe.gate._forward_hooks


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
@pytest.mark.parametrize('corrupt', ['router', 'expert'])
def test_qwen_capture_rejects_nonfinite_values_and_removes_hooks(value, corrupt):
    config = Qwen3MoeConfig(hidden_size=8, intermediate_size=16, moe_intermediate_size=8,
                           num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                           num_experts=4, num_experts_per_tok=2, vocab_size=32,
                           attn_implementation='eager', experts_implementation='eager')
    model = Qwen3MoeForCausalLM(config).eval()
    moe = model.model.layers[0].mlp
    with torch.no_grad():
        weights = moe.gate.weight if corrupt == 'router' else moe.experts.gate_up_proj
        weights[0].fill_(value)
    tokens = torch.tensor([[1, 2]])
    with pytest.raises(ValueError, match='nonfinite'):
        rank_responses(model, tokens, torch.ones_like(tokens))
    assert not moe.gate._forward_hooks


def test_qwen_capture_rejects_dense_layer_and_removes_preceding_hooks():
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP

    config = Qwen3MoeConfig(
        hidden_size=8, intermediate_size=16, moe_intermediate_size=8,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        num_experts=4, num_experts_per_tok=2, vocab_size=32,
        attn_implementation='eager', experts_implementation='eager',
    )
    model = Qwen3MoeForCausalLM(config).eval()
    model.model.layers[1].mlp = Qwen3MoeMLP(config).eval()
    tokens = torch.tensor([[1, 2]])
    with pytest.raises(ValueError, match='sparse MoE block'):
        rank_responses(model, tokens, torch.ones_like(tokens))
    assert not model.model.layers[0].mlp.gate._forward_hooks
