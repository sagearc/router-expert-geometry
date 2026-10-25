import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
from transformers import OlmoeConfig, OlmoeForCausalLM

from src.hf.olmoe import capture_olmoe, expert_response_norms


def test_native_olmoe_capture_matches_gate_up_not_down_and_preserves_forward(monkeypatch):
    config = OlmoeConfig(hidden_size=16, intermediate_size=12, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=2, num_experts=4,
                        num_experts_per_tok=2, vocab_size=32, max_position_embeddings=16)
    config._attn_implementation = 'eager'
    model = OlmoeForCausalLM(config).eval()
    tokens = torch.tensor([[1, 2, 3], [4, 5, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    hidden = []
    gate = model.model.layers[0].mlp.gate
    handle = gate.register_forward_pre_hook(lambda module, args: hidden.append(args[0].detach()))
    with torch.inference_mode():
        before = model(tokens, attention_mask=mask, use_cache=False).logits
        linear = torch.nn.functional.linear
        router_projection_shapes = []

        def record_linear(input, weight, bias=None):
            if weight is gate.weight:
                router_projection_shapes.append(tuple(input.shape))
            return linear(input, weight, bias)

        with monkeypatch.context() as patch:
            patch.setattr(torch.nn.functional, 'linear', record_linear)
            result = capture_olmoe(model, tokens, mask)
        after = model(tokens, attention_mask=mask, use_cache=False).logits
    handle.remove()
    # Native routing and raw-score capture must project the same full batch.
    assert router_projection_shapes == [(6, 16), (6, 16)]
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    h = hidden[0][[2, 4]]
    expected_scores = h @ gate.weight.T
    torch.testing.assert_close(result['router_scores'][:, 0], expected_scores)
    torch.testing.assert_close(result['selection_scores'][:, 0], expected_scores.softmax(-1))
    torch.testing.assert_close(result['selected_experts'][:, 0], expected_scores.softmax(-1).topk(2).indices)
    packed = model.model.layers[0].mlp.experts.gate_up_proj
    for key, offset in [('w1_l2', 0), ('w3_l2', 12)]:
        expected = torch.stack([(h @ weights[offset:offset + 12].T).norm(dim=-1)
                                for weights in packed], dim=-1)
        torch.testing.assert_close(result[key][:, 0], expected)
    assert all(not layer.mlp.gate._forward_hooks for layer in model.model.layers)


def test_saved_fp16_inputs_use_fp32_projection_of_bf16_gate_and_up_weights():
    config = OlmoeConfig(hidden_size=16, intermediate_size=12, num_hidden_layers=1,
                        num_attention_heads=2, num_key_value_heads=2, num_experts=17,
                        num_experts_per_tok=2, vocab_size=32)
    experts = OlmoeForCausalLM(config).bfloat16().model.layers[0].mlp.experts
    hidden = torch.randn(3, 16).half()
    result = expert_response_norms(experts, hidden)
    # Match the retained experiment's batched einsum projection independently.
    packed = experts.gate_up_proj
    for key, weights in [('w1_l2', packed[:, :12]), ('w3_l2', packed[:, 12:])]:
        expected = torch.cat([
            torch.einsum('ehd,bd->beh', weights[start:start + 8].float(), hidden.float()).norm(dim=-1)
            for start in range(0, 17, 8)
        ], dim=-1)
        assert result[key].dtype == torch.float32
        torch.testing.assert_close(result[key], expected, rtol=0, atol=0)
