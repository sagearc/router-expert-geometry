import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
from transformers import OlmoeConfig, OlmoeForCausalLM

from src.constants import CAUSAL_INTERVENTIONS
from src.hf.causal import edit_inputs, intervene_olmoe_direction


@pytest.mark.parametrize('setting', CAUSAL_INTERVENTIONS)
def test_public_intervention_preserves_routing_and_matches_manual_response(setting):
    torch.manual_seed(42)
    config = OlmoeConfig(hidden_size=8, intermediate_size=6, num_hidden_layers=1,
                        num_attention_heads=2, num_key_value_heads=2, num_experts=4,
                        num_experts_per_tok=2, vocab_size=32)
    config._attn_implementation = 'eager'
    config._experts_implementation = 'eager'
    model = OlmoeForCausalLM(config).eval()
    moe = model.model.layers[0].mlp
    unit = torch.nn.functional.normalize(moe.gate.weight.detach().float(), dim=-1)
    tokens = torch.tensor([[1, 2, 3]])
    captures = []
    handle = moe.experts.register_forward_pre_hook(lambda module, args: captures.append(args))
    with torch.inference_mode():
        baseline = model(tokens, use_cache=False).logits
        hidden, selected, weights = captures[0]
        with intervene_olmoe_direction(moe.experts, unit, setting):
            actual = moe.experts(hidden, selected, weights)
            model(tokens, use_cache=False)
        # Same target-layer routing, since projection happens after selection.
        torch.testing.assert_close(captures[-1][1], selected, rtol=0, atol=0)
        torch.testing.assert_close(captures[-1][2], weights, rtol=0, atol=0)
        expected = torch.zeros_like(hidden)
        for token in range(len(hidden)):
            for slot, expert in enumerate(selected[token]):
                x, v = hidden[token], unit[expert]
                dot = x @ v
                method, strength = CAUSAL_INTERVENTIONS[setting]
                if x.square().sum() == 0:
                    h = x
                elif method == 'projection':
                    h = x - dot * v
                elif method == 'relative':
                    h = x - torch.where(dot >= 0, 1., -1.) * strength * x.norm() * v
                else:
                    direction = v - dot * x / x.square().sum()
                    direction = direction / direction.norm()
                    angle = torch.tensor(strength)
                    h = angle.cos() * x - angle.sin() * x.norm() * torch.where(dot >= 0, 1., -1.) * direction
                gate, up = (moe.experts.gate_up_proj[expert] @ h).chunk(2)
                expected[token] += weights[token, slot] * (
                    moe.experts.down_proj[expert] @ (torch.nn.functional.silu(gate) * up))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(model(tokens, use_cache=False).logits, baseline, rtol=0, atol=0)
    handle.remove()
    with pytest.raises(RuntimeError, match='abort'), intervene_olmoe_direction(moe.experts, unit, 'projection-1'):
        raise RuntimeError('abort')
    assert 'forward' not in moe.experts.__dict__
    moe.experts.config._experts_implementation = 'grouped_mm'
    with pytest.raises(ValueError, match='both use eager'), intervene_olmoe_direction(moe.experts, unit, 'projection-1'):
        pass


def test_intervention_magnitudes_and_unit_direction_sign_symmetry():
    x = torch.tensor([[3., 4., 0.], [-3., 4., 0.]])
    v = torch.tensor([1., 0., 0.])
    for setting, (method, strength) in CAUSAL_INTERVENTIONS.items():
        torch.testing.assert_close(edit_inputs(torch.zeros_like(x), v, setting), torch.zeros_like(x))
        y = edit_inputs(x, v, setting)
        torch.testing.assert_close(y, edit_inputs(x, -v, setting))
        if method == 'projection':
            torch.testing.assert_close(y[:, 0], torch.zeros(2))
        elif method == 'relative':
            torch.testing.assert_close((y-x).norm(dim=-1) / x.norm(dim=-1), torch.full((2,), strength))
        else:
            torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1))
            torch.testing.assert_close((x*y).sum(-1) / (x.norm(dim=-1)*y.norm(dim=-1)),
                                       torch.full((2,), torch.cos(torch.tensor(strength))))
