import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
from transformers import OlmoeConfig, OlmoeForCausalLM

from src.analysis.causal_summary import paired_summary
from src.hf.causal import orthogonal_candidates, sequence_losses
from src.sequences import evaluation_sequences


@pytest.mark.parametrize('step,revision', [
    (5000, '37c50a5f4d30fba036185f57168ee8df0003f2e0'),
    (250000, '64060e961bc36ccbff5de271e6efa5e6f10acd2b'),
])
def test_cli_resolves_recorded_checkpoint(monkeypatch, tmp_path, step, revision):
    from src.cli import public_causal as cli

    received = []

    def observe(selected_revision, eval_root):
        received.append((selected_revision, eval_root))
        return {}

    monkeypatch.setattr('sys.argv', ['public-causal', '--checkpoint-step', str(step),
                                    '--eval-root', str(tmp_path), '--out', str(tmp_path / 'result.json')])
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(cli, 'run', observe)
    cli.main()
    assert received == [(revision, tmp_path)]


def test_control_power_iteration_recovers_known_orthogonal_energy_axes():
    gate = torch.diag(torch.tensor([10., 3., 2., 1.]))[None]
    router = torch.tensor([[1., 0., 0., 0.]])
    vectors, energies = orthogonal_candidates(
        gate, torch.zeros_like(gate), router, count=2, iterations=30, seed=6198)
    torch.testing.assert_close(energies, torch.tensor([[9., 4.]]))
    torch.testing.assert_close(vectors @ vectors.transpose(1, 2), torch.eye(2)[None], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(vectors[..., 0], torch.zeros(1, 2), atol=1e-6, rtol=0)


def test_causal_sample_split_skips_calibration_and_preserves_raw_and_npy(tmp_path):
    tokens = np.arange(21 * 513, dtype=np.uint16)
    raw, npy = tmp_path / 'raw.npy', tmp_path / 'standard.npy'
    raw.write_bytes(tokens.tobytes())
    np.save(npy, tokens)
    expected = tokens[4 * 513:20 * 513].reshape(16, 513)
    for path in (raw, npy):
        np.testing.assert_array_equal(evaluation_sequences(path), expected)


def test_causal_loss_uses_each_next_token_once_and_paired_bootstrap():
    config = OlmoeConfig(hidden_size=8, intermediate_size=8, num_hidden_layers=1,
                        num_attention_heads=2, num_key_value_heads=2, num_experts=4,
                        num_experts_per_tok=2, vocab_size=32,
                        experts_implementation='eager', attn_implementation='eager')
    model = OlmoeForCausalLM(config).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        expected = model(tokens, labels=tokens, use_cache=False).loss.item()
    actual = sequence_losses(model, tokens.numpy())
    np.testing.assert_allclose(actual, [expected], atol=1e-6, rtol=0)
    summary = paired_summary(np.array([0., 0., 0., 4.]))
    assert summary == {'mean': 1., 'ci95': [0., 3.]}


def test_controls_include_up_matrix_energy_not_only_gate():
    router = torch.tensor([[1., 0., 0., 0.]])
    gate = torch.diag(torch.tensor([10., 2., 0., 1.]))[None]
    up = torch.diag(torch.tensor([20., 0., 5., 0.]))[None]
    vectors, energies = orthogonal_candidates(gate, up, router, count=2, iterations=24, seed=7)
    torch.testing.assert_close(energies, torch.tensor([[25., 4.]]), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(vectors.abs(), torch.tensor([[[0., 0., 1., 0.], [0., 1., 0., 0.]]]), atol=1e-3, rtol=0)


def test_loader_reads_checkpoint_settings_without_architecture_conversion(monkeypatch):
    from src.hf import causal
    config = OlmoeConfig(hidden_size=8, intermediate_size=4, num_hidden_layers=16,
                        num_attention_heads=2, num_key_value_heads=2, num_experts=64,
                        num_experts_per_tok=8, vocab_size=32, rms_norm_eps=7e-4,
                        rope_parameters={'rope_type': 'default', 'rope_theta': 12345})
    model = OlmoeForCausalLM(config)
    calls = []

    def load(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return model

    original_to = OlmoeForCausalLM.to

    def cpu_only(self, device):
        assert device == 'cuda'
        return original_to(self, 'cpu')

    monkeypatch.setattr(OlmoeForCausalLM, 'from_pretrained', load)
    monkeypatch.setattr(OlmoeForCausalLM, 'to', cpu_only)
    actual = causal.load_model('fixed-revision')
    assert calls == [(causal.OLMOE_MODEL_ID, {'revision': 'fixed-revision', 'dtype': torch.float32,
                     'attn_implementation': 'eager', 'experts_implementation': 'eager'})]
    assert actual.config.rms_norm_eps == 7e-4
    assert actual.config.rope_parameters['rope_theta'] == 12345
    assert actual is model and not actual.training


def test_cached_run_covers_every_layer_method_and_replays_full_forward(monkeypatch, tmp_path):
    from src.hf import causal
    torch.manual_seed(42)
    config = OlmoeConfig(hidden_size=8, intermediate_size=8, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=2, num_experts=4,
                        num_experts_per_tok=2, vocab_size=32, rms_norm_eps=7e-4,
                        rope_parameters={'rope_type': 'default', 'rope_theta': 12345},
                        experts_implementation='eager', attn_implementation='eager')
    model = OlmoeForCausalLM(config).eval()
    sequences = np.array([[2, 3, 4, 5], [6, 2, 8, 9], [3, 0, 7, 2]], dtype=np.uint16)
    monkeypatch.setattr(causal, 'CAUSAL_LAYERS', (0, 1))
    monkeypatch.setattr(causal, 'CAUSAL_CANDIDATES', 2)
    monkeypatch.setattr(causal, 'CAUSAL_POWER_ITERATIONS', 4)
    monkeypatch.setattr(causal, 'load_model', lambda revision: model)
    monkeypatch.setattr(causal, 'load_sequences', lambda root: (sequences, ['a', 'b', 'c'], {}))
    baseline = sequence_losses(model, sequences)
    result = causal.run('fixed-revision', tmp_path)
    assert result['complete'] and list(result['layers']) == ['0', '1']
    assert result['backend'] == 'checkpoint_eager_fp32'
    np.testing.assert_array_equal(result['baseline_sequence_ce'], baseline)
    for layer in (0, 1):
        moe = model.model.layers[layer].mlp
        unit = torch.nn.functional.normalize(moe.gate.weight.detach().float(), dim=-1)
        gate, up = moe.experts.gate_up_proj.detach().chunk(2, dim=1)
        controls, _ = orthogonal_candidates(gate, up, unit, count=2, iterations=4, seed=12987 + layer)
        measured = result['layers'][str(layer)]['interventions']
        assert set(measured) == set(causal.CAUSAL_INTERVENTIONS)
        for setting, values in measured.items():
            assert set(values['conditions']) == {'router', 'candidate00', 'candidate01'}
            for name, direction in [('router', unit), ('candidate00', controls[:, 0]), ('candidate01', controls[:, 1])]:
                with causal.intervene_olmoe_direction(moe.experts, direction, setting):
                    full = sequence_losses(model, sequences)
                np.testing.assert_allclose(values['conditions'][name]['sequence_ce'], full, rtol=0, atol=1e-6)
                assert values['conditions'][name]['target_routes_unchanged']
        assert not model.model.layers[layer]._forward_pre_hooks and 'forward' not in moe.experts.__dict__
    np.testing.assert_array_equal(sequence_losses(model, sequences), baseline)
