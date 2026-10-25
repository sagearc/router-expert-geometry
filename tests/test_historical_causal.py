"""Distinct steering contracts, without downloading a checkpoint."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('olmo_core')
from olmo_core.nn.transformer import TransformerConfig
from transformers import OlmoeConfig

from src.constants import VALIDATION_DOMAINS
from src.hf import historical_causal as runner
from src.hf.causal import orthogonal_candidates, paired_summary


def test_factory_architecture_defaults_are_not_overridden(monkeypatch):
    config = OlmoeConfig(vocab_size=32, hidden_size=16, intermediate_size=12,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                        num_experts=64, num_experts_per_tok=8, rms_norm_eps=7e-4,
                        rope_parameters={'rope_type': 'default', 'rope_theta': 12345})
    original = TransformerConfig.olmoe_1B_7B
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(TransformerConfig, 'olmoe_1B_7B', factory)
    actual = runner.model_config(config)
    assert calls == [{'vocab_size': 32, 'dtype': runner.DType.float32}]
    model = actual.build(init_device='meta')
    block = model.blocks['0']
    norms = [block.attention_norm, block.feed_forward_norm, block.attention.q_norm,
             block.attention.k_norm, model.lm_head.norm]
    assert block.attention.rope.theta == 500000
    assert all(norm.eps == 1e-6 for norm in norms)
    assert type(block).__name__ == 'MoEReorderedNormTransformerBlock'
    assert actual.d_model == 2048 and actual.n_layers == 16


@pytest.mark.parametrize('step,revision', [
    (50000, 'a4cc2707a1c90cf72a30ea3c50f679680ee290b5'),
    (1220000, '811a27549703ad99a2f7303da68cc720c242b456'),
])
def test_cli_uses_fixed_checkpoint_without_architecture_modes(monkeypatch, tmp_path, step, revision):
    from src.cli import historical_causal as cli
    received = []
    monkeypatch.setattr('sys.argv', ['historical_causal.py', '--checkpoint-step', str(step),
                                    '--eval-root', str(tmp_path), '--out', str(tmp_path / 'result.json')])
    monkeypatch.setattr(cli, 'run', lambda *args: received.append(args))
    cli.main()
    assert received == [(revision, tmp_path, tmp_path / 'result.json', 'pilot')]


def test_checkpoint_loader_preserves_fp32_weights_and_expert_computation(monkeypatch, tmp_path):
    from olmo_core.nn.moe import mlp as upstream_mlp
    from safetensors.torch import save_file
    from transformers import OlmoeForCausalLM
    from transformers.core_model_loading import revert_weight_conversion

    config = OlmoeConfig(vocab_size=32, hidden_size=16, intermediate_size=8,
                        num_hidden_layers=16, num_attention_heads=2, num_key_value_heads=2,
                        num_experts=64, num_experts_per_tok=8, rms_norm_eps=1e-5,
                        rope_parameters={'rope_type': 'default', 'rope_theta': 10000})
    # Public checkpoints store separate experts, not Transformers 5.1's fused state.
    native = OlmoeForCausalLM(config).to(dtype=torch.bfloat16)
    source = revert_weight_conversion(native, native.state_dict())
    config.save_pretrained(tmp_path)
    keys = sorted(source)
    weight_map = {}
    for shard in range(2):
        filename = f'model-{shard:05d}-of-00002.safetensors'
        # Split individual experts across files to exercise index-based assembly.
        tensors = {key: source[key].contiguous() for key in keys[shard::2]}
        save_file(tensors, tmp_path / filename)
        weight_map.update(dict.fromkeys(tensors, filename))
    metadata = {'total_size': sum(tensor.numel() * tensor.element_size() for tensor in source.values())}
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'metadata': metadata, 'weight_map': weight_map}))

    def bf16_backend(*args, **kwargs):
        raise AssertionError('steering must not use the BF16-only backend')

    original_to = runner.Transformer.to

    def keep_on_cpu(model, device):
        assert device == 'cuda'
        return original_to(model, 'cpu')

    monkeypatch.setattr(runner, 'snapshot_download', lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(runner, 'model_config', lambda config:
                        TransformerConfig.olmoe_1B_7B(vocab_size=32, d_model=16,
                                                     n_layers=16, n_heads=2))
    monkeypatch.setattr(upstream_mlp, 'gmm', bf16_backend)
    monkeypatch.setattr(runner.Transformer, 'to', keep_on_cpu)
    model, _ = runner.load_model('pinned-revision')
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}
    for layer, block in model.blocks.items():
        mlp = block.feed_forward_moe.experts.mlp
        assert mlp._gmm is None
        expected = native.model.layers[int(layer)].mlp.experts
        gate, up = expected.gate_up_proj.float().chunk(2, dim=1)
        for actual, reference in ((mlp.w1, gate), (mlp.w3, up), (mlp.w2, expected.down_proj.float().transpose(1, 2))):
            torch.testing.assert_close(actual, reference.reshape_as(actual), rtol=0, atol=0)
        x = torch.randn(2, 16)
        counts = torch.zeros(64, dtype=torch.long)
        counts[0] = 2
        output = mlp.gmm(x, mlp.w1.view(64, 8, 16), counts, trans_b=True)
        torch.testing.assert_close(output, x @ mlp.w1[:8].T)


def test_projection_strength_norm_restoration_and_hook_cleanup():
    values = torch.tensor([[3., 4., 1.], [2., -1., 5.], [7., 2., -3.]])
    counts = torch.tensor([2, 1])
    unit = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    expected = torch.tensor([[0., 4., 1.], [0., -1., 5.], [7., 0., -3.]])
    assert runner.project(values, counts, unit, 0, False) is values
    torch.testing.assert_close(runner.project(values, counts, unit, 1, False), expected)
    torch.testing.assert_close(runner.project(values, counts, unit, .5, False), (values + expected) / 2)
    restored = runner.project(values, counts, unit, 1, True)
    torch.testing.assert_close(restored.norm(dim=-1), values.norm(dim=-1))
    torch.testing.assert_close((restored * unit[torch.tensor([0, 0, 1])]).sum(-1), torch.zeros(3))
    assert torch.isfinite(runner.project(unit, torch.ones(2, dtype=torch.long), unit, 1, True)).all()

    class Expert(torch.nn.Module):
        def forward(self, x, sizes):
            return x, sizes

    expert = Expert()
    with pytest.raises(RuntimeError, match='failure'), runner.intervention(expert, unit, 1, False):
        torch.testing.assert_close(expert(values, counts)[0], expected)
        raise RuntimeError('failure')
    assert not expert._forward_pre_hooks
    assert expert(values, counts)[0] is values


def test_power_iteration_uses_both_input_matrices_and_excludes_router():
    router = torch.tensor([[1., 0., 0., 0.]])
    gate = torch.diag(torch.tensor([10., 2., 0., 1.]))[None]
    up = torch.diag(torch.tensor([20., 0., 5., 0.]))[None]
    vectors, energies = orthogonal_candidates(gate, up, router, count=2, iterations=24, seed=7)
    torch.testing.assert_close(energies, torch.tensor([[25., 4.]]), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(vectors[..., 0], torch.zeros(1, 2), atol=1e-6, rtol=0)
    torch.testing.assert_close(vectors @ vectors.transpose(1, 2), torch.eye(2)[None], atol=1e-6, rtol=1e-6)
    assert len(runner.conditions('pilot')) == 11
    all_controls = runner.conditions('all_controls')
    assert len(all_controls) == 33 and len({condition[1] for condition in all_controls}) == 33


def test_upstream_conversion_keeps_dropless_layout_and_norms_distinct():
    config = OlmoeConfig(num_hidden_layers=1, num_experts=2)
    gate = torch.arange(24.).reshape(2, 3, 4)
    up, down = gate + 100, gate.transpose(1, 2) + 200
    prefix = 'blocks.0.'
    mlp = prefix + 'feed_forward_moe.experts.mlp.'
    source = {'model.layers.0.input_layernorm.weight': torch.ones(4),
              'model.layers.0.post_attention_layernorm.weight': torch.full((4,), 2.)}
    for expert in range(2):
        for name, values in (('gate', gate), ('up', up), ('down', down)):
            source[f'model.layers.0.mlp.experts.{expert}.{name}_proj.weight'] = values[expert]
    original_mappings = deepcopy(runner.get_converter_from_hf('olmoe').mapping_templates)
    actual = runner.convert_weights(source, config)
    assert runner.get_converter_from_hf('olmoe').mapping_templates == original_mappings
    for name, expected in (('w1', gate), ('w2', down.transpose(1, 2)), ('w3', up)):
        torch.testing.assert_close(actual[mlp + name], expected.reshape(6, 4))
    torch.testing.assert_close(actual[prefix + 'attention_norm.weight'], torch.ones(4))
    torch.testing.assert_close(actual[prefix + 'feed_forward_norm.weight'], torch.full((4,), 2.))


def test_matching_uses_routed_activation_energy_not_just_eigenvalues(monkeypatch):
    # Router eta is 10; candidates have eta 2 and 40.5. Match the first,
    # not the highest-eigenvalue/highest-influence second candidate.
    controls = runner.Controls(
        router=torch.tensor([[1., 0., 0.]]),
        vectors=torch.tensor([[[0., 1., 0.], [0., 0., 1.]]]),
        router_energy=torch.tensor([4.]), candidate_energy=torch.tensor([[1., 9.]]))

    class Expert(torch.nn.Module):
        def forward(self, x, counts):
            return x

    expert = Expert()
    observations = [torch.tensor([[1., 2., 0.]]), torch.tensor([[2., 0., 3.]])]

    def observe(_model, sequence):
        expert(observations[int(sequence)], torch.tensor([1]))
        return 0.

    monkeypatch.setattr(runner, 'loss', observe)
    directions, calibration = controls.calibrate(None, expert, torch.tensor([0, 1]))
    assert calibration['matched_index'] == [0] and calibration['hard_index'] == [1]
    assert calibration['counts'] == [2.]
    assert calibration['router_eta'] == [10.] and calibration['candidate_eta'] == [[2., 40.5]]
    torch.testing.assert_close(directions['matched'], controls.vectors[:, 0])
    assert not expert._forward_pre_hooks


def test_corpus_split_preserves_calibration_order_and_predictions(tmp_path: Path):
    tokens = np.arange(21 * 513, dtype=np.uint16)
    for index, domain in enumerate(VALIDATION_DOMAINS):
        path = tmp_path / domain / 'val' / 'part-0-00000.npy'
        path.parent.mkdir(parents=True)
        if index % 2:
            np.save(path, tokens)
        else:
            path.write_bytes(tokens.tobytes())
    calibration, evaluation, ids = runner.load_sequences(tmp_path)
    assert calibration.shape == (44, 513) and evaluation.shape == (176, 513)
    np.testing.assert_array_equal(calibration[:4], tokens[:4 * 513].reshape(4, 513))
    np.testing.assert_array_equal(evaluation[:16], tokens[4 * 513:20 * 513].reshape(16, 513))
    assert ids[0] == 'c4_en-eval-0000' and ids[-1] == 'wikitext_103-eval-0015'
    assert paired_summary(np.array([0., 0., 0., 4.])) == {'mean': 1., 'ci95': [0., 3.]}


@pytest.fixture
def layer_evaluation_case(monkeypatch):
    """CPU hook exercise with varying routes and known projection losses."""
    class Router(torch.nn.Module):
        weight = torch.tensor([[1., 0., 0.]])

        def forward(self, x):
            return x.sum().sigmoid().reshape(1, 1), torch.zeros(1, 1), torch.ones(1), None

    class Expert(torch.nn.Module):
        def forward(self, x, counts):
            return x

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feed_forward_moe = SimpleNamespace(router=Router(), experts=SimpleNamespace(mlp=Expert()))
            self.feed_forward_norm = SimpleNamespace(eps=1e-6)
            self.attention = SimpleNamespace(rope=SimpleNamespace(theta=500000))

        def forward(self, x):
            self.feed_forward_moe.router(x)
            return self.feed_forward_moe.experts.mlp(x, torch.ones(1))

    block = Block()
    model = SimpleNamespace(blocks={'0': block}, embeddings=SimpleNamespace(num_embeddings=16))
    sequences = torch.tensor([[1, 2, 3], [4, 5, 6], [2, 0, 1]])
    directions = dict(zip(('router', 'candidate00', 'candidate01'), torch.eye(3)[:, None], strict=True))
    directions.update({name: directions['candidate00'] for name in ('matched', 'hard', 'weighted0', 'weighted1', 'weighted2')})
    controls = SimpleNamespace(calibrate=lambda *args: (directions, {'counts': [3]}))
    monkeypatch.setattr(runner.Controls, 'build', lambda *args: controls)
    monkeypatch.setattr(runner, 'CAUSAL_CANDIDATES', 2)

    def loss(_model, sequence, layer=None, hidden=None):
        x = sequence[None].float() if hidden is None else hidden
        return 100. - float((block(x) * torch.tensor([1., 2., 3.])).sum())

    monkeypatch.setattr(runner, 'loss', loss)
    return model, sequences, loss


@pytest.mark.parametrize('profile,failure', [
    ('pilot', None), ('all_controls', None), ('all_controls', 'routes'), ('all_controls', 'loss'),
])
def test_layer_evaluation_ranks_and_rejects_invalid_cache(monkeypatch, layer_evaluation_case, profile, failure):
    model, sequences, loss = layer_evaluation_case
    moe = model.blocks['0'].feed_forward_moe
    routes = []
    with runner.capture_routes(moe.router, routes):
        baseline = np.array([loss(model, row) for row in sequences])
    hidden = [row[None].float() for row in sequences]
    if failure == 'routes':
        routes[0] = 'incorrect route fingerprint'
    elif failure == 'loss':
        monkeypatch.setattr(runner, 'loss', lambda *args: loss(*args) + (0.125 if len(args) == 4 else 0.))
    if failure:
        with pytest.raises(ValueError if failure == 'routes' else AssertionError):
            runner.evaluate_layer(model, 0, sequences, sequences, profile, baseline, hidden, routes)
    else:
        result = runner.evaluate_layer(model, 0, sequences, sequences, profile, baseline, hidden, routes)
        np.testing.assert_array_equal(result['conditions']['router_alpha1']['sequence_ce'], baseline + sequences[:, 0].numpy())
        if profile == 'all_controls':
            assert result['comparisons']['router_rank'] == 3
        else:
            assert set(result['comparisons']) == {'0.5', '1', '1_norm'}
    assert not moe.router._forward_hooks and not moe.experts.mlp._forward_pre_hooks


@pytest.mark.skipif(not torch.cuda.is_available(), reason='native MoE dispatch requires CUDA')
def test_suffix_matches_full_forward_and_preserves_routes_on_tiny_model():
    config = TransformerConfig.olmoe_1B_7B(vocab_size=32, d_model=16, n_layers=2, n_heads=2)
    model = config.build(init_device='cuda').eval()
    model.init_weights()
    sequence = torch.tensor([1, 4, 2, 8, 3])
    block = model.blocks['1']
    captured = []
    handle = block.register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0].clone()))
    with torch.inference_mode():
        baseline = runner.loss(model, sequence)
        handle.remove()
        assert runner.loss(model, sequence, 1, captured[0]) == baseline
        moe, mlp = block.feed_forward_moe, block.feed_forward_moe.experts.mlp
        unit = torch.nn.functional.normalize(moe.router.weight.view(64, 16).float(), dim=-1)
        routes, modified_routes = [], []
        with runner.capture_routes(moe.router, routes):
            runner.loss(model, sequence)
        with runner.intervention(mlp, unit, 1, True), runner.capture_routes(moe.router, modified_routes):
            cached = runner.loss(model, sequence, 1, captured[0])
            full = runner.loss(model, sequence)
        assert cached == full
        assert routes * 2 == modified_routes


@pytest.mark.parametrize('profile,layers', [('pilot', [15]), ('all_controls', list(range(16)))])
def test_run_evaluates_every_requested_layer_with_its_cached_inputs(
    monkeypatch, tmp_path, layer_evaluation_case, profile, layers,
):
    model, sequences, _ = layer_evaluation_case
    model.blocks = {str(layer): deepcopy(model.blocks['0']) for layer in range(16)}
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(runner, 'load_model', lambda revision: (model, revision))
    monkeypatch.setattr(runner, 'load_sequences', lambda root: (sequences, sequences, ['a', 'b', 'c']))

    def forward(_model, sequence):
        for block in model.blocks.values():
            block(sequence[None].float())
        return float(sequence.sum())

    visited = []

    def evaluate(_model, layer, calibration, evaluation, condition_set, baseline, hidden, routes):
        visited.append(layer)
        assert condition_set == profile and len(routes) == len(sequences)
        torch.testing.assert_close(torch.cat(hidden), sequences.float())
        np.testing.assert_array_equal(baseline, sequences.sum(-1).numpy())
        return {'comparisons': {'router_rank': 1}}

    monkeypatch.setattr(runner, 'loss', forward)
    monkeypatch.setattr(runner, 'evaluate_layer', evaluate)
    out = tmp_path / 'result.json'
    runner.run('pinned-revision', tmp_path, out, profile)
    result = json.loads(out.read_text())
    assert visited == layers and list(result['layers']) == [str(layer) for layer in layers]
    assert result['complete']
