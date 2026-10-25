"""Router-axis steering with calibrated orthogonal controls and norm restoration.

Uses the upstream OLMo-core factory without architecture overrides.
"""

import hashlib
import json
import math
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from huggingface_hub import snapshot_download
from olmo_core.config import DType
from olmo_core.nn.conversion.state_mapping import TemplatePlaceholder
from olmo_core.nn.hf.convert import get_converter_from_hf
from olmo_core.nn.moe.mlp import DroplessMoEMLP
from olmo_core.nn.moe.router import MoERouter
from olmo_core.nn.transformer import Transformer, TransformerConfig
from olmo_core.nn.transformer.block import MoETransformerBlock
from safetensors.torch import load_file
from transformers import AutoConfig, OlmoeConfig

from src.constants import (
    CAUSAL_CALIBRATION_SEQUENCES,
    CAUSAL_CANDIDATE_SEED,
    CAUSAL_CANDIDATES,
    CAUSAL_POWER_ITERATIONS,
    CAUSAL_PREDICTIONS,
    CAUSAL_SEQUENCES_PER_DOMAIN,
    OLMOE_LAYERS,
    OLMOE_MODEL_ID,
    VALIDATION_DOMAINS,
)
from src.hf.causal import expert_energy, orthogonal_candidates, paired_summary
from src.sequences import causal_rows
from src.types import Float64Array, RouterOutput

type ConditionSet = Literal['pilot', 'all_controls']


def convert_weights(
    source: dict[str, torch.Tensor], config: OlmoeConfig,
) -> dict[str, torch.Tensor]:
    """Use upstream conversion with OLMoE norm naming and dropless input layouts."""
    layer, expert = TemplatePlaceholder.LAYER, TemplatePlaceholder.EXPERT
    converter = get_converter_from_hf(model_type='olmoe')
    for i, mapping in enumerate(converter.mapping_templates):
        if mapping.source_template_keys == f'model.layers.{layer}.post_attention_layernorm.weight':
            converter.mapping_templates[i] = replace(
                mapping, dest_template_keys=f'blocks.{layer}.feed_forward_norm.weight',
            )
        elif mapping.dest_template_keys in {
            f'blocks.{layer}.feed_forward_moe.experts.mlp.w1',
            f'blocks.{layer}.feed_forward_moe.experts.mlp.w3',
        }:
            # Dropless gate/up are [experts * hidden, model], without a transpose.
            converter.mapping_templates[i] = replace(mapping, source_concat_dim=0, dims_permutation=None)
    return converter.convert(source, {layer: config.num_hidden_layers, expert: config.num_experts})


def model_config(config: OlmoeConfig) -> TransformerConfig:
    if config.vocab_size is None:
        raise ValueError('checkpoint is missing its vocabulary size')
    return TransformerConfig.olmoe_1B_7B(
        vocab_size=config.vocab_size, dtype=DType.float32,
    )


def load_model(revision: str) -> tuple[Transformer, str]:
    snapshot = Path(snapshot_download(OLMOE_MODEL_ID, revision=revision,
                                     allow_patterns=['config.json', 'model.safetensors.index.json', 'model-*.safetensors']))
    config = AutoConfig.from_pretrained(snapshot)
    if not isinstance(config, OlmoeConfig) or (
        config.num_hidden_layers, config.num_experts, config.num_experts_per_tok
    ) != (16, 64, 8):
        raise ValueError('requires public OLMoE with 16 layers, 64 experts and top-8 routing')
    model = model_config(config).build(init_device='meta')
    index = json.loads((snapshot / 'model.safetensors.index.json').read_text())
    source = {}
    for shard in sorted(set(index['weight_map'].values())):
        source.update({key: tensor.float() for key, tensor in load_file(snapshot / shard).items()})
    model.load_state_dict(convert_weights(source, config), assign=True)
    del source
    for block in model.blocks.values():
        mlp = cast(MoETransformerBlock, block).feed_forward_moe.experts.mlp
        # The optional grouped-GEMM backend unconditionally casts to BF16.
        cast(DroplessMoEMLP, mlp)._gmm = None
    return model.to('cuda').eval(), snapshot.name


def expert_ids(x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    return torch.repeat_interleave(torch.arange(len(counts), device=x.device),
                                   counts.long(), output_size=len(x))


def project(
    x: torch.Tensor, counts: torch.Tensor, directions: torch.Tensor,
    alpha: float, restore_norm: bool,
) -> torch.Tensor:
    if not 0 <= alpha <= 1 or directions.shape != (len(counts), x.shape[-1]):
        raise ValueError('invalid projection strength or expert directions')
    if alpha == 0:
        return x
    unit = directions[expert_ids(x, counts)].float()
    values = x.float()
    result = values - alpha * (values * unit).sum(-1, keepdim=True) * unit
    if restore_norm:
        norm = result.norm(dim=-1, keepdim=True)
        eps = torch.finfo(result.dtype).eps
        result *= torch.where(norm > eps, values.norm(dim=-1, keepdim=True) / norm.clamp_min(eps), 1.0)
    return result.to(x.dtype)


@contextmanager
def intervention(
    mlp: DroplessMoEMLP, directions: torch.Tensor, alpha: float, restore_norm: bool,
) -> Generator[None, None, None]:
    def hook(_module: DroplessMoEMLP, args: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x, counts = args
        return project(x, counts, directions, alpha, restore_norm), counts

    with mlp.register_forward_pre_hook(hook):
        yield


@dataclass
class Controls:
    router: torch.Tensor
    vectors: torch.Tensor
    router_energy: torch.Tensor
    candidate_energy: torch.Tensor

    @classmethod
    def build(cls, mlp: DroplessMoEMLP, router_weight: torch.Tensor, layer: int) -> 'Controls':
        router = router_weight.detach().view(mlp.num_experts, mlp.d_model).float()
        norm = router.norm(dim=-1, keepdim=True)
        if (norm <= torch.finfo(router.dtype).tiny).any():
            raise ValueError('zero router row')
        router = router / norm
        gate = mlp.w1.detach().view(mlp.num_experts, mlp.hidden_size, mlp.d_model)
        up = mlp.w3.detach().view_as(gate)
        vectors, energies = orthogonal_candidates(
            gate, up, router, count=CAUSAL_CANDIDATES, iterations=CAUSAL_POWER_ITERATIONS,
            seed=CAUSAL_CANDIDATE_SEED + layer)
        unit = router[:, None]
        router_energy = (unit * expert_energy(unit, gate, up)).sum(-1).squeeze(1)
        return cls(router, vectors, router_energy, energies)

    def calibrate(
        self, model: Transformer, mlp: DroplessMoEMLP, sequences: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        router_sum = torch.zeros_like(self.router_energy)
        candidate_sum = torch.zeros_like(self.candidate_energy)
        counts_sum = torch.zeros_like(self.router_energy)

        def moments(_module: DroplessMoEMLP, args: tuple[torch.Tensor, torch.Tensor]) -> None:
            x, counts = args
            ids = expert_ids(x, counts)
            x = x.float()
            router_sum.index_add_(0, ids, (x * self.router[ids]).sum(-1).square())
            dots = torch.einsum('nd,nkd->nk', x, self.vectors[ids])
            candidate_sum.index_add_(0, ids, dots.square())
            counts_sum.add_(counts)

        with mlp.register_forward_pre_hook(moments):
            for sequence in sequences:
                loss(model, sequence)
        # Preserve historical operation order for exact matched-control selection.
        candidate_eta = self.candidate_energy * (candidate_sum / counts_sum.clamp_min(1)[:, None])
        router_eta = self.router_energy * (router_sum / counts_sum.clamp_min(1))
        matched = (candidate_eta - router_eta[:, None]).abs().argmin(-1)
        hard = candidate_eta.argmax(-1)
        rows = torch.arange(len(self.router), device=self.router.device)
        directions = {'router': self.router, 'matched': self.vectors[rows, matched],
                      'hard': self.vectors[rows, hard]}
        probabilities = candidate_eta.clamp_min(0)
        probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(torch.finfo(probabilities.dtype).eps)
        probabilities[candidate_eta.sum(-1) <= 0] = 1 / probabilities.shape[-1]
        for index, seed in enumerate((1701, 1702, 1703)):
            generator = torch.Generator(device='cpu').manual_seed(seed)
            chosen = torch.multinomial(probabilities.cpu(), 1, replacement=True, generator=generator).squeeze(-1)
            directions[f'weighted{index}'] = self.vectors[rows, chosen.to(rows.device)]
        directions.update({f'candidate{i:02d}': self.vectors[:, i] for i in range(self.vectors.shape[1])})
        calibration: dict[str, object] = {'counts': counts_sum.tolist(), 'matched_index': matched.tolist(),
                                         'hard_index': hard.tolist(), 'router_eta': router_eta.tolist(),
                                         'candidate_eta': candidate_eta.tolist()}
        return directions, calibration


def conditions(profile: ConditionSet) -> list[tuple[str, str, float, bool]]:
    if profile == 'all_controls':
        return [(f'{name}_alpha1', name, 1.0, False)
                for name in ['router', *[f'candidate{i:02d}' for i in range(CAUSAL_CANDIDATES)]]]
    return [('identity_alpha0', 'router', 0.0, False),
            ('router_alpha0.5', 'router', .5, False), ('router_alpha1', 'router', 1.0, False),
            ('matched_alpha0.5', 'matched', .5, False), ('matched_alpha1', 'matched', 1.0, False),
            ('router_alpha1_norm', 'router', 1.0, True), ('matched_alpha1_norm', 'matched', 1.0, True),
            *[(f'{name}_alpha1', name, 1.0, False) for name in ('hard', 'weighted0', 'weighted1', 'weighted2')]]


def load_sequences(root: Path) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    calibration: list[torch.Tensor] = []
    evaluation: list[torch.Tensor] = []
    ids: list[str] = []
    for domain in VALIDATION_DOMAINS:
        rows = torch.from_numpy(causal_rows(root / domain / 'val' / 'part-0-00000.npy').astype(np.int64))
        calibration.extend(rows[:CAUSAL_CALIBRATION_SEQUENCES])
        evaluation.extend(rows[CAUSAL_CALIBRATION_SEQUENCES:])
        ids.extend(f'{domain}-eval-{i:04d}' for i in range(CAUSAL_SEQUENCES_PER_DOMAIN))
    return torch.stack(calibration), torch.stack(evaluation), ids


def loss(model: Transformer, sequence: torch.Tensor, layer: int | None = None, hidden: torch.Tensor | None = None) -> float:
    tokens = sequence.to(next(model.parameters()).device)
    inputs, labels = tokens[None, :-1], tokens[None, 1:]
    if layer is None:
        output = model(input_ids=inputs, labels=labels, loss_reduction='none', return_logits=False)
    else:
        _, prepared_labels, shared, per_block, head = model._prepare_inputs(inputs, labels, loss_reduction='none', return_logits=False)
        assert hidden is not None
        h = hidden.to(tokens.device)
        for key, block in model.blocks.items():
            if int(key) >= layer:
                h = block(h, **shared, **per_block.get(int(key), {}))
        output = model.lm_head(h, labels=prepared_labels, **head)
    value = float(output.ce_loss.mean().item())
    if not math.isfinite(value):
        raise ValueError('nonfinite sequence loss')
    return value


@contextmanager
def capture_routes(router: MoERouter, records: list[str]) -> Generator[None, None, None]:
    def hook(_module: MoERouter, _args: tuple[torch.Tensor, ...], output: RouterOutput) -> None:
        weights, indices, counts, _aux = output
        digest = hashlib.sha256()
        for value in (indices.int(), counts.long(), weights.float()):
            digest.update(value.detach().contiguous().cpu().numpy().tobytes())
        records.append(digest.hexdigest())

    with router.register_forward_hook(hook):
        yield


def evaluate_layer(
    model: Transformer, layer: int, calibration: torch.Tensor, sequences: torch.Tensor,
    condition_set: ConditionSet, baseline: Float64Array, hidden: list[torch.Tensor], routes: list[str],
) -> dict[str, object]:
    """Steer one layer; reject changed routing or cached/full-forward disagreement."""
    moe = cast(MoETransformerBlock, model.blocks[str(layer)]).feed_forward_moe
    mlp = cast(DroplessMoEMLP, moe.experts.mlp)
    controls = Controls.build(mlp, cast(torch.Tensor, moe.router.weight), layer)
    directions, diagnostics = controls.calibrate(model, mlp, calibration)
    losses: dict[str, Float64Array] = {}
    records = {}
    parity_indices = sorted({0, len(sequences) // 2, len(sequences) - 1})

    for index, (name, direction, alpha, restore) in enumerate(conditions(condition_set)):
        observed: list[str] = []
        with intervention(mlp, directions[direction], alpha, restore), capture_routes(moe.router, observed):
            values = np.array([loss(model, row, layer, hidden[i]) for i, row in enumerate(sequences)])
            if observed != routes:
                raise ValueError(f'intervention changed target routing: layer {layer}, {name}')
            if alpha == 0:
                np.testing.assert_array_equal(values, baseline)
            # Repeat the first two conditions without prefix caching.
            if index < 2:
                observed.clear()
                reference = [loss(model, sequences[i]) for i in parity_indices]
                np.testing.assert_array_equal(values[parity_indices], reference)
                if observed != [routes[i] for i in parity_indices]:
                    raise ValueError('cached/full target routing differs')
        losses[name] = values
        records[name] = {'sequence_ce': values.tolist(), 'delta_ce': paired_summary(values - baseline),
                         'target_routes_unchanged': True}
        print(f'layer={layer} {name} CE={values.mean():.9f}', flush=True)

    comparisons: dict[str, object] = {}
    if condition_set == 'all_controls':
        router_loss = float(losses['router_alpha1'].mean())
        comparisons['router_rank'] = 1 + sum(float(values.mean()) > router_loss
                                             for name, values in losses.items() if name != 'router_alpha1')
    else:
        for suffix in ('0.5', '1', '1_norm'):
            delta = losses[f'router_alpha{suffix}'] - losses[f'matched_alpha{suffix}']
            comparisons[suffix] = {**paired_summary(delta), 'extra_ppl_percent': 100 * math.expm1(float(delta.mean()))}
    return {'calibration': diagnostics, 'conditions': records, 'comparisons': comparisons,
            'cached_full_parity': True}


@torch.inference_mode()
def run(revision: str, eval_root: Path, out: Path, condition_set: ConditionSet) -> None:
    if not torch.cuda.is_available() or out.exists():
        raise ValueError('requires CUDA and a new output file')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    layers = [OLMOE_LAYERS - 1] if condition_set == 'pilot' else list(range(OLMOE_LAYERS))
    calibration, sequences, sample_ids = load_sequences(eval_root)
    model, revision = load_model(revision)
    vocab = model.embeddings.num_embeddings
    if min(calibration.min(), sequences.min()) < 0 or max(calibration.max(), sequences.max()) >= vocab:
        raise ValueError('token outside model vocabulary')
    first_block = cast(MoETransformerBlock, model.blocks['0'])
    layer_results: dict[str, object] = {}
    result: dict[str, object] = {
        'model': OLMOE_MODEL_ID, 'revision': revision,
        'architecture': 'OLMo-core olmoe_1B_7B factory',
        'block_type': type(first_block).__name__,
        'rms_norm_eps': first_block.feed_forward_norm.eps,
        'rope_theta': first_block.attention.rope.theta,
        'torch': torch.__version__, 'condition_set': condition_set,
        'sequence_length': CAUSAL_PREDICTIONS, 'sample_ids': sample_ids,
        'sequence_sha256': [hashlib.sha256(row.numpy().astype('<i8').tobytes()).hexdigest() for row in sequences],
        'candidate_count': CAUSAL_CANDIDATES, 'power_iterations': CAUSAL_POWER_ITERATIONS,
        'seed': CAUSAL_CANDIDATE_SEED, 'layers': layer_results,
    }
    # Cache unchanged prefixes, not autoregressive KV states. No generation.
    cache: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    routes: dict[int, list[str]] = {layer: [] for layer in layers}
    with ExitStack() as stack:
        for layer in layers:
            block = cast(MoETransformerBlock, model.blocks[str(layer)])

            def capture(_module: MoETransformerBlock, inputs: tuple[torch.Tensor, ...], layer: int = layer) -> None:
                cache[layer].append(inputs[0].detach().cpu().clone())

            stack.enter_context(block.register_forward_pre_hook(capture))
            stack.enter_context(capture_routes(block.feed_forward_moe.router, routes[layer]))
        baseline = np.array([loss(model, sequence) for sequence in sequences])
    result['baseline_sequence_ce'] = baseline.tolist()
    out.parent.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        layer_results[str(layer)] = evaluate_layer(
            model, layer, calibration, sequences, condition_set, baseline, cache[layer], routes[layer])
        result['complete'] = layer == layers[-1]
        out.write_text(json.dumps(result, indent=2) + '\n')
