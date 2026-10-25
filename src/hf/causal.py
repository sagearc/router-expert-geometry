"""Public OLMoE router-axis removal versus all 32 orthogonal controls.

Checkpoint settings through Transformers, without architecture conversion.
Four fixed interventions at every layer; 512 predictions, 16 sequences/domain
after skipping four original calibration rows; weight-derived controls.
"""

import hashlib
import math
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import OlmoeConfig, OlmoeForCausalLM
from transformers.models.olmoe.modeling_olmoe import OlmoeDecoderLayer, OlmoeExperts

from src.artifacts import CausalCondition, CausalIntervention, PublicCausalResult
from src.constants import (
    CAUSAL_BOOTSTRAP_DRAWS,
    CAUSAL_BOOTSTRAP_SEED,
    CAUSAL_CALIBRATION_SEQUENCES,
    CAUSAL_CANDIDATE_SEED,
    CAUSAL_CANDIDATES,
    CAUSAL_INTERVENTIONS,
    CAUSAL_LAYERS,
    CAUSAL_POWER_ITERATIONS,
    CAUSAL_PREDICTIONS,
    CAUSAL_SEQUENCES_PER_DOMAIN,
    OLMOE_EXPERTS,
    OLMOE_LAYERS,
    OLMOE_MODEL_ID,
    OLMOE_TOP_K,
    VALIDATION_DOMAINS,
)
from src.sequences import evaluation_sequences
from src.types import Float64Array, IntArray


def expert_energy(vectors: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Apply W_gate.T W_gate + W_up.T W_up without a d_model-square matrix."""
    gate, up = gate.float(), up.float()
    return (torch.bmm(torch.bmm(vectors, gate.transpose(1, 2)), gate)
            + torch.bmm(torch.bmm(vectors, up.transpose(1, 2)), up))


def orthonormalize(vectors: torch.Tensor, router: torch.Tensor) -> torch.Tensor:
    vectors = vectors - torch.einsum('ekd,ed->ek', vectors, router)[..., None] * router[:, None]
    basis: list[torch.Tensor] = []
    for vector in vectors.unbind(dim=1):
        for previous in basis:
            vector = vector - (vector * previous).sum(-1, keepdim=True) * previous
        norm = vector.norm(dim=-1, keepdim=True)
        if (norm <= torch.finfo(vector.dtype).eps).any():
            raise ValueError('degenerate orthogonal control direction')
        basis.append(vector / norm)
    return torch.stack(basis, dim=1)


@torch.no_grad()
def orthogonal_candidates(
    gate: torch.Tensor,
    up: torch.Tensor,
    router: torch.Tensor,
    *,
    count: int,
    iterations: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block power iteration in each unit router row's orthogonal complement."""
    if gate.ndim != 3 or gate.shape != up.shape or router.shape != (gate.shape[0], gate.shape[2]):
        raise ValueError('expected expert input matrices and one router row per expert')
    if not 1 <= count < router.shape[-1] or iterations < 0:
        raise ValueError('invalid candidate count or iteration count')
    if not torch.isfinite(router).all() or not torch.allclose(
        router.norm(dim=-1), torch.ones(len(router), device=router.device), atol=1e-5, rtol=1e-5
    ):
        raise ValueError('router rows must be finite unit vectors')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    vectors = torch.randn(len(router), count, router.shape[-1], generator=generator,
                          dtype=torch.float32).to(router.device)
    vectors = orthonormalize(vectors, router)
    for _ in range(iterations):
        vectors = orthonormalize(expert_energy(vectors, gate, up), router)
    energies = (vectors * expert_energy(vectors, gate, up)).sum(-1).clamp_min(0)
    order = energies.argsort(dim=-1, descending=True)
    vectors = vectors.gather(1, order[..., None].expand_as(vectors))
    if (vectors * router[:, None]).sum(-1).abs().max() > 2e-4:
        raise ValueError('orthogonal-control residual exceeds retained tolerance')
    return vectors, energies.gather(1, order)


def edit_inputs(x: torch.Tensor, v: torch.Tensor, setting: str) -> torch.Tensor:
    """The four declared edits, applied identically to router and controls."""
    method, strength = CAUSAL_INTERVENTIONS[setting]
    dot = (x * v).sum(-1, keepdim=True)
    if method == 'projection':
        return x - strength * dot * v
    norm: torch.Tensor = x.norm(dim=-1, keepdim=True)
    sign: torch.Tensor = torch.where(dot >= 0, 1., -1.)
    if method == 'relative':
        return x - strength * norm * sign * v
    unit = x / norm.clamp_min(1e-20)
    tangent = v - (unit * v).sum(-1, keepdim=True) * unit
    tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-20)
    rotated: torch.Tensor = norm * (math.cos(strength) * unit - math.sin(strength) * sign * tangent)
    return rotated


@contextmanager
def intervene_olmoe_direction(
    experts: OlmoeExperts, directions: torch.Tensor, setting: str,
) -> Generator[None, None, None]:
    """Keep checkpoint routing and decoder normalization; change expert inputs only.

    Uses the pinned Transformers eager expert loop, with one projection before
    gate/up multiplication. No checkpoint conversion or decoder replacement.
    Intended for single-device inference, not concurrent calls or training.
    """
    if not isinstance(experts, OlmoeExperts) or experts.training:
        raise ValueError('requires eval-mode OlmoeExperts')
    if setting not in CAUSAL_INTERVENTIONS:
        raise ValueError('unknown intervention')
    if cast(OlmoeConfig, experts.config)._experts_implementation != 'eager':
        raise ValueError('baseline and intervention must both use eager experts')
    if directions.shape != (experts.num_experts, experts.hidden_dim):
        raise ValueError('invalid direction shape')
    unit = directions.to(device=experts.gate_up_proj.device, dtype=torch.float32)
    if not torch.isfinite(unit).all() or not torch.allclose(
        unit.norm(dim=-1), torch.ones(len(unit), device=unit.device), atol=1e-5, rtol=1e-5
    ):
        raise ValueError('directions must be finite unit vectors')
    if 'forward' in experts.__dict__:
        raise ValueError('expert forward is already overridden')

    def forward(
        self: OlmoeExperts, hidden_states: torch.Tensor,
        top_k_index: torch.Tensor, top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hits = (mask.sum(dim=(-1, -2)) > 0).nonzero()
        for expert in hits:
            expert = expert[0]
            positions, tokens = torch.where(mask[expert])
            state = hidden_states[tokens]
            values = state.float()
            direction = unit[expert]
            state = edit_inputs(values, direction, setting).to(state.dtype)
            gate, up = F.linear(state, self.gate_up_proj[expert]).chunk(2, dim=-1)
            output = F.linear(self.act_fn(gate) * up, self.down_proj[expert])
            output = output * top_k_weights[tokens, positions, None]
            result.index_add_(0, tokens, output.to(result.dtype))
        return result

    experts.forward = MethodType(forward, experts)  # type: ignore[method-assign]  # Scoped replacement, restored below.
    try:
        yield
    finally:
        del experts.forward


@torch.inference_mode()
def sequence_losses(model: OlmoeForCausalLM, sequences: IntArray) -> Float64Array:
    device = next(model.parameters()).device
    values = []
    for sequence in sequences:
        tokens = torch.as_tensor(sequence, device=device, dtype=torch.long)
        logits = model(tokens[None, :-1], use_cache=False).logits
        loss = F.cross_entropy(logits[0].float(), tokens[1:])
        if not torch.isfinite(loss):
            raise ValueError('nonfinite sequence loss')
        values.append(float(loss))
    return np.asarray(values)


@dataclass(frozen=True)
class Prefix:
    """A decoder input for fixed-token suffix replay, not a generation KV cache."""
    hidden: torch.Tensor
    mask: torch.Tensor | None
    positions: torch.Tensor
    cache_position: torch.Tensor
    embeddings: tuple[torch.Tensor, torch.Tensor]

    @classmethod
    def capture(cls, hidden: torch.Tensor, kwargs: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor] | bool | None]) -> 'Prefix':
        allowed = {'attention_mask', 'position_ids', 'cache_position', 'position_embeddings',
                   'past_key_values', 'use_cache', 'output_router_logits'}
        if kwargs.keys() - allowed or kwargs.get('past_key_values') is not None or kwargs.get('use_cache') is not False:
            raise ValueError('suffix replay requires fixed-token inference without a KV cache')
        if kwargs.get('output_router_logits', False):
            raise ValueError('router-logit output is not part of suffix replay')
        mask, positions = kwargs.get('attention_mask'), kwargs.get('position_ids')
        cache_position, embeddings = kwargs.get('cache_position'), kwargs.get('position_embeddings')
        if (mask is not None and not isinstance(mask, torch.Tensor)) or not isinstance(positions, torch.Tensor):
            raise ValueError('unexpected decoder mask or position IDs')
        if positions.dtype != torch.long or not isinstance(cache_position, torch.Tensor):
            raise ValueError('unexpected decoder position tensors')
        if not isinstance(embeddings, tuple) or not all(isinstance(t, torch.Tensor) for t in embeddings):
            raise ValueError('expected rotary cosine/sine tensors')
        return cls(hidden.detach().cpu().clone(), None if mask is None else mask.detach().cpu(),
                   positions.detach().cpu(), cache_position.detach().cpu(),
                   (embeddings[0].detach().cpu(), embeddings[1].detach().cpu()))

    def logits(self, model: OlmoeForCausalLM, layer: int) -> torch.Tensor:
        device = next(model.parameters()).device
        h = self.hidden.to(device)
        for decoder in model.model.layers[layer:]:
            h = decoder(h, attention_mask=None if self.mask is None else self.mask.to(device),
                        # Captured IDs are checked above before this Transformers typing boundary.
                        position_ids=cast(torch.LongTensor, self.positions.to(device)),
                        cache_position=self.cache_position.to(device),
                        position_embeddings=(self.embeddings[0].to(device), self.embeddings[1].to(device)),
                        use_cache=False)
        logits: torch.Tensor = model.lm_head(model.model.norm(h))
        return logits[0]


@torch.inference_mode()
def suffix_losses(model: OlmoeForCausalLM, sequences: IntArray, layer: int, prefixes: list[Prefix]) -> Float64Array:
    device = next(model.parameters()).device
    losses = []
    for sequence, prefix in zip(sequences, prefixes, strict=True):
        targets = torch.as_tensor(sequence[1:], device=device, dtype=torch.long)
        loss = F.cross_entropy(prefix.logits(model, layer).float(), targets)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite suffix loss')
        losses.append(float(loss))
    return np.asarray(losses)


@contextmanager
def capture_routes(experts: OlmoeExperts, records: list[str]) -> Generator[None, None, None]:
    def hook(_module: OlmoeExperts, args: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        digest = hashlib.sha256()
        for tensor in args[1:]:
            digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
        records.append(digest.hexdigest())
    with experts.register_forward_pre_hook(hook):
        yield


def load_sequences(eval_root: Path) -> tuple[IntArray, list[str], dict[str, str]]:
    """All-domain evaluation rows, their sample IDs and each shard's SHA-256."""
    sequence_rows: list[IntArray] = []
    sample_ids: list[str] = []
    sources: dict[str, str] = {}
    for domain in VALIDATION_DOMAINS:
        path = eval_root / domain / 'val' / 'part-0-00000.npy'
        sequence_rows.extend(evaluation_sequences(path))
        sample_ids.extend(f'{domain}-eval-{i:04d}' for i in range(CAUSAL_SEQUENCES_PER_DOMAIN))
        with path.open('rb') as stream:
            sources[domain] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return np.stack(sequence_rows), sample_ids, sources


def load_model(revision: str) -> OlmoeForCausalLM:
    model = OlmoeForCausalLM.from_pretrained(
        OLMOE_MODEL_ID, revision=revision, dtype=torch.float32,
        attn_implementation='eager', experts_implementation='eager',
    ).to('cuda').eval()  # type: ignore[arg-type]  # Transformers 5.1 @wraps loses the bound .to() signature.
    config = model.config
    if (config.num_hidden_layers, config.num_experts, config.num_experts_per_tok) != (OLMOE_LAYERS, OLMOE_EXPERTS, OLMOE_TOP_K):
        raise ValueError('expected public OLMoE 16-layer / 64-expert / top-8 architecture')
    return cast(OlmoeForCausalLM, model)


@torch.inference_mode()
def run(revision: str, eval_root: Path) -> PublicCausalResult:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sequences, sample_ids, sources = load_sequences(eval_root)
    model = load_model(revision)
    vocab_size = model.config.vocab_size
    if vocab_size is None or (sequences < 0).any() or (sequences >= vocab_size).any():
        raise ValueError('token ID outside vocabulary')
    cache: dict[int, list[Prefix]] = {layer: [] for layer in CAUSAL_LAYERS}
    routes: dict[int, list[str]] = {layer: [] for layer in CAUSAL_LAYERS}
    with ExitStack() as stack:
        for layer in CAUSAL_LAYERS:
            def capture(_module: OlmoeDecoderLayer, args: tuple[torch.Tensor, ...],
                        kwargs: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor] | bool | None],
                        layer: int = layer) -> None:
                cache[layer].append(Prefix.capture(args[0], kwargs))
            stack.enter_context(model.model.layers[layer].register_forward_pre_hook(capture, with_kwargs=True))
            stack.enter_context(capture_routes(model.model.layers[layer].mlp.experts, routes[layer]))
        baseline = sequence_losses(model, sequences)
    eps = model.config.rms_norm_eps
    if eps is None:
        raise ValueError('checkpoint is missing its RMSNorm epsilon')
    result: PublicCausalResult = {'revision': revision, 'transformers': transformers.__version__,
              'torch': torch.__version__, 'backend': 'checkpoint_eager_fp32', 'complete': False,
              'model_settings': {'decoder_class': type(model.model.layers[0]).__name__,
                                 'rope_theta': float(model.config.rope_parameters['rope_theta']),
                                 'rms_norm_eps': float(eps)},
              'protocol': {'candidate_count': CAUSAL_CANDIDATES, 'power_iterations': CAUSAL_POWER_ITERATIONS,
                           'candidate_seed': CAUSAL_CANDIDATE_SEED, 'bootstrap_seed': CAUSAL_BOOTSTRAP_SEED,
                           'bootstrap_draws': CAUSAL_BOOTSTRAP_DRAWS, 'predictions_per_sequence': CAUSAL_PREDICTIONS,
                           'calibration_sequences_skipped_per_domain': CAUSAL_CALIBRATION_SEQUENCES,
                           'evaluation_sequences_per_domain': CAUSAL_SEQUENCES_PER_DOMAIN,
                           'interventions': list(CAUSAL_INTERVENTIONS)},
              'sources_sha256': sources, 'sample_ids': sample_ids,
              # Canonical little-endian int64 bytes, including input and target
              # tokens. Repeated content is preserved, not silently deduplicated.
              'sequence_sha256': [hashlib.sha256(row.astype('<i8').tobytes()).hexdigest()
                                  for row in sequences],
              'baseline_sequence_ce': baseline.tolist(), 'layers': {}}
    for layer in CAUSAL_LAYERS:
        prefixes = cache.pop(layer)
        check = sorted({0, len(sequences) // 2, len(sequences) - 1})
        replay = suffix_losses(model, sequences[check], layer, [prefixes[i] for i in check])
        np.testing.assert_allclose(replay, baseline[check], rtol=0, atol=2e-5)
        moe = model.model.layers[layer].mlp
        unit = F.normalize(moe.gate.weight.detach().float(), dim=-1)
        gate, up = moe.experts.gate_up_proj.detach().chunk(2, dim=1)
        candidates, energies = orthogonal_candidates(gate, up, unit, count=CAUSAL_CANDIDATES, iterations=CAUSAL_POWER_ITERATIONS, seed=CAUSAL_CANDIDATE_SEED + layer)
        interventions: dict[str, CausalIntervention] = {}
        for setting in CAUSAL_INTERVENTIONS:
            conditions: dict[str, CausalCondition] = {}
            max_error = 0.
            for name, direction in [('router', unit), *[(f'candidate{i:02d}', candidates[:, i]) for i in range(CAUSAL_CANDIDATES)]]:
                observed: list[str] = []
                with intervene_olmoe_direction(moe.experts, direction, setting), capture_routes(moe.experts, observed):
                    losses = suffix_losses(model, sequences, layer, prefixes)
                    if observed != routes[layer]:
                        raise ValueError('intervention changed target routing or mixture weights')
                    if name in ('router', 'candidate00'):
                        observed.clear()
                        full = sequence_losses(model, sequences[:1])
                        np.testing.assert_allclose(losses[:1], full, rtol=0, atol=2e-5)
                        max_error = max(max_error, float(abs(losses[0] - full[0])))
                        if observed != routes[layer][:1]:
                            raise ValueError('full/cached target routing differs')
                conditions[name] = {'sequence_ce': losses.tolist(), 'target_routes_unchanged': True}
                print(f'layer {layer} {setting} {name}: CE {losses.mean():.9f}', flush=True)
            interventions[setting] = {'conditions': conditions, 'cached_full_max_error': max_error}
        result['layers'][str(layer)] = {'interventions': interventions,
            'candidate_energies': energies.cpu().tolist(), 'cached_baseline_max_error': float(abs(replay - baseline[check]).max())}
    result['complete'] = True
    return result
