"""Public OLMoE router-axis removal versus all 32 orthogonal controls.

Native Transformers model, no architecture conversion. Historical all-controls
protocol: layers 0/5/8/10/15, 512 predictions, 16 sequences/domain after skipping
four calibration sequences, block power iteration 12, seed 12987 + layer.
"""

import hashlib
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import OlmoeConfig, OlmoeForCausalLM
from transformers.models.olmoe.modeling_olmoe import OlmoeExperts

from src.artifacts import CausalCondition, PairedSummary, PublicCausalResult
from src.constants import (
    CAUSAL_BOOTSTRAP_DRAWS,
    CAUSAL_BOOTSTRAP_SEED,
    CAUSAL_CALIBRATION_SEQUENCES,
    CAUSAL_CANDIDATE_SEED,
    CAUSAL_CANDIDATES,
    CAUSAL_LAYERS,
    CAUSAL_POWER_ITERATIONS,
    CAUSAL_PREDICTIONS,
    CAUSAL_SEQUENCES_PER_DOMAIN,
    CI95_QUANTILES,
    OLMOE_EXPERTS,
    OLMOE_LAYERS,
    OLMOE_MODEL_ID,
    OLMOE_TOP_K,
    VALIDATION_DOMAINS,
)
from src.sequences import evaluation_sequences
from src.types import Float64Array, FloatArray, IntArray


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


@contextmanager
def remove_olmoe_direction(
    experts: OlmoeExperts, directions: torch.Tensor, *, alpha: float = 1.0,
) -> Generator[None, None, None]:
    """Keep native routing and decoder normalization; change expert inputs only.

    Uses the pinned Transformers eager expert loop, with one projection before
    gate/up multiplication. No checkpoint conversion or decoder replacement.
    Intended for single-device inference, not concurrent calls or training.
    """
    if not isinstance(experts, OlmoeExperts) or experts.training:
        raise ValueError('requires native eval-mode OlmoeExperts')
    if cast(OlmoeConfig, experts.config)._experts_implementation != 'eager':
        raise ValueError('baseline and intervention must both use eager experts')
    if directions.shape != (experts.num_experts, experts.hidden_dim) or not 0 <= alpha <= 1:
        raise ValueError('invalid directions or projection strength')
    unit = directions.to(device=experts.gate_up_proj.device, dtype=torch.float32)
    if not torch.isfinite(unit).all() or not torch.allclose(
        unit.norm(dim=-1), torch.ones(len(unit), device=unit.device), atol=1e-5, rtol=1e-5
    ):
        raise ValueError('directions must be finite unit vectors')
    if 'forward' in experts.__dict__:
        raise ValueError('expert forward is already overridden')
    original = experts.forward

    def forward(
        self: OlmoeExperts, hidden_states: torch.Tensor,
        top_k_index: torch.Tensor, top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if alpha == 0:
            return original(hidden_states, top_k_index, top_k_weights)
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
            state = (values - alpha * (values * direction).sum(-1, keepdim=True) * direction).to(state.dtype)
            gate, up = F.linear(state, self.gate_up_proj[expert]).chunk(2, dim=-1)
            output = F.linear(self.act_fn(gate) * up, self.down_proj[expert])
            output = output * top_k_weights[tokens, positions, None]
            result.index_add_(0, tokens, output.to(result.dtype))
        return result

    setattr(experts, 'forward', MethodType(forward, experts))
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


def paired_summary(values: FloatArray) -> PairedSummary:
    rng = np.random.default_rng(CAUSAL_BOOTSTRAP_SEED)
    means = values[rng.integers(0, len(values), size=(CAUSAL_BOOTSTRAP_DRAWS, len(values)))].mean(-1)
    return {'mean': float(values.mean()), 'ci95': np.quantile(means, CI95_QUANTILES).tolist()}


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
        OLMOE_MODEL_ID, revision=revision, dtype=torch.bfloat16,
        attn_implementation='eager', experts_implementation='eager',
    ).to('cuda').eval()  # type: ignore[arg-type]  # Transformers 5.1 @wraps loses the bound .to() signature.
    config = model.config
    if (config.num_hidden_layers, config.num_experts, config.num_experts_per_tok) != (OLMOE_LAYERS, OLMOE_EXPERTS, OLMOE_TOP_K):
        raise ValueError('expected public OLMoE 16-layer / 64-expert / top-8 architecture')
    return cast(OlmoeForCausalLM, model)


def run(revision: str, eval_root: Path) -> PublicCausalResult:
    sequences, sample_ids, sources = load_sequences(eval_root)
    model = load_model(revision)
    vocab_size = model.config.vocab_size
    if vocab_size is None or (sequences < 0).any() or (sequences >= vocab_size).any():
        raise ValueError('token ID outside vocabulary')
    baseline = sequence_losses(model, sequences)
    result: PublicCausalResult = {'revision': revision, 'transformers': transformers.__version__,
              'torch': torch.__version__, 'backend': 'native_eager_bfloat16',
              'protocol': {'candidate_count': CAUSAL_CANDIDATES, 'power_iterations': CAUSAL_POWER_ITERATIONS,
                           'candidate_seed': CAUSAL_CANDIDATE_SEED, 'bootstrap_seed': CAUSAL_BOOTSTRAP_SEED,
                           'bootstrap_draws': CAUSAL_BOOTSTRAP_DRAWS, 'predictions_per_sequence': CAUSAL_PREDICTIONS,
                           'calibration_sequences_skipped_per_domain': CAUSAL_CALIBRATION_SEQUENCES,
                           'evaluation_sequences_per_domain': CAUSAL_SEQUENCES_PER_DOMAIN},
              'sources_sha256': sources, 'sample_ids': sample_ids,
              # Canonical little-endian int64 bytes, including input and target
              # tokens. Repeated content is preserved, not silently deduplicated.
              'sequence_sha256': [hashlib.sha256(row.astype('<i8').tobytes()).hexdigest()
                                  for row in sequences],
              'baseline_sequence_ce': baseline.tolist(), 'layers': {}}
    for layer in CAUSAL_LAYERS:
        moe = model.model.layers[layer].mlp
        unit = F.normalize(moe.gate.weight.detach().float(), dim=-1)
        gate, up = moe.experts.gate_up_proj.detach().chunk(2, dim=1)
        candidates, energies = orthogonal_candidates(gate, up, unit, count=CAUSAL_CANDIDATES, iterations=CAUSAL_POWER_ITERATIONS, seed=CAUSAL_CANDIDATE_SEED + layer)
        with remove_olmoe_direction(moe.experts, unit, alpha=0):
            identity = sequence_losses(model, sequences)
        np.testing.assert_array_equal(identity, baseline)
        conditions: dict[str, CausalCondition] = {}
        router_losses = None
        for name, direction in [('router', unit), *[(f'candidate{i:02d}', candidates[:, i]) for i in range(CAUSAL_CANDIDATES)]]:
            with remove_olmoe_direction(moe.experts, direction):
                losses = sequence_losses(model, sequences)
            record: CausalCondition = {'sequence_ce': losses.tolist(), 'delta_ce': paired_summary(losses - baseline)}
            if name == 'router':
                router_losses = losses
            else:
                if router_losses is None:
                    raise ValueError('router condition must precede controls')
                record['router_minus_control'] = paired_summary(router_losses - losses)
            conditions[name] = record
            print(f'layer {layer} {name}: delta CE {record["delta_ce"]["mean"]:.6f}', flush=True)
        result['layers'][str(layer)] = {'conditions': conditions, 'candidate_energies': energies.cpu().tolist()}
    return result
