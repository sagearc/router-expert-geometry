"""Last-token measurements on native Transformers 5.1.0 OLMoE, no conversion."""

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle
from transformers.models.olmoe.modeling_olmoe import (
    OlmoeDecoderLayer,
    OlmoeExperts,
    OlmoeForCausalLM,
    OlmoeTopKRouter,
)

from src.constants import (
    OLMOE_EXPERT_CHUNK,
)
from src.types import PublicCaptureMetric


@torch.inference_mode()
def expert_response_norms(
    experts: OlmoeExperts, hidden: torch.Tensor,
) -> dict[Literal['w1_l2', 'w3_l2'], torch.Tensor]:
    """Full FP32 gate/up L2 responses for every expert on the same inputs."""
    h = hidden.to(device=experts.gate_up_proj.device, dtype=torch.float32)
    norms: dict[Literal['w1_l2', 'w3_l2'], list[torch.Tensor]] = {'w1_l2': [], 'w3_l2': []}
    # Preserve the retained layer-table projection's eight-expert GEMM shape.
    for start in range(0, experts.num_experts, OLMOE_EXPERT_CHUNK):
        gate, up = experts.gate_up_proj[start:start + OLMOE_EXPERT_CHUNK].split(experts.intermediate_dim, dim=1)
        projections: tuple[tuple[Literal['w1_l2', 'w3_l2'], torch.Tensor], ...] = (('w1_l2', gate), ('w3_l2', up))
        for key, weights in projections:
            projected = torch.einsum('ehd,bd->beh', weights.float(), h)
            norms[key].append(projected.norm(dim=-1).cpu())
    return {key: torch.cat(values, dim=-1) for key, values in norms.items()}


@torch.inference_mode()
def capture_olmoe(
    model: OlmoeForCausalLM, input_ids: torch.Tensor, attention_mask: torch.Tensor,
) -> dict[PublicCaptureMetric, torch.Tensor]:
    if not isinstance(model, OlmoeForCausalLM) or model.training:
        raise ValueError('requires a native eval-mode OlmoeForCausalLM')
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError('expected matching batch/sequence input and mask')
    lengths = attention_mask.sum(-1)
    expected_mask = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :] < lengths[:, None]
    if (lengths < 1).any() or not torch.equal(attention_mask.bool(), expected_mask):
        raise ValueError('requires nonempty right-padded prompts')
    positions = torch.arange(len(input_ids), device=input_ids.device) * input_ids.shape[1] + lengths - 1
    records: dict[int, dict[PublicCaptureMetric, torch.Tensor]] = {}
    handles: list[RemovableHandle] = []

    def hook_for(
        layer: int, experts: OlmoeExperts
    ) -> Callable[
        [OlmoeTopKRouter, tuple[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]], None
    ]:
        def hook(
            router: OlmoeTopKRouter, args: tuple[torch.Tensor],
            output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ) -> None:
            if layer in records:
                raise ValueError('router executed more than once')
            rows = positions.to(args[0].device)
            router_input = args[0].reshape(-1, router.hidden_dim)
            h = router_input[rows]
            probabilities, _, selected = output
            # Preserve native dtype AND GEMM shape before sampling last tokens.
            # A smaller GEMM can select a different low-precision kernel.
            raw = F.linear(router_input, router.weight)[rows].float()
            norms = expert_response_norms(experts, h)
            records[layer] = {
                'router_scores': raw.cpu(),
                'selection_scores': probabilities[rows].float().cpu(),
                'selected_experts': selected[rows].long().cpu(),
                'w1_l2': norms['w1_l2'],
                'w3_l2': norms['w3_l2'],
            }
        return hook

    try:
        for i, module in enumerate(model.model.layers):
            if not isinstance(module, OlmoeDecoderLayer):
                raise ValueError('requires native OlmoeDecoderLayer modules')
            layer = module
            handles.append(layer.mlp.gate.register_forward_hook(hook_for(i, layer.mlp.experts)))
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if len(records) != len(model.model.layers):
        raise ValueError('not every layer was observed')
    return {key: torch.stack([records[i][key] for i in range(len(records))], dim=1)
            for key in records[0]}
