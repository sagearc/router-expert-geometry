"""Retained Qwen rank-response measurement on native Transformers 5.1."""

from collections.abc import Callable

import numpy as np
import torch
from torch.utils.hooks import RemovableHandle
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeExperts,
    Qwen3MoeForCausalLM,
    Qwen3MoeSparseMoeBlock,
    Qwen3MoeTopKRouter,
)

from src.constants import (
    QWEN_EXPERT_CHUNK,
)
from src.types import Float64Array, FloatArray


@torch.inference_mode()
def rank_responses(
    model: Qwen3MoeForCausalLM, input_ids: torch.Tensor, attention_mask: torch.Tensor,
) -> tuple[Float64Array, int, int]:
    """Return rank sums, observation count and legacy/native rank differences.

    Each valid token/layer contributes equally. Historical rank scores apply a
    second softmax to native probabilities; this is not raw r^T h scoring.
    """
    if not isinstance(model, Qwen3MoeForCausalLM) or model.training:
        raise ValueError('requires native eval-mode Qwen3MoeForCausalLM')
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape or not attention_mask.bool().any():
        raise ValueError('requires matching nonempty token input and mask')
    keep = attention_mask.bool().flatten()
    totals: list[Float64Array] = []
    observations: list[int] = []
    changed: list[int] = []
    seen: set[int] = set()
    handles: list[RemovableHandle] = []

    def hook_for(
        index: int, experts: Qwen3MoeExperts
    ) -> Callable[
        [Qwen3MoeTopKRouter, tuple[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]], None
    ]:
        def hook(
            router: Qwen3MoeTopKRouter, args: tuple[torch.Tensor],
            output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ) -> None:
            if index in seen:
                raise ValueError('router executed twice in one capture')
            seen.add(index)
            h = args[0].reshape(-1, router.hidden_dim)[keep.to(args[0].device)].float()
            probabilities = output[0][keep.to(output[0].device)]
            native = probabilities.float().cpu().numpy()
            if not np.isfinite(native).all():
                raise ValueError('nonfinite Qwen router scores')
            legacy = probabilities.softmax(-1).float().cpu().numpy()
            order = np.argsort(-legacy, axis=-1, kind='stable')
            changed.append(int(np.count_nonzero(order != np.argsort(-native, axis=-1, kind='stable'))))
            responses: list[list[FloatArray]] = [[], []]
            # Retained rank-v2 run uses the capture's 16-expert default.
            for start in range(0, experts.num_experts, QWEN_EXPERT_CHUNK):
                packed = experts.gate_up_proj[start:start + QWEN_EXPERT_CHUNK]
                for target, weights in zip(responses, packed.chunk(2, dim=1), strict=True):
                    values = torch.einsum('ehd,bd->beh', weights.float(), h)
                    norms = values.norm(dim=-1).cpu().numpy()
                    if not np.isfinite(norms).all():
                        raise ValueError('nonfinite Qwen expert responses')
                    target.append(norms)
            totals.append(np.stack([
                np.take_along_axis(np.concatenate(parts, axis=1), order, axis=1).sum(axis=0, dtype=np.float64)
                for parts in responses
            ]))
            observations.append(len(h))
        return hook

    try:
        for i, module in enumerate(model.model.layers):
            if not isinstance(module, Qwen3MoeDecoderLayer):
                raise ValueError('requires native Qwen3MoeDecoderLayer modules')
            layer = module
            if not isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                raise ValueError('requires a sparse MoE block at every Qwen layer')
            handles.append(layer.mlp.gate.register_forward_hook(hook_for(i, layer.mlp.experts)))
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if len(seen) != len(model.model.layers):
        raise ValueError('not every Qwen layer was measured')
    return np.sum(totals, axis=0), sum(observations), sum(changed)
