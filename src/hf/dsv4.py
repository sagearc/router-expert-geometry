"""DeepSeek-V4 rank responses through the retained vLLM MXFP4 expert kernel."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import torch
from torch.utils.hooks import RemovableHandle

from src.constants import (
    DSV4_EXPERT_CHUNK,
    DSV4_EXPERTS,
    DSV4_HASH_LAYERS,
    DSV4_SCORE_LAYERS,
)
from src.types import Float64Array, Int64Array

if TYPE_CHECKING:
    from triton_kernels.matmul_ogs import GatherIndx
    from triton_kernels.tensor import Tensor as TritonTensor
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm.models.deepseek_v4.amd.model import (
        DeepseekV4DecoderLayer,
        DeepseekV4ForCausalLM,
        DeepseekV4MoE,
    )


@dataclass
class LayerCapture:
    hidden: list[torch.Tensor] = field(default_factory=list)
    logits: list[torch.Tensor] = field(default_factory=list)
    count: int = 0


@dataclass
class RankCapture:
    states: dict[int, LayerCapture]
    handles: list[RemovableHandle]
    expected: int


if TYPE_CHECKING:
    class _CapturingDeepseekV4ForCausalLM(DeepseekV4ForCausalLM):
        # Added only between install_capture() and finish_capture().
        _rank_capture: RankCapture


def score_layers(model: DeepseekV4ForCausalLM) -> list[tuple[int, DeepseekV4MoE]]:
    """The retained ROCm model has three hash layers followed by 40 scored layers."""
    layers = [(i, cast('DeepseekV4DecoderLayer', layer).ffn) for i, layer in enumerate(model.model.layers) if i >= DSV4_HASH_LAYERS]
    if [i for i, _ in layers] != list(DSV4_SCORE_LAYERS):
        raise ValueError('expected score-routed layers 3 through 42')
    if any(moe.n_routed_experts != DSV4_EXPERTS for _, moe in layers):
        raise ValueError('expected 256 experts at every score-routed layer')
    return layers


def install_capture(model: DeepseekV4ForCausalLM, total_prompt_tokens: int) -> None:
    """Capture one-token generation prefills only; do not silently truncate rows.

    Call on a fresh engine, without cached prefixes. The runner must generate
    exactly one output token, so every model forward belongs to the prefill.
    """
    if total_prompt_tokens < 1 or hasattr(model, '_rank_capture'):
        raise ValueError('requires positive token count and no existing capture')
    states: dict[int, LayerCapture] = {}
    handles: list[RemovableHandle] = []

    def capture(
        state: LayerCapture,
    ) -> Callable[[GateLinear, tuple[torch.Tensor], tuple[torch.Tensor, torch.Tensor | None]], None]:
        def hook(
            module: GateLinear, args: tuple[torch.Tensor],
            output: tuple[torch.Tensor, torch.Tensor | None],
        ) -> None:
            hidden = args[0].detach().reshape(-1, args[0].shape[-1])
            logits = output[0]
            logits = logits.detach().reshape(-1, logits.shape[-1])
            if len(hidden) != len(logits) or state.count + len(hidden) > total_prompt_tokens:
                raise ValueError('unexpected padding, decode, or extra capture rows')
            state.hidden.append(hidden.clone())
            state.logits.append(logits.clone())
            state.count += len(hidden)
        return hook

    try:
        for index, moe in score_layers(model):
            state = LayerCapture()
            states[index] = state
            handles.append(moe.gate.register_forward_hook(capture(state)))
    except Exception:
        for handle in handles:
            handle.remove()
        raise
    cast('_CapturingDeepseekV4ForCausalLM', model)._rank_capture = RankCapture(states, handles, total_prompt_tokens)


@torch.inference_mode()
def finish_capture(model: DeepseekV4ForCausalLM) -> tuple[Float64Array, Int64Array]:
    """Return only per-layer ranked response sums and token counts for the plot."""
    capturing = cast('_CapturingDeepseekV4ForCausalLM', model)
    capture = capturing._rank_capture
    states, handles, expected = capture.states, capture.handles, capture.expected
    try:
        sums = torch.zeros((len(DSV4_SCORE_LAYERS), DSV4_EXPERTS, 2), dtype=torch.float64)
        counts = torch.zeros((len(DSV4_SCORE_LAYERS), DSV4_EXPERTS), dtype=torch.int64)
        for out_index, (index, moe) in enumerate(score_layers(model)):
            state = states[index]
            if state.count != expected:
                raise ValueError('incomplete prompt-token capture')
            for hidden, logits in zip(state.hidden, state.logits, strict=True):
                bias = moe.gate.e_score_correction_bias
                if not isinstance(bias, torch.Tensor):
                    raise ValueError('score-routed gate requires a correction-bias tensor')
                selection = logits.float().sigmoid() + bias.float()
                ids = selection.argsort(dim=-1, descending=True)
                responses = expert_rank_responses(moe.experts.routed_experts, hidden, ids)
                if not torch.isfinite(responses).all():
                    raise ValueError('non-finite expert responses')
                sums[out_index] += responses.double().sum(dim=0).cpu()
                counts[out_index] += len(hidden)
        return sums.numpy(), counts.numpy()
    finally:
        for handle in handles:
            handle.remove()
        del capturing._rank_capture


def unpack_responses(
    packed: torch.Tensor, destination_indices: torch.Tensor, rows: int, experts_per_token: int,
) -> torch.Tensor:
    """Undo the native expert grouping, then split gate/up input projections."""
    width = packed.shape[-1]
    if width % 2 or destination_indices.numel() != rows * experts_per_token:
        raise ValueError('invalid packed gate/up dimensions')
    ordered = packed.reshape(-1, width)[destination_indices]
    ordered = ordered.reshape(rows, experts_per_token, width).float()
    gate, up = ordered.chunk(2, dim=-1)
    return torch.stack((gate.norm(dim=-1), up.norm(dim=-1)), dim=-1)


@torch.inference_mode()
def expert_rank_responses(
    routed: RoutedExperts, hidden: torch.Tensor, rank_ids: torch.Tensor,
) -> torch.Tensor:
    """Evaluate every ranked expert, without activation or mixture weighting.

    This intentionally targets the retained UnfusedOAITritonExperts backend;
    other packed layouts are not interchangeable with this measurement.
    """
    from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (
        UnfusedOAITritonExperts,
        matmul_ogs,
    )
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod

    method = cast(Mxfp4MoEMethod, routed.quant_method)
    if method.moe_kernel is None:
        raise ValueError('requires initialized MXFP4 experts')
    impl = method.moe_kernel.fused_experts
    if not isinstance(impl, UnfusedOAITritonExperts):
        raise ValueError('requires the retained UnfusedOAITritonExperts kernel')
    weights = cast('TritonTensor', routed.w13_weight)
    if rank_ids.shape != (len(hidden), DSV4_EXPERTS) or weights.shape[0] != DSV4_EXPERTS:
        raise ValueError('requires all 256 experts for every token')
    result = torch.empty((len(hidden), DSV4_EXPERTS, 2), device=hidden.device, dtype=torch.float32)
    for start in range(0, DSV4_EXPERTS, DSV4_EXPERT_CHUNK):
        ids = rank_ids[:, start:start + DSV4_EXPERT_CHUNK].to(torch.int32).contiguous()
        routing, gather, _ = impl._make_routing_data(
            ids, torch.ones_like(ids, dtype=torch.float32), DSV4_EXPERTS)
        packed = torch.empty((1, len(hidden) * DSV4_EXPERT_CHUNK, weights.shape[-1]),
                             device=hidden.device, dtype=torch.bfloat16)
        matmul_ogs(hidden.contiguous(), weights, impl.quant_config.w1_bias,
                   routing, gather_indx=gather, precision_config=impl.quant_config.w1_precision,
                   gammas=None, fused_activation=None, y=packed)
        # vLLM 0.25.0 annotates this as Tensor, but make_routing_data constructs GatherIndx.
        destination = cast('GatherIndx', gather).dst_indx
        result[:, start:start + DSV4_EXPERT_CHUNK] = unpack_responses(packed, destination, len(hidden), DSV4_EXPERT_CHUNK)
    return result
