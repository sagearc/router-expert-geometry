"""Last-token measurements for unsharded paper models, without changing routing."""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from olmo_core.nn.moe.mlp import DroplessMoEMLP
from olmo_core.nn.moe.router import MoERouter
from olmo_core.nn.transformer.block import MoETransformerBlock
from olmo_core.nn.transformer.model import MoETransformer
from torch.utils.hooks import RemovableHandle

from src.types import CaptureMetric, ResponseMetric, RouterOutput


def expert_responses(h: torch.Tensor, mlp: DroplessMoEMLP) -> dict[ResponseMetric, torch.Tensor]:
    """Float32 gate/up projections for OLMo-core's dropless weight layout."""
    shape = (mlp.num_experts, mlp.hidden_size, h.shape[-1])
    gate = torch.einsum('ehd,bd->beh', mlp.w1.float().view(shape), h.float())
    up = torch.einsum('ehd,bd->beh', mlp.w3.float().view(shape), h.float())
    return {
        'w1_top10_l2': gate.topk(10, dim=-1).values.square().sum(-1).sqrt(),
        'w1_silu_mean': F.silu(gate).mean(dim=-1),
        'w1_l2': gate.norm(dim=-1),
        'w3_l2': up.norm(dim=-1),
    }


@torch.inference_mode()
def capture_last_tokens(
    model: MoETransformer, input_ids: torch.Tensor, lengths: torch.Tensor,
) -> dict[CaptureMetric, torch.Tensor]:
    """Run one right-padded batch and return CPU tensors (titles, layers, experts).

    Pass an eval-mode, unsharded OLMo-core paper model. ``lengths`` counts real
    tokens, including any prompt tokens, before right padding. The caller owns
    tokenization and must reuse the same token IDs across checkpoints.
    Scores are raw router logits recomputed in float32, as in the retained
    analysis; selected IDs come from the actual forward pass, including bias.
    """
    if model.training:
        raise ValueError('capture requires model.eval()')
    if input_ids.ndim != 2 or lengths.shape != input_ids.shape[:1]:
        raise ValueError('expected input_ids (batch, sequence) and lengths (batch,)')
    if lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError('lengths must be integers')
    if (lengths < 1).any() or (lengths > input_ids.shape[1]).any():
        raise ValueError('lengths must identify real tokens within the padded batch')

    records: dict[int, dict[CaptureMetric, torch.Tensor]] = {}
    handles: list[RemovableHandle] = []
    layers: list[tuple[MoERouter, DroplessMoEMLP]] = []
    for block in model.blocks.values():
        if not isinstance(block, MoETransformerBlock):
            raise ValueError('capture requires native MoETransformerBlock modules')
        moe = block.feed_forward_moe
        mlp = moe.experts.mlp
        if not isinstance(mlp, DroplessMoEMLP) or mlp.num_local_experts != mlp.num_experts:
            raise ValueError('capture supports unsharded dropless experts only')
        if mlp.hidden_sharding_degree != 1:
            raise ValueError('capture does not support tensor-sharded experts')
        if mlp.hidden_size < 10:
            raise ValueError('top-10 response requires at least ten hidden units')
        layers.append((moe.router, mlp))

    def hook_for(
        layer: int, mlp: DroplessMoEMLP
    ) -> Callable[[MoERouter, tuple[torch.Tensor], RouterOutput], None]:
        def hook(router: MoERouter, args: tuple[torch.Tensor], output: RouterOutput) -> None:
            if layer in records:
                raise ValueError('router executed more than once in this forward pass')
            rows = torch.arange(input_ids.shape[0], device=input_ids.device)
            last = lengths.to(input_ids.device) - 1
            h = args[0][rows, last].float()
            records[layer] = {
                'router_scores': router.get_expert_logits(h).float().cpu(),
                'selected_experts': output[1][rows, last].long().cpu(),
                **{key: value.cpu() for key, value in expert_responses(h, mlp).items()},
            }
        return hook

    try:
        for layer, (router, mlp) in enumerate(layers):
            handles.append(router.register_forward_hook(hook_for(layer, mlp)))
        model(input_ids)
    finally:
        for handle in handles:
            handle.remove()
    if len(records) != len(layers) or not records:
        raise ValueError('not every MoE layer was captured')
    return {key: torch.stack([records[i][key] for i in range(len(layers))], dim=1)
            for key in records[0]}
