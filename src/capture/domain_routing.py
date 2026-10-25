"""Six-domain routing counts used to choose the specialist-ablation targets."""

from collections.abc import Callable
from typing import cast

import numpy as np
import torch
from olmo_core.nn.moe.router import MoERouter
from olmo_core.nn.transformer.block import MoETransformerBlock
from olmo_core.nn.transformer.model import MoETransformer
from torch.utils.hooks import RemovableHandle

from src.constants import (
    PAPER_EVAL_BATCH_SIZE,
)
from src.types import (
    Float64Array,
    Int64Array,
    IntArray,
    RouterOutput,
)


@torch.inference_mode()
def routing_mass(
    model: MoETransformer, sequences: IntArray, batch_size: int = PAPER_EVAL_BATCH_SIZE,
) -> tuple[Int64Array, Float64Array]:
    routers = [cast(MoETransformerBlock, block).feed_forward_moe.router for block in model.blocks.values()]
    if model.training or not routers:
        raise ValueError('requires an eval model with MoE routers')
    device = next(model.parameters()).device
    counts = np.zeros((len(routers), routers[0].num_experts), dtype=np.int64)
    weights = np.zeros(counts.shape, dtype=np.float64)
    records: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    handles: list[RemovableHandle] = []

    def capture(layer: int) -> Callable[[MoERouter, tuple[torch.Tensor], RouterOutput], None]:
        def hook(module: MoERouter, args: tuple[torch.Tensor], output: RouterOutput) -> None:
            if layer in records:
                raise ValueError('router executed twice in one batch')
            records[layer] = (output[1].detach().long().cpu(), output[0].detach().float().cpu())
        return hook

    try:
        for layer, router in enumerate(routers):
            handles.append(router.register_forward_hook(capture(layer)))
        for start in range(0, len(sequences), batch_size):
            records.clear()
            model(torch.as_tensor(sequences[start:start + batch_size], device=device, dtype=torch.long))
            if len(records) != len(routers):
                raise ValueError('not every router was observed')
            for layer, (ids, mass) in records.items():
                if ids.shape != mass.shape or not torch.isfinite(mass).all() or (mass < 0).any():
                    raise ValueError('invalid native routing weights')
                flat = ids.flatten()
                counts[layer] += torch.bincount(flat, minlength=counts.shape[1]).numpy()
                weights[layer] += torch.zeros(counts.shape[1], dtype=torch.float64).scatter_add_(
                    0, flat, mass.flatten().double()).numpy()
    finally:
        for handle in handles:
            handle.remove()
    return counts, weights
