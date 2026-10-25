"""The retained specialist intervention, with an explicit exclusion check."""

from collections.abc import Generator, Sequence
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from olmo_core.nn.moe.router import MoERouter
from olmo_core.nn.transformer.model import MoETransformer

from src.types import ExpertLogits, Float64Array, IntArray, TopK


@torch.inference_mode()
def sequence_nll(
    model: MoETransformer, sequences: IntArray, batch_size: int, device: torch.device | str,
) -> Float64Array:
    """Mean next-token NLL per unpadded sequence, retaining paired ordering."""
    losses: list[torch.Tensor] = []
    for start in range(0, len(sequences), batch_size):
        tokens = torch.as_tensor(sequences[start:start + batch_size], device=device, dtype=torch.long)
        logits = model(tokens)
        if not isinstance(logits, torch.Tensor):
            raise ValueError('unlabelled paper-model forward must return a logits tensor')
        values = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                 tokens[:, 1:].reshape(-1), reduction='none')
        losses.append(values.reshape(len(tokens), -1).mean(-1).cpu())
    return torch.cat(losses).double().numpy()


@contextmanager
def mask_specialist_logits(
    routers: Sequence[MoERouter],
    experts_per_layer: Sequence[IntArray | torch.Tensor] | IntArray | torch.Tensor,
) -> Generator[None, None, None]:
    """Mask raw logits before gating, exactly as the retained experiment did.

    Fail if a masked expert is still selected. With sigmoid gating, -inf becomes
    zero (plus epsilon), so a positive selection bias can otherwise reselect it.
    This check prevents describing such an execution as expert exclusion; it
    does not silently change the intervention to a different masking rule.
    """
    if len(routers) != len(experts_per_layer):
        raise ValueError('provide specialist IDs for every router')
    originals: list[tuple[MoERouter, ExpertLogits, TopK]] = []
    try:
        for router, expert_ids in zip(routers, experts_per_layer, strict=True):
            if router.training:
                raise ValueError('specialist ablation requires eval mode')
            ids = torch.as_tensor(expert_ids)
            if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
                raise ValueError('expert IDs must be one-dimensional integers')
            if len(ids.unique()) != len(ids) or (ids < 0).any() or (ids >= router.num_experts).any():
                raise ValueError('invalid or duplicate specialist IDs')
            if router.num_experts - len(ids) < router.top_k:
                raise ValueError('not enough remaining experts for native top-k')
            mask = torch.zeros(router.num_experts, dtype=torch.bool)
            mask[ids] = True
            original_logits, original_topk = router.get_expert_logits, router.get_top_k
            originals.append((router, original_logits, original_topk))

            def logits(
                x: torch.Tensor,
                original: ExpertLogits = original_logits,
                excluded: torch.Tensor = mask,
            ) -> torch.Tensor:
                values = original(x)
                return values.masked_fill(excluded.to(values.device), float('-inf'))

            def topk(
                scores: torch.Tensor, original: TopK = original_topk, excluded: torch.Tensor = mask,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                weights, selected = original(scores)
                if excluded.to(selected.device)[selected].any():
                    raise RuntimeError('logit masking did not exclude a specialist; selection bias reselected it')
                return weights, selected

            setattr(router, 'get_expert_logits', logits)
            setattr(router, 'get_top_k', topk)
        yield
    finally:
        for router, restore_logits, restore_topk in originals:
            setattr(router, 'get_expert_logits', restore_logits)
            setattr(router, 'get_top_k', restore_topk)
