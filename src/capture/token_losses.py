"""Per-token NLL and selection-boundary capture for the paper models."""

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Literal, cast

import torch
import torch.nn.functional as F
from olmo_core.nn.moe.router import MoERouter
from olmo_core.nn.transformer.block import MoETransformerBlock
from olmo_core.nn.transformer.model import MoETransformer

from src.constants import (
    PAPER_TOP_K,
)
from src.types import TopK


@contextmanager
def selection_margins(
    routers: Sequence[MoERouter],
) -> Generator[dict[int, torch.Tensor], None, None]:
    """Observe native gated scores, including selection bias, without rerouting."""
    records: dict[int, torch.Tensor] = {}
    originals: list[tuple[MoERouter, TopK | None]] = []
    try:
        for layer, router in enumerate(routers):
            if router.training or router.uniform_expert_assignment:
                raise ValueError('requires eval-mode score-based routing')
            if router.top_k != PAPER_TOP_K or router.num_experts <= PAPER_TOP_K:
                raise ValueError('the paper boundary is between experts ranked 6 and 7')
            original = router.get_top_k
            prior_override = cast(TopK | None, router.__dict__.get('get_top_k'))
            originals.append((router, prior_override))

            def capture(
                scores: torch.Tensor, original: TopK = original,
                router: MoERouter = router, layer: int = layer,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                if layer in records:
                    raise ValueError('router executed more than once in a captured batch')
                weights, selected = original(scores)
                with torch.no_grad():
                    selection = scores.float()
                    bias = router.score_bias
                    if isinstance(bias, torch.Tensor):
                        selection = selection + bias
                    if not torch.isfinite(selection).all():
                        raise ValueError('nonfinite selection scores')
                    ranked = selection.sort(dim=-1, descending=True).values
                    actual = selection.gather(-1, selected).sort(dim=-1, descending=True).values
                    # Compare values, not IDs: tied cutoffs may select different IDs.
                    torch.testing.assert_close(actual, ranked[..., :PAPER_TOP_K], rtol=0, atol=0)
                    records[layer] = (ranked[..., PAPER_TOP_K - 1] - ranked[..., PAPER_TOP_K]).cpu()
                return weights, selected

            setattr(router, 'get_top_k', capture)
        yield records
    finally:
        for router, override in originals:
            if override is None:
                del router.get_top_k
            else:
                setattr(router, 'get_top_k', override)


@torch.inference_mode()
def capture_token_losses(
    model: MoETransformer, tokens: torch.Tensor,
) -> dict[Literal['nll', 'targets', 'margin6'], torch.Tensor]:
    """Capture one unpadded batch; position t predicts target token t+1."""
    if model.training or tokens.ndim != 2 or tokens.shape[1] < 2:
        raise ValueError('requires eval model and (batch, sequence >= 2) tokens')
    routers = [cast(MoETransformerBlock, block).feed_forward_moe.router for block in model.blocks.values()]
    with selection_margins(routers) as records:
        logits = model(tokens)
    if not isinstance(logits, torch.Tensor):
        raise ValueError('unlabelled paper-model forward must return a logits tensor')
    if len(records) != len(routers) or not records:
        raise ValueError('not every router was observed')
    nll = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                          tokens[:, 1:].reshape(-1), reduction='none')
    if not torch.isfinite(nll).all():
        raise ValueError('nonfinite per-token loss; capture aborted')
    return {
        'nll': nll.reshape(tokens.shape[0], -1).cpu(),
        'targets': tokens[:, 1:].cpu(),
        'margin6': torch.stack([records[i][:, :-1] for i in range(len(routers))]),
    }
