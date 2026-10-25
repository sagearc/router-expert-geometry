"""DeepSeek-V3's sequence-wise auxiliary loss on OLMo-core's linear router."""

from dataclasses import dataclass, replace
from typing import TypedDict, Unpack, cast

import torch
from olmo_core.distributed.utils import (
    _HiddenTensor,
    get_local_tensor,
    hide_from_torch,
    unhide_from_torch,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.buffer_cache import BufferCache
from olmo_core.nn.moe import MoEBase, MoEConfig
from olmo_core.nn.moe.loss import MoELoadBalancingLossGranularity
from olmo_core.nn.moe.router import (
    MoELinearRouter,
    MoERouterConfig,
    MoERouterGatingFunction,
)
from olmo_core.ops.moe import batched_histc
from olmo_core.train.common import ReduceType


def sequence_auxiliary_loss(
    *,
    num_experts: int,
    top_k: int,
    expert_scores: torch.Tensor,
    batched_batch_size_per_expert: torch.Tensor,
    loss_div_factor: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Complementary sequence-wise balance loss (arXiv:2412.19437, section 2.1.2).

    Per sequence, ``N * sum_i f_i P_i`` with ``f_i`` the fraction of the sequence's
    ``K * S`` assignments routed to expert ``i`` and ``P_i`` the mean L1-normalized
    gate score of expert ``i``, averaged over sequences. ``loss_div_factor`` is the
    whole batch's token count, so summing microbatch contributions yields the
    batch mean instead of overcounting by the number of microbatches.
    """
    scores = cast(torch.Tensor, get_local_tensor(expert_scores))
    counts = cast(torch.Tensor, get_local_tensor(batched_batch_size_per_expert))
    _, sequence_length, _ = scores.shape
    normalized = scores / scores.sum(dim=-1, keepdim=True).clamp(min=1e-20)
    fraction = counts.float() / float(sequence_length * top_k)
    per_sequence = float(num_experts) * (fraction * normalized.mean(dim=1)).sum(dim=-1)
    if loss_div_factor is None:
        return per_sequence.mean()
    return per_sequence.sum() / (loss_div_factor / float(sequence_length))


class LinearRouterOptions(TypedDict, total=False):
    """Keyword options of OLMo-core's linear router constructor."""

    top_k: int
    jitter_eps: float | None
    normalize_expert_weights: float | None
    uniform_expert_assignment: bool
    bias_gamma: float | None
    gating_function: MoERouterGatingFunction
    lb_loss_weight: float | None
    lb_loss_granularity: MoELoadBalancingLossGranularity
    z_loss_weight: float | None
    dtype: torch.dtype
    init_device: str


class PaperSeqAuxRouter(MoELinearRouter):
    """A linear router that adds the sequence-wise loss to its auxiliary loss.

    The loss needs the gated scores and per-sequence assignment counts that
    :meth:`MoERouter.forward` keeps internal, so they are recomputed from the same
    logits. The extra cost is one ``d_model x num_experts`` product per token,
    and the gradient is identical to computing the loss inline.
    """

    def __init__(
        self, *, d_model: int, num_experts: int, seq_aux_loss_weight: float,
        **options: Unpack[LinearRouterOptions],
    ) -> None:
        super().__init__(d_model=d_model, num_experts=num_experts, **options)
        if self.jitter_eps is not None:
            raise OLMoConfigurationError('jitter would differ between routing and the recomputed scores')
        if seq_aux_loss_weight <= 0:
            raise OLMoConfigurationError('seq_aux_loss_weight must be positive')
        self.seq_aux_loss_weight = seq_aux_loss_weight
        self._seq_aux_loss: _HiddenTensor | None = None

    @property
    def seq_aux_loss(self) -> torch.Tensor:
        """Unscaled loss accumulated since the last metrics reset."""
        if self._seq_aux_loss is None:
            self._seq_aux_loss = hide_from_torch(torch.zeros([], device=self.device))
        elif self._seq_aux_loss.device != self.device:
            self._seq_aux_loss = self._seq_aux_loss.to(self.device)
        return cast(torch.Tensor, unhide_from_torch(self._seq_aux_loss))

    def _gate(self, logits: torch.Tensor) -> torch.Tensor:
        if self.gating_function == MoERouterGatingFunction.softmax:
            return logits.softmax(dim=-1)
        if self.gating_function == MoERouterGatingFunction.sigmoid:
            return torch.sigmoid(logits) + 1e-7
        raise NotImplementedError(self.gating_function)

    def forward(
        self, x: torch.Tensor, *, loss_div_factor: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        expert_weights, expert_indices, batch_size_per_expert, aux_loss = super().forward(
            x, loss_div_factor=loss_div_factor)
        if self.training and torch.is_grad_enabled():
            with torch.autocast(enabled=False, device_type=x.device.type):
                loss = sequence_auxiliary_loss(
                    num_experts=self.num_experts, top_k=self.top_k,
                    expert_scores=self._gate(self.get_expert_logits(x).float()),
                    batched_batch_size_per_expert=batched_histc(expert_indices, self.num_experts).sum(dim=1),
                    loss_div_factor=loss_div_factor,
                )
            self.seq_aux_loss.add_(loss.detach())
            scaled = self.seq_aux_loss_weight * loss
            aux_loss = scaled if aux_loss is None else aux_loss + scaled
        return expert_weights, expert_indices, batch_size_per_expert, aux_loss

    @torch.no_grad()
    def compute_metrics(self, reset: bool = True) -> dict[str, tuple[torch.Tensor, ReduceType | None]]:
        out = super().compute_metrics(reset=False)
        out['seq aux loss'] = (self.seq_aux_loss_weight * self.seq_aux_loss, ReduceType.mean)
        out['seq aux loss unscaled'] = (self.seq_aux_loss.clone(), ReduceType.mean)
        if reset:
            self.reset_metrics()
        return out

    def reset_metrics(self) -> None:
        super().reset_metrics()
        self.seq_aux_loss.zero_()


@dataclass
class PaperSeqAuxConfig(MoERouterConfig):
    """Build :class:`PaperSeqAuxRouter`; the inherited ``name`` is unused."""

    seq_aux_loss_weight: float = 1e-4

    def build(
        self,
        d_model: int,
        num_experts: int,
        *,
        lb_loss_weight: float | None = None,
        lb_loss_granularity: MoELoadBalancingLossGranularity = MoELoadBalancingLossGranularity.local_batch,
        z_loss_weight: float | None = None,
        dtype: torch.dtype | None = None,
        init_device: str = 'cpu',
    ) -> PaperSeqAuxRouter:
        options = self.as_dict(exclude_none=True, recurse=False)
        options.pop('name')
        options.pop('seq_aux_loss_weight')
        options.update(
            lb_loss_weight=lb_loss_weight, lb_loss_granularity=lb_loss_granularity,
            z_loss_weight=z_loss_weight, init_device=init_device,
        )
        if self.dtype is not None:
            options['dtype'] = self.dtype.as_pt()
        elif dtype is not None:
            options['dtype'] = dtype
        return PaperSeqAuxRouter(
            d_model=d_model, num_experts=num_experts, seq_aux_loss_weight=self.seq_aux_loss_weight,
            **cast(LinearRouterOptions, options),
        )


@dataclass
class PaperMoEConfig(MoEConfig):
    """Scale the sequence-wise loss weight per layer, as OLMo-core scales its router losses."""

    def build(
        self, d_model: int, *, n_layers: int = 1, init_device: str = 'cpu', cache: BufferCache | None = None,
    ) -> MoEBase:
        config = self
        if self.scale_loss_by_num_layers and isinstance(self.router, PaperSeqAuxConfig):
            router = replace(self.router, seq_aux_loss_weight=self.router.seq_aux_loss_weight / n_layers)
            config = replace(self, router=router)
        return MoEConfig.build(config, d_model, n_layers=n_layers, init_device=init_device, cache=cache)
