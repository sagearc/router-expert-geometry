"""The paper's K-means router on OLMo-core's router base.

OLMo-core only ships learned linear routers. The paper's fourth recipe (the
historical fork at commit 8f705a4) routes by cosine similarity between the
hidden state and one running-mean centroid per expert. Centroids start at zero,
so every score is zero until the first update, and then move toward the mean of
their routed hidden states by ``centroid_lr_lambda`` times the current learning
rate. Load is balanced by DeepSeek-V3's auxiliary-loss-free selection bias with
a step of ``bias_lr_lambda`` times the learning rate. No gradient reaches the
centroids or the bias. Checkpoints hold ``_centroid``, ``_centroid_step`` and
``score_bias``, the keys of the retained K-means checkpoints.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from olmo_core.distributed.utils import (
    _HiddenTensor,
    distribute_like,
    get_local_tensor,
    hide_from_torch,
    is_distributed,
    unhide_from_torch,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.moe.loss import MoELoadBalancingLossGranularity
from olmo_core.nn.moe.router import MoERouter, MoERouterConfig, MoERouterGatingFunction
from olmo_core.ops.moe import batched_histc


class PaperCentroidRouter(MoERouter):
    """Cosine routing against running-mean centroids, balanced by a selection bias."""

    score_bias: torch.Tensor
    _centroid: torch.Tensor
    _centroid_step: torch.Tensor

    def __init__(
        self, *, d_model: int, num_experts: int, top_k: int,
        bias_lr_lambda: float, centroid_lr_lambda: float, init_device: str = 'cpu',
    ) -> None:
        if bias_lr_lambda <= 0 or centroid_lr_lambda <= 0:
            raise OLMoConfigurationError('bias_lr_lambda and centroid_lr_lambda must be positive')
        super().__init__(d_model=d_model, num_experts=num_experts, top_k=top_k, init_device=init_device)
        self.bias_lr_lambda = bias_lr_lambda
        self.centroid_lr_lambda = centroid_lr_lambda
        # Both update steps scale with the optimizer's learning rate, which
        # RouterLearningRateCallback supplies before every batch.
        self.learning_rate: float | None = None
        self.register_buffer('score_bias', torch.zeros(num_experts, device=init_device))
        self.register_buffer('_centroid', torch.zeros(num_experts, d_model, device=init_device))
        self.register_buffer('_centroid_step', torch.zeros((), dtype=torch.long, device=init_device))
        # Per-batch sums, hidden from FSDP and torch.compile like the base-class accumulators.
        self._routed_sum: _HiddenTensor | None = None
        self._routed_count: _HiddenTensor | None = None

    @property
    def device(self) -> torch.device:
        return self._centroid.device if self._centroid.device.type != 'meta' else torch.device('cpu')

    @property
    def weight(self) -> torch.Tensor:
        """There is no learned routing matrix.

        OLMo-core's MoE initializer fills ``router.weight`` for every router. An
        empty tensor gives it nothing to fill, so the centroids stay at zero.
        """
        return torch.empty(0, device=self.device)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.score_bias.zero_()
        self._centroid.zero_()
        self._centroid_step.zero_()
        self._discard_batch()

    def _discard_batch(self) -> None:
        self._routed_sum = None
        self._routed_count = None

    def get_expert_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Cosine similarity to every centroid; zero while the centroids are zero."""
        flat = x.float().reshape(-1, self.d_model)
        scores = F.normalize(flat, dim=-1) @ F.normalize(self._centroid, dim=-1).T
        return scores.reshape(*x.shape[:-1], self.num_experts)

    def get_top_k(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select with the balancing bias added; the returned weights exclude it."""
        with torch.no_grad():
            _, expert_indices = torch.topk(scores + self.score_bias, self.top_k, dim=-1)
        return scores.gather(-1, expert_indices), expert_indices

    def forward(
        self, x: torch.Tensor, *, loss_div_factor: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        del loss_div_factor  # No auxiliary losses: the selection bias balances load.
        # Cosine scores are bounded gate weights already; no gating function is applied.
        scores = self.get_expert_logits(x)
        expert_weights, expert_indices = self.get_top_k(scores)
        with torch.no_grad():
            histogram = batched_histc(expert_indices, self.num_experts)
            batch_size_per_expert = histogram.reshape(-1, self.num_experts).sum(dim=0)
        if self.training and torch.is_grad_enabled():
            self.batch_size_per_expert += batch_size_per_expert
            self._accumulate(x, expert_indices, batch_size_per_expert)
        return expert_weights, expert_indices, batch_size_per_expert, None

    @torch.no_grad()
    def _accumulate(
        self, x: torch.Tensor, expert_indices: torch.Tensor, batch_size_per_expert: torch.Tensor,
    ) -> None:
        tokens = x.detach().float().reshape(-1, self.d_model).repeat_interleave(self.top_k, dim=0)
        routed_sum = torch.zeros_like(self._centroid).index_add_(0, expert_indices.reshape(-1), tokens)
        if self._routed_sum is None or self._routed_count is None:
            self._routed_sum = hide_from_torch(routed_sum)
            self._routed_count = hide_from_torch(batch_size_per_expert.float().clone())
        else:
            unhide_from_torch(self._routed_sum).add_(routed_sum)
            unhide_from_torch(self._routed_count).add_(batch_size_per_expert)

    @torch.no_grad()
    def post_batch(self, dry_run: bool = False) -> None:
        if not self.training or self._routed_sum is None or self._routed_count is None:
            return
        if self.learning_rate is None:
            raise ValueError('the K-means router needs the current learning rate before each batch')
        step = self.centroid_lr_lambda * self.learning_rate
        if not 0 < step <= 1:
            raise OLMoConfigurationError(f'centroid step {step} is outside (0, 1]')
        routed_sum = unhide_from_torch(self._routed_sum)
        routed_count = unhide_from_torch(self._routed_count)
        if is_distributed():
            dist.all_reduce(routed_sum, group=self.group)
            dist.all_reduce(routed_count, group=self.group)
        if not dry_run:
            # Dual step: push every expert's bias toward uniform load, as in DeepSeek-V3.
            bias_delta = self.bias_lr_lambda * self.learning_rate * (routed_count.mean() - routed_count).sign()
            get_local_tensor(self.score_bias).add_(get_local_tensor(distribute_like(self.score_bias, bias_delta)))
            # Primal step: move each routed expert's centroid toward the mean of its tokens.
            routed = routed_count > 0
            mean = routed_sum[routed] / routed_count[routed, None]
            self._centroid[routed] = self._centroid[routed].lerp(mean, step)
            self._centroid_step += 1
        self._discard_batch()


@dataclass
class PaperCentroidConfig(MoERouterConfig):
    """Build :class:`PaperCentroidRouter` in place of OLMo-core's linear router.

    ``name`` and ``dtype`` are inherited and unused: there is one implementation
    and centroids are always float32. ``gating_function`` must keep its default;
    cosine scores are the gate weights. Every other inherited option is rejected.
    """

    bias_lr_lambda: float = 1.0
    centroid_lr_lambda: float = 10.0

    def num_params(self, d_model: int, num_experts: int) -> int:
        del d_model, num_experts  # Centroids are statistics, not parameters.
        return 0

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
    ) -> PaperCentroidRouter:
        del lb_loss_granularity, dtype
        unsupported = (self.jitter_eps, self.normalize_expert_weights, self.bias_gamma, lb_loss_weight, z_loss_weight)
        if (any(value is not None for value in unsupported) or self.uniform_expert_assignment
                or self.gating_function != MoERouterGatingFunction.softmax):
            raise OLMoConfigurationError(
                'the paper K-means router supports only top_k, bias_lr_lambda and centroid_lr_lambda')
        return PaperCentroidRouter(
            d_model=d_model, num_experts=num_experts, top_k=self.top_k,
            bias_lr_lambda=self.bias_lr_lambda, centroid_lr_lambda=self.centroid_lr_lambda,
            init_device=init_device,
        )
