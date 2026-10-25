"""The four 1B paper architectures on OLMo-core's MoE transformer.

These are our trained models, not converters for public OLMoE checkpoints.
In particular, reordered normalization must not be applied to public OLMoE.
"""


from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.moe import MoERouterGatingFunction
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.config import TransformerBlockConfig
from olmo_core.optim import (
    AdamWConfig,
    CosWithWarmup,
    OptimGroupOverride,
    SkipStepAdamWConfig,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

from src.constants import (
    PAPER_EXPERTS,
    PAPER_LAYERS,
    PAPER_MICROBATCH_SEQUENCES,
    PAPER_ROUTINGS,
    PAPER_SEQUENCE_LENGTH,
    PAPER_TOP_K,
)
from src.routers.centroid import PaperCentroidConfig
from src.routers.seq_aux import PaperMoEConfig, PaperSeqAuxConfig
from src.types import Routing


def model_config(routing: Routing) -> TransformerConfig:
    if routing not in PAPER_ROUTINGS:
        raise ValueError(f'not a paper routing variant: {routing}')
    model = TransformerConfig.llama_like_moe(
        d_model=1024,
        vocab_size=TokenizerConfig.gpt_neox_olmo_dolma_v1_5().padded_vocab_size(),
        n_layers=PAPER_LAYERS,
        n_heads=8,
        num_experts=PAPER_EXPERTS,
        top_k=PAPER_TOP_K,
        expert_hidden_size=512,
        shared_expert_hidden_size=1024,
        dropless=True,
        reordered_norm=True,
        qk_norm=True,
        rope_theta=500_000,
        layer_norm_eps=1e-6,
        lb_loss_weight=0.01,
        z_loss_weight=0.001,
        init_std=0.02,
    )
    if not isinstance(model.block, TransformerBlockConfig) or model.block.feed_forward_moe is None:
        raise ValueError('expected a shared MoE block configuration')
    moe = model.block.feed_forward_moe
    if routing == 'aux_loss':
        return model
    moe.lb_loss_weight = None
    moe.z_loss_weight = None
    if routing == 'kmeans':
        moe.router = PaperCentroidConfig(top_k=moe.router.top_k, centroid_lr_lambda=10.0, bias_lr_lambda=1.0)
    else:
        moe.router.gating_function = MoERouterGatingFunction.sigmoid
        moe.router.bias_gamma = 1e-3
        moe.router.normalize_expert_weights = 1.0
        if routing == 'seq_aux':
            moe.router = PaperSeqAuxConfig(**moe.router.as_dict(recurse=False), seq_aux_loss_weight=1e-4)
            model.block.feed_forward_moe = PaperMoEConfig(**moe.as_dict(recurse=False))
    return model


def train_module_config(routing: Routing) -> TransformerTrainModuleConfig:
    """Recorded per-run optimizer and common schedule; no cluster paths."""
    if routing not in PAPER_ROUTINGS:
        raise ValueError(f'not a paper routing variant: {routing}')
    optimizer = SkipStepAdamWConfig if routing in ('aux_loss', 'loss_free') else AdamWConfig
    optim = optimizer(
        lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95),
        group_overrides=[OptimGroupOverride(params=['embeddings.weight'], opts={'weight_decay': 0.0})],
    )
    if isinstance(optim, AdamWConfig):
        optim.fused = True
    return TransformerTrainModuleConfig(
        rank_microbatch_size=PAPER_MICROBATCH_SEQUENCES * PAPER_SEQUENCE_LENGTH,
        max_sequence_length=PAPER_SEQUENCE_LENGTH,
        optim=optim,
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup=1000, alpha_f=0.1),
    )
