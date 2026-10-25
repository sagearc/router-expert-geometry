"""Two complete recipes behind the rebuttal's local 6.9B cosine table.

These differ in more than balancing. They are not public OLMoE converters.
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
    BALANCED_ROUTINGS,
    GEOMETRY_SEQUENCE_LENGTH,
)
from src.types import BalancedRouting


def model_config(recipe: BalancedRouting) -> TransformerConfig:
    if recipe not in BALANCED_ROUTINGS:
        raise ValueError('only the two retained 12k recipes are supported')
    model = TransformerConfig.olmoe_1B_7B(
        vocab_size=TokenizerConfig.gpt_neox_olmo_dolma_v1_5().padded_vocab_size())
    if not isinstance(model.block, TransformerBlockConfig) or model.block.feed_forward_moe is None:
        raise ValueError('expected a shared MoE block configuration')
    if recipe == 'loss_free':
        moe = model.block.feed_forward_moe
        moe.router.gating_function = MoERouterGatingFunction.sigmoid
        moe.router.normalize_expert_weights = 1.0
        moe.router.bias_gamma = 1e-3
        moe.lb_loss_weight = None
        moe.z_loss_weight = None
    return model


def train_module_config(recipe: BalancedRouting, microbatch_sequences: int) -> TransformerTrainModuleConfig:
    if recipe not in BALANCED_ROUTINGS or microbatch_sequences < 1:
        raise ValueError('requires a retained recipe and positive microbatch')
    optimizer = AdamWConfig if recipe == 'aux_loss' else SkipStepAdamWConfig
    optim = optimizer(
        lr=4e-4 if recipe == 'aux_loss' else 2e-4,
        weight_decay=0.1, betas=(0.9, 0.95),
        group_overrides=[OptimGroupOverride(params=['embeddings.weight'], opts={'weight_decay': 0.0})],
    )
    if isinstance(optim, AdamWConfig):
        optim.fused = True
    return TransformerTrainModuleConfig(
        rank_microbatch_size=microbatch_sequences * GEOMETRY_SEQUENCE_LENGTH,
        max_sequence_length=GEOMETRY_SEQUENCE_LENGTH,
        optim=optim,
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        z_loss_multiplier=1e-5, max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup=2000 if recipe == 'aux_loss' else 6400, alpha_f=0.1),
    )
