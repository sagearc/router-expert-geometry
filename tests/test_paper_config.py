"""Run with the models extra; configuration construction allocates no model."""

import pytest

from src.constants import (
    PAPER_GLOBAL_BATCH_SIZE,
)

torch = pytest.importorskip('torch')

pytest.importorskip('olmo_core')

from olmo_core.optim import AdamWConfig, SkipStepAdamWConfig

from src.configs.paper_1b import (
    model_config,
    train_module_config,
)
from src.routers.centroid import PaperCentroidConfig
from src.routers.seq_aux import PaperMoEConfig, PaperSeqAuxConfig


def test_four_paper_routing_recipes():
    configs = {name: model_config(name) for name in
               ('aux_loss', 'loss_free', 'seq_aux', 'kmeans')}
    for model in configs.values():
        assert model.d_model == 1024 and model.n_layers == 9
        assert model.block.name == 'moe_reordered_norm'
        moe = model.block.feed_forward_moe
        assert (moe.num_experts, moe.hidden_size, moe.router.top_k) == (64, 512, 6)
        assert moe.shared_mlp.hidden_size == 1024
    aux = configs['aux_loss'].block.feed_forward_moe
    assert (aux.lb_loss_weight, aux.z_loss_weight) == (0.01, 0.001)
    assert aux.router.gating_function == 'softmax'
    for name in ('loss_free', 'seq_aux'):
        moe = configs[name].block.feed_forward_moe
        assert moe.lb_loss_weight is None and moe.z_loss_weight is None
        assert moe.router.gating_function == 'sigmoid'
        assert moe.router.bias_gamma == 1e-3
        assert moe.router.normalize_expert_weights == 1.0
    seq_aux = configs['seq_aux'].block.feed_forward_moe
    assert isinstance(seq_aux, PaperMoEConfig) and isinstance(seq_aux.router, PaperSeqAuxConfig)
    assert seq_aux.router.seq_aux_loss_weight == 1e-4 and seq_aux.scale_loss_by_num_layers
    km = configs['kmeans'].block.feed_forward_moe.router
    assert isinstance(km, PaperCentroidConfig)
    assert (km.centroid_lr_lambda, km.bias_lr_lambda) == (10.0, 1.0)


def test_recorded_training_recipe_keeps_optimizer_and_loss_terms_distinct():
    config = train_module_config('loss_free')
    assert isinstance(config.optim, SkipStepAdamWConfig)
    assert config.optim.lr == 1e-3
    assert config.optim.betas == (0.9, 0.95)
    assert config.optim.weight_decay == 0.1
    assert config.optim.group_overrides[0].params == ['embeddings.weight']
    assert config.optim.group_overrides[0].opts == {'weight_decay': 0.0}
    assert config.scheduler.warmup == 1000 and config.scheduler.alpha_f == 0.1
    assert config.z_loss_multiplier == 1e-5  # LM z-loss, not router z-loss.
    assert config.max_grad_norm == 1.0
    assert PAPER_GLOBAL_BATCH_SIZE == 1152 * 2048
    assert config.rank_microbatch_size == 96 * 2048
    for routing in ('seq_aux', 'kmeans'):
        assert isinstance(train_module_config(routing).optim, AdamWConfig)


def test_all_paper_optimizer_configs_build_with_embedding_override():
    model = torch.nn.Module()
    model.embeddings = torch.nn.Embedding(4, 2)
    model.projection = torch.nn.Linear(2, 2, bias=False)
    for routing in ('aux_loss', 'loss_free', 'seq_aux', 'kmeans'):
        config = train_module_config(routing).optim
        if routing in ('aux_loss', 'loss_free'):
            # Both retained step-22k specialist checkpoint configs record these.
            assert config.foreach is True
            assert config.step_increment_bugfix is True
            assert config.rolling_interval_length == 128
            assert config.sigma_factor == 6
            assert config.eps == 1e-8
        optimizer = config.build(model)
        groups = {id(p): group for group in optimizer.param_groups for p in group['params']}
        assert groups[id(model.embeddings.weight)]['weight_decay'] == 0
        assert groups[id(model.projection.weight)]['weight_decay'] == 0.1
        if routing in ('seq_aux', 'kmeans'):
            assert config.fused is True and config.foreach is None
