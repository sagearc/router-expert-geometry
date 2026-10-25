from types import SimpleNamespace

import pytest

from src.constants import (
    GEOMETRY_GLOBAL_BATCH_SIZE,
)

pytest.importorskip('olmo_core')

from olmo_core.optim import AdamWConfig, SkipStepAdamWConfig

from src.configs.geometry_7b import (
    model_config,
    train_module_config,
)
from src.configs.runs import build_7b_config


def test_retained_7b_recipes_preserve_architecture_and_optimization_differences():
    aux, loss_free = model_config('aux_loss'), model_config('loss_free')
    for config in (aux, loss_free):
        meta_model = config.build(init_device='meta')
        assert sum(p.numel() for p in meta_model.parameters()) == 6_919_161_856
        assert config.d_model == 2048 and config.n_layers == 16
        assert config.block.name == 'moe_reordered_norm'
        moe = config.block.feed_forward_moe
        assert moe.num_experts == 64 and moe.router.top_k == 8 and moe.hidden_size == 1024
    assert aux.block.feed_forward_moe.lb_loss_weight == 0.01
    assert aux.block.feed_forward_moe.z_loss_weight == 0.001
    assert loss_free.block.feed_forward_moe.lb_loss_weight is None
    assert loss_free.block.feed_forward_moe.router.gating_function == 'sigmoid'
    assert loss_free.block.feed_forward_moe.router.normalize_expert_weights == 1.0
    assert loss_free.block.feed_forward_moe.router.bias_gamma == 1e-3
    a, b = train_module_config('aux_loss', 16), train_module_config('loss_free', 16)
    assert type(a.optim) is AdamWConfig and type(b.optim) is SkipStepAdamWConfig
    assert (a.optim.lr, b.optim.lr) == (4e-4, 2e-4)
    assert (a.scheduler.warmup, b.scheduler.warmup) == (2000, 6400)
    assert a.scheduler.alpha_f == b.scheduler.alpha_f == 0.1
    assert GEOMETRY_GLOBAL_BATCH_SIZE == 1_310_720
    assert b.rank_microbatch_size == 65536


@pytest.mark.parametrize('recipe', ['aux_loss', 'loss_free'])
def test_7b_training_entry_preserves_full_schedule_and_recorded_intervals(recipe, tmp_path):
    config = build_7b_config(SimpleNamespace(
        recipe=recipe, data_root=tmp_path, eval_root=tmp_path,
        save_folder=tmp_path / recipe, cache_dir=tmp_path / 'cache',
        stop_after=2, microbatch_sequences=16, wandb_project=None, wandb_entity=None,
    ))
    assert config.trainer.max_duration.value == 100_000_000_000
    assert config.trainer.hard_stop.value == 2
    assert config.data_loader.global_batch_size == GEOMETRY_GLOBAL_BATCH_SIZE
    assert config.dataset.sequence_length == config.dataset.max_target_sequence_length == 4096
    assert len(config.dataset.paths) == 35
    aux = recipe == 'aux_loss'
    checkpointer = config.trainer.callbacks['checkpointer']
    assert checkpointer.save_interval == (10000 if aux else 16000)
    assert checkpointer.ephemeral_save_interval == (1000 if aux else 200)
    assert config.trainer.callbacks['lm_evaluator'].eval_interval == (1000 if aux else 400)
    assert config.as_config_dict()['model']['n_layers'] == 16
