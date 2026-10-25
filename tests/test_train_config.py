from types import SimpleNamespace

import pytest

pytest.importorskip('olmo_core')

from src.configs.runs import build_config


@pytest.mark.parametrize('routing', ['aux_loss', 'loss_free', 'seq_aux', 'kmeans'])
def test_short_run_preserves_full_schedule_and_global_batch(routing, tmp_path):
    args = SimpleNamespace(
        routing=routing, data_root=tmp_path / 'tokenized', eval_root=tmp_path,
        save_folder=tmp_path / routing, cache_dir=tmp_path / 'cache',
        microbatch_sequences=2, stop_after=2, wandb_project=None, wandb_entity=None,
    )
    config = build_config(args)
    assert config.trainer.max_duration.value == 100_000_000_000
    assert config.trainer.max_duration.unit == 'tokens'
    assert config.trainer.hard_stop.value == 2
    assert config.trainer.hard_stop.unit == 'steps'
    assert config.train_module.scheduler.warmup == 1000
    assert config.data_loader.global_batch_size == 1152 * 2048
    assert config.train_module.rank_microbatch_size == 2 * 2048
    assert config.dataset.max_target_sequence_length == 4096
    assert len(config.dataset.paths) == 35
    assert len(set(config.dataset.paths)) == 35
    assert config.init_seed == config.data_loader.seed == 6198
    assert config.trainer.callbacks['lm_evaluator'].eval_interval == 250
    assert ('router_lr' in config.trainer.callbacks) == (routing == 'kmeans')
    # Saving the concrete configuration must work for the centroid subclass too.
    assert config.as_config_dict()['model']['n_layers'] == 9
