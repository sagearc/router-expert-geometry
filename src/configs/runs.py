"""Trainer configurations for the paper's 1B routers and the local 6.9B recipes.

The ``train-1b`` and ``train-geometry-7b`` commands only parse arguments;
everything that decides what a run is lives here.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from olmo_core.config import Config
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    LMEvaluatorCallbackConfig,
    WandBCallback,
)
from olmo_core.train.train_module import TransformerTrainModuleConfig
from olmo_core.utils import seed_all

from src.configs.geometry_7b import model_config as geometry_model_config
from src.configs.geometry_7b import (
    train_module_config as geometry_train_module_config,
)
from src.configs.paper_1b import model_config, train_module_config
from src.constants import (
    GEOMETRY_GLOBAL_BATCH_SIZE,
    GEOMETRY_SEQUENCE_LENGTH,
    PAPER_GLOBAL_BATCH_SIZE,
    PAPER_SEQUENCE_LENGTH,
    TRAINING_SEED,
    TRAINING_TOKENS,
)
from src.routers.training import RouterLearningRateCallback
from src.types import BalancedRouting, Routing

# Preserve the retained training shard order. The files are raw token arrays.
TRAINING_SHARDS = [
    'algebraic-stack/part-0-00000.npy',
    'dclm/part-00-00000.npy',
    'dclm/part-01-00000.npy',
    'dclm/part-02-00000.npy',
    'dclm/part-02-00001.npy',
    'dclm/part-03-00000.npy',
    'dclm/part-04-00000.npy',
    'dclm/part-05-00000.npy',
    'dclm/part-05-00001.npy',
    'dclm/part-06-00000.npy',
    'dclm/part-07-00000.npy',
    'dclm/part-08-00000.npy',
    'dclm/part-09-00000.npy',
    'dclm/part-10-00000.npy',
    'dclm/part-10-00001.npy',
    'dclm/part-11-00000.npy',
    'dclm/part-12-00000.npy',
    'dclm/part-13-00000.npy',
    'dclm/part-13-00001.npy',
    'dclm/part-14-00000.npy',
    'dclm/part-15-00000.npy',
    'open-web-math/part-0-00000.npy',
    'pes2o/part-0-00000.npy',
    'pes2o/part-0-00001.npy',
    'starcoder/part-0-00000.npy',
    'starcoder/part-1-00000.npy',
    'starcoder/part-2-00000.npy',
    'starcoder/part-3-00000.npy',
    'starcoder/part-4-00000.npy',
    'starcoder/part-5-00000.npy',
    'starcoder/part-6-00000.npy',
    'starcoder/part-7-00000.npy',
    'starcoder/part-8-00000.npy',
    'wiki/part-0-00000.npy',
    'wiki/part-0-00001.npy',
]


class RunArgs:
    """Fields shared by the 1B and 6.9B entry points; argparse fills a subclass."""

    data_root: Path
    eval_root: Path
    save_folder: Path
    cache_dir: Path
    microbatch_sequences: int
    stop_after: int | None
    wandb_project: str | None
    wandb_entity: str | None
    print_config: bool


def add_run_arguments(parser: argparse.ArgumentParser, microbatch_sequences: int) -> None:
    """The flags both training commands share: paths, memory, stopping and logging."""
    parser.add_argument('--data-root', type=Path, required=True, help='root of the 35 tokenized training shards')
    parser.add_argument('--eval-root', type=Path, required=True, help='base directory for v3_small_ppl_validation')
    parser.add_argument('--save-folder', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--microbatch-sequences', type=int, default=microbatch_sequences,
                        help='sequences per rank microbatch; changes memory use, not the global batch')
    parser.add_argument('--stop-after', type=int, help='hard stop at this step without shortening the 100B-token schedule')
    parser.add_argument('--wandb-project')
    parser.add_argument('--wandb-entity')
    parser.add_argument('--print-config', action='store_true', help='print the complete configuration and exit')


def parse_run_args[A: RunArgs](parser: argparse.ArgumentParser, namespace: A) -> A:
    args = parser.parse_args(namespace=namespace)
    if args.microbatch_sequences < 1 or (args.stop_after is not None and args.stop_after < 1):
        parser.error('microbatch and stopping step must be positive')
    return args


class TrainingArgs(RunArgs):
    routing: Routing


class GeometryArgs(RunArgs):
    recipe: BalancedRouting


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyFSLDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = TRAINING_SEED


def build_config(args: TrainingArgs) -> ExperimentConfig:
    tokenizer = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
    module = train_module_config(args.routing)
    module.rank_microbatch_size = args.microbatch_sequences * PAPER_SEQUENCE_LENGTH
    trainer = TrainerConfig(
        save_folder=str(args.save_folder), metrics_collect_interval=10,
        cancel_check_interval=1, max_duration=Duration.tokens(TRAINING_TOKENS),
        hard_stop=Duration.steps(args.stop_after) if args.stop_after is not None else None,
    ).with_callback('checkpointer', CheckpointerCallback(
        save_interval=1000, ephemeral_save_interval=80, pre_train_checkpoint=False, save_async=True,
    )).with_callback('config_saver', ConfigSaverCallback()).with_callback(
        'lm_evaluator', LMEvaluatorCallbackConfig(
            eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                DataMix.v3_small_ppl_validation, mix_base_dir=str(args.eval_root),
                sequence_length=PAPER_SEQUENCE_LENGTH, tokenizer=tokenizer, work_dir=str(args.cache_dir),
            ), eval_interval=250,
        ),
    )
    if args.routing == 'kmeans':
        # Centroid and bias steps scale with the learning rate of the previous optimizer step.
        trainer = trainer.with_callback('router_lr', RouterLearningRateCallback())
    if args.wandb_project:
        trainer = trainer.with_callback('wandb', WandBCallback(
            name=args.save_folder.name, project=args.wandb_project,
            entity=args.wandb_entity, group=args.routing,
        ))
    return ExperimentConfig(
        model=model_config(args.routing),
        dataset=NumpyFSLDatasetConfig(
            paths=[str(args.data_root / shard) for shard in TRAINING_SHARDS],
            sequence_length=PAPER_SEQUENCE_LENGTH, max_target_sequence_length=4096,
            tokenizer=tokenizer, work_dir=str(args.cache_dir),
        ),
        data_loader=NumpyDataLoaderConfig(global_batch_size=PAPER_GLOBAL_BATCH_SIZE, seed=TRAINING_SEED, num_workers=4),
        train_module=module, trainer=trainer,
    )


def build_7b_config(args: GeometryArgs) -> ExperimentConfig:
    aux = args.recipe == 'aux_loss'
    tokenizer = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
    trainer = TrainerConfig(
        save_folder=str(args.save_folder), metrics_collect_interval=10,
        cancel_check_interval=1, max_duration=Duration.tokens(TRAINING_TOKENS),
        hard_stop=Duration.steps(args.stop_after) if args.stop_after else None,
    ).with_callback('checkpointer', CheckpointerCallback(
        save_interval=10000 if aux else 16000,
        ephemeral_save_interval=1000 if aux else 200,
        pre_train_checkpoint=False, save_async=True,
    )).with_callback('config_saver', ConfigSaverCallback()).with_callback(
        'lm_evaluator', LMEvaluatorCallbackConfig(
            eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                DataMix.v3_small_ppl_validation, mix_base_dir=str(args.eval_root),
                sequence_length=GEOMETRY_SEQUENCE_LENGTH, tokenizer=tokenizer, work_dir=str(args.cache_dir),
            ), eval_interval=1000 if aux else 400,
        ),
    )
    if args.wandb_project:
        trainer = trainer.with_callback('wandb', WandBCallback(
            name=args.save_folder.name, project=args.wandb_project,
            entity=args.wandb_entity, group=args.recipe,
        ))
    return ExperimentConfig(
        model=geometry_model_config(args.recipe),
        dataset=NumpyFSLDatasetConfig(
            paths=[str(args.data_root / shard) for shard in TRAINING_SHARDS],
            sequence_length=GEOMETRY_SEQUENCE_LENGTH, max_target_sequence_length=GEOMETRY_SEQUENCE_LENGTH,
            tokenizer=tokenizer, work_dir=str(args.cache_dir),
        ),
        data_loader=NumpyDataLoaderConfig(global_batch_size=GEOMETRY_GLOBAL_BATCH_SIZE, seed=TRAINING_SEED, num_workers=4),
        train_module=geometry_train_module_config(args.recipe, args.microbatch_sequences), trainer=trainer,
    )


def run(config: ExperimentConfig) -> None:
    prepare_training_environment()
    try:
        seed_all(config.init_seed)
        model = config.model.build(init_device='meta')
        module = config.train_module.build(model)
        dataset = config.dataset.build()
        loader = config.data_loader.build(dataset, dp_process_group=module.dp_process_group)
        trainer = config.trainer.build(module, loader)
        saver = trainer.callbacks['config_saver']
        if not isinstance(saver, ConfigSaverCallback):
            raise TypeError('expected the configured ConfigSaverCallback')
        saver.config = config.as_config_dict()
        trainer.fit()
    finally:
        teardown_training_environment()


def print_or_run(config: ExperimentConfig, print_config: bool) -> None:
    if print_config:
        print(json.dumps(config.as_config_dict(), indent=2))
    else:
        run(config)
