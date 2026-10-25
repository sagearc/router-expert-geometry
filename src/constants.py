"""Fixed model identities and experiment protocols, shared by writers and readers."""

from pathlib import Path
from typing import Final, Literal

from src.configs.public_olmoe import OLMOE_CHECKPOINT_REVISIONS

PAPER_ROUTINGS: Final = ('aux_loss', 'loss_free', 'seq_aux', 'kmeans')
BALANCED_ROUTINGS: Final = ('aux_loss', 'loss_free')
ERROR_ROUTINGS: Final = ('aux_loss', 'loss_free', 'kmeans')

# Native paper models. The local 6.9B recipes have a different sequence length.
OLMO_CORE_REVISION: Final = '60930ef14c8e4fb93eec071d44e104324d5c1c30'
TRAINING_SEED: Final = 6198
PAPER_LAYERS: Final = 9
PAPER_EXPERTS: Final = 64
PAPER_TOP_K: Final = 6
PAPER_VOCAB_SIZE: Final = 50304
PAPER_SEQUENCE_LENGTH: Final = 2048
PAPER_GLOBAL_BATCH_SIZE: Final = 1152 * PAPER_SEQUENCE_LENGTH
PAPER_MICROBATCH_SEQUENCES: Final = 96
PAPER_CAPTURE_BATCH_SIZE: Final = 32
PAPER_EVAL_BATCH_SIZE: Final = 8
GEOMETRY_SEQUENCE_LENGTH: Final = 4096
GEOMETRY_GLOBAL_BATCH_SIZE: Final = 1_310_720
GEOMETRY_MICROBATCH_SEQUENCES: Final = 16
TRAINING_TOKENS: Final = 100_000_000_000

# Public checkpoints and the retained projection/kernel shapes.
OLMOE_MODEL_ID: Final = 'allenai/OLMoE-1B-7B-0924'
# Title preparation uses the retained 5k checkpoint's tokenizer snapshot.
OLMOE_TOKENIZER_REVISION: Final = OLMOE_CHECKPOINT_REVISIONS[5000]
OLMOE_LAYERS: Final = 16
OLMOE_EXPERTS: Final = 64
OLMOE_TOP_K: Final = 8
OLMOE_HIDDEN_SIZE: Final = 2048
OLMOE_EXPERT_CHUNK: Final = 8
OLMOE_CAPTURE_BATCH_SIZE: Final = 8
QWEN_MODEL_ID: Final = 'Qwen/Qwen3-30B-A3B-Base'
QWEN_REVISION: Final = '1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9'
QWEN_LAYERS: Final = 48
QWEN_EXPERTS: Final = 128
QWEN_TOP_K: Final = 8
QWEN_EXPERT_CHUNK: Final = 16
DSV4_MODEL_ID: Final = 'deepseek-ai/DeepSeek-V4-Flash'
DSV4_REVISION: Final = '60d8d70770c6776ff598c94bb586a859a38244f1'
DSV4_LAYERS: Final = 43
DSV4_EXPERTS: Final = 256
DSV4_TOP_K: Final = 6
DSV4_HASH_LAYERS: Final = 3
DSV4_SCORE_LAYERS: Final = tuple(range(DSV4_HASH_LAYERS, DSV4_LAYERS))
DSV4_EXPERT_CHUNK: Final = 32
DSV4_VLLM_VERSION: Final = '0.25.0'

PUBLIC_RESPONSE_METRICS: Final = ('w1_l2', 'w3_l2')
DSV4_RESPONSE_METRICS: Final = ('w_gate_x_l2', 'w_up_x_l2')
PUBLIC_TITLE_COUNT: Final = 256
PUBLIC_MAX_TOKENS: Final = 64
PUBLIC_TITLES_PATH: Final = Path(__file__).resolve().parents[1] / 'data/public-titles256.json'
VALIDATION_DOMAINS: Final = (
    'c4_en', 'dolma_books', 'dolma_common-crawl', 'dolma_pes2o', 'dolma_reddit',
    'dolma_stack', 'dolma_wiki', 'ice', 'm2d2_s2orc', 'pile', 'wikitext_103',
)
SPECIALIST_DOMAINS: Final = (
    'dolma_wiki', 'dolma_pes2o', 'dolma_stack', 'dolma_common-crawl',
    'dolma_books', 'dolma_reddit',
)
SPECIALIST_CODE_DOMAIN: Final = 'dolma_stack'
SPECIALIST_SEQUENCE_COUNT: Final = 240
SPECIALIST_EXPERT_COUNTS: Final = (1, 4)
SPECIALIST_CONTROL_EXPERTS: Final = 4
SPECIALIST_CONTROL_SEQUENCES: Final = 64
SPECIALIST_CONTROL_SEED: Final = 0

FIGURE2_STEPS: Final = (0, 1000, 5000, 10000, 20000, 39750)
FIGURE2_TITLE_COUNT: Final = 1024
FIGURE2_SEED: Final = 0
FIGURE2_BOOTSTRAP_SEED: Final = 10_000
FIGURE2_BOOTSTRAP_DRAWS: Final = 30
FIGURE2_IDENTITY_DRAWS: Final = 20
FIGURE2_FINAL_IDENTITY_DRAWS: Final = 50
FIGURE2_SHUFFLE_DRAWS: Final = 20

# Causal writer and summary must agree on the full retained protocol.
CAUSAL_LAYERS: Final = tuple(range(OLMOE_LAYERS))
CAUSAL_INTERVENTIONS: Final[dict[str, tuple[Literal['projection', 'relative', 'rotation'], float]]] = {
    'projection-1': ('projection', 1.),
    'relative-0.05': ('relative', .05),
    'relative-0.1': ('relative', .1),
    'rotation-0.1': ('rotation', .1),
}
CAUSAL_CANDIDATES: Final = 32
CAUSAL_POWER_ITERATIONS: Final = 12
CAUSAL_CANDIDATE_SEED: Final = 12987
CAUSAL_BOOTSTRAP_SEED: Final = 19871
CAUSAL_BOOTSTRAP_DRAWS: Final = 10_000
CAUSAL_PREDICTIONS: Final = 512
CAUSAL_SEQUENCE_LENGTH: Final = CAUSAL_PREDICTIONS + 1
CAUSAL_CALIBRATION_SEQUENCES: Final = 4
CAUSAL_SEQUENCES_PER_DOMAIN: Final = 16
CAUSAL_SOURCE_SEQUENCES: Final = CAUSAL_CALIBRATION_SEQUENCES + CAUSAL_SEQUENCES_PER_DOMAIN
CAUSAL_SAMPLE_COUNT: Final = len(VALIDATION_DOMAINS) * CAUSAL_SEQUENCES_PER_DOMAIN
ERROR_BOOTSTRAP_DRAWS: Final = 1000
ERROR_BOOTSTRAP_SEED: Final = 0

CI95_QUANTILES: Final = (.025, .975)
CI95_PERCENTILES: Final = tuple(100 * q for q in CI95_QUANTILES)
