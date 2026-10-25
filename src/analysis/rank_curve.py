"""Read retained Qwen and DeepSeek-V4 rank captures into one curve schema."""

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np

from src.artifacts import RankCurve, RankMetadata
from src.constants import (
    DSV4_EXPERTS,
    DSV4_HASH_LAYERS,
    DSV4_MODEL_ID,
    DSV4_RESPONSE_METRICS,
    DSV4_SCORE_LAYERS,
    DSV4_TOP_K,
    PUBLIC_RESPONSE_METRICS,
    QWEN_EXPERTS,
    QWEN_MODEL_ID,
    QWEN_TOP_K,
)


def rank_curve(path: Path) -> RankCurve:
    with np.load(path, allow_pickle=False) as data:
        metadata: RankMetadata = {}
        if 'metadata_json' in data:
            metadata = cast(RankMetadata, json.loads(str(data['metadata_json'])))
            model_id = metadata.get('model')
            if model_id not in (QWEN_MODEL_ID, DSV4_MODEL_ID):
                raise ValueError('unsupported capture model identity')
        else:
            model_id = QWEN_MODEL_ID
            metadata = {'source_format': 'retained_qwen_compact_rank_summary'}
        ranks = data['expert_rank']
        if model_id == QWEN_MODEL_ID:
            if (not np.array_equal(data['metric_names'], PUBLIC_RESPONSE_METRICS)
                    or int(data['top_k']) != QWEN_TOP_K or data['mean'].shape != (2, QWEN_EXPERTS)):
                raise ValueError('expected retained Qwen3-30B-A3B-Base rank summary')
            mean = data['mean'].astype(np.float64)
            top_k, experts = QWEN_TOP_K, QWEN_EXPERTS
            model_id = QWEN_MODEL_ID
        else:
            if (metadata.get('model') != DSV4_MODEL_ID
                    or metadata.get('top_k') != DSV4_TOP_K or metadata.get('token_scope') != 'all'
                    or metadata.get('excluded_hash_moe_layers') != DSV4_HASH_LAYERS
                    or not np.array_equal(data['model_layer_indices'], np.asarray(DSV4_SCORE_LAYERS))
                    or metadata.get('response_metrics') != list(DSV4_RESPONSE_METRICS)):
                raise ValueError('expected retained DeepSeek-V4 score-routed layer capture')
            values, counts = data['response_mean'], data['rank_count']
            if (values.shape != (len(DSV4_SCORE_LAYERS), DSV4_EXPERTS, 2) or counts.shape != (len(DSV4_SCORE_LAYERS), DSV4_EXPERTS)
                    or not np.issubdtype(counts.dtype, np.integer) or (counts < 1).any()
                    or not np.isfinite(values).all() or (values < 0).any()):
                raise ValueError('invalid DeepSeek response means or observation counts')
            # Every token/layer observation gets equal weight at its expert rank.
            mean = ((values.astype(np.float64) * counts[..., None]).sum(axis=0)
                    / counts.sum(axis=0)[:, None]).T
            top_k, experts = DSV4_TOP_K, DSV4_EXPERTS
            model_id = metadata['model']
    if (not np.array_equal(ranks, np.arange(1, experts + 1))
            or not np.isfinite(mean).all() or (mean < 0).any()):
        raise ValueError('invalid expert ranks or raw L2 responses')
    return {
        'model': model_id, 'expert_rank': ranks.tolist(), 'top_k': top_k,
        'metric_names': ['w_gate_h_l2', 'w_up_h_l2'], 'response_mean': mean.tolist(),
        'aggregation': 'mean_response_at_each_per_token_per_layer_expert_rank',
        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'capture_metadata': metadata,
    }
