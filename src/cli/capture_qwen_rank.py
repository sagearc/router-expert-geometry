"""Regenerate the retained Qwen3-30B-A3B-Base all-token rank-response curve."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import Qwen3MoeForCausalLM

from src.constants import (
    PUBLIC_MAX_TOKENS,
    PUBLIC_RESPONSE_METRICS,
    PUBLIC_TITLE_COUNT,
    QWEN_EXPERT_CHUNK,
    QWEN_EXPERTS,
    QWEN_LAYERS,
    QWEN_MODEL_ID,
    QWEN_REVISION,
    QWEN_TOP_K,
)
from src.hf.qwen import rank_responses


class Args:
    tokens: Path
    out: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', type=Path, required=True, help='fixed 256-title NPZ, input_ids and lengths')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(namespace=Args())
    if args.out.exists() or not torch.cuda.is_available():
        parser.error('requires CUDA and a new output')
    with np.load(args.tokens, allow_pickle=False) as data:
        ids, lengths = data['input_ids'], data['lengths']
    if (ids.ndim != 2 or ids.shape[0] != PUBLIC_TITLE_COUNT or ids.shape[1] > PUBLIC_MAX_TOKENS or lengths.shape != (PUBLIC_TITLE_COUNT,)
            or not np.issubdtype(ids.dtype, np.integer) or not np.issubdtype(lengths.dtype, np.integer)
            or (lengths < 1).any() or (lengths > ids.shape[1]).any()):
        raise ValueError('expected 256 nonempty right-padded titles, at most 64 tokens each')
    model = Qwen3MoeForCausalLM.from_pretrained(
        QWEN_MODEL_ID, revision=QWEN_REVISION,
        dtype=torch.bfloat16,
    ).to('cuda').eval()  # type: ignore[arg-type]  # Transformers 5.1 @wraps loses the bound .to() signature.
    if (model.config.num_hidden_layers, model.config.num_experts, model.config.num_experts_per_tok) != (QWEN_LAYERS, QWEN_EXPERTS, QWEN_TOP_K):
        raise ValueError('unexpected Qwen architecture')
    if (ids < 0).any() or (ids >= model.config.vocab_size).any():
        raise ValueError('token ID outside vocabulary')
    total, count, changes = np.zeros((2, QWEN_EXPERTS)), 0, 0
    for start in range(PUBLIC_TITLE_COUNT):
        # The retained rank-v2 run used batch size one, with no batch padding.
        tokens = torch.as_tensor(ids[start:start + 1, :lengths[start]], device='cuda', dtype=torch.long)
        mask = torch.ones_like(tokens, dtype=torch.bool)
        sums, observations, differences = rank_responses(model, tokens, mask)
        total += sums
        count += observations
        changes += differences
        print(f'captured {start + 1}/256 titles', flush=True)
    metadata = {'model': QWEN_MODEL_ID, 'revision': QWEN_REVISION,
                'torch': torch.__version__, 'transformers': transformers.__version__,
                'rank_rule': 'stable_descending_softmax_of_native_probabilities',
                'native_rank_entry_differences': changes, 'token_layer_observations': count,
                'token_scope': 'all_nonpadding', 'batch_size': 1,
                'dtype': 'bfloat16', 'projection_dtype': 'float32', 'expert_chunk_size': QWEN_EXPERT_CHUNK,
                'attention_implementation': model.config._attn_implementation,
                'experts_implementation': model.config._experts_implementation,
                'tokens_sha256': hashlib.sha256(args.tokens.read_bytes()).hexdigest()}
    with args.out.open('xb') as stream:
        np.savez_compressed(stream, mean=total / count, expert_rank=np.arange(1, QWEN_EXPERTS + 1),
                            top_k=QWEN_TOP_K, metric_names=np.array(PUBLIC_RESPONSE_METRICS),
                            metadata_json=json.dumps(metadata))
