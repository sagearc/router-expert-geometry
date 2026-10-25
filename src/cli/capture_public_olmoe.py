"""Capture public OLMoE checkpoints with native normalization and expert routing."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import OlmoeForCausalLM

from src.configs.public_olmoe import OLMOE_CHECKPOINT_REVISIONS
from src.constants import (
    OLMOE_CAPTURE_BATCH_SIZE,
    OLMOE_EXPERTS,
    OLMOE_LAYERS,
    OLMOE_MODEL_ID,
    OLMOE_TOP_K,
)
from src.hf.olmoe import capture_olmoe
from src.types import FloatArray, IntArray, PublicCaptureMetric


class Args:
    checkpoint_step: int
    tokens: Path
    out: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-step', type=int, choices=OLMOE_CHECKPOINT_REVISIONS, required=True,
                        help='retained experiment checkpoint; its commit is fixed in configuration')
    parser.add_argument('--tokens', required=True, type=Path, help='fixed NPZ with input_ids and lengths')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(namespace=Args())
    revision = OLMOE_CHECKPOINT_REVISIONS[args.checkpoint_step]
    if args.out.exists() or not torch.cuda.is_available():
        parser.error('requires a new output and CUDA')
    with np.load(args.tokens, allow_pickle=False) as data:
        ids, lengths = data['input_ids'], data['lengths']
    if ids.ndim != 2 or not len(ids) or lengths.shape != ids.shape[:1]:
        parser.error('expected input_ids (titles, sequence) and lengths (titles,)')
    if not np.issubdtype(ids.dtype, np.integer) or not np.issubdtype(lengths.dtype, np.integer):
        parser.error('token IDs and lengths must be integers')
    if (lengths < 1).any() or (lengths > ids.shape[1]).any():
        parser.error('invalid prompt lengths')
    model = OlmoeForCausalLM.from_pretrained(
        OLMOE_MODEL_ID, revision=revision, dtype=torch.bfloat16, attn_implementation='sdpa',
    ).to('cuda').eval()  # type: ignore[arg-type]  # Transformers 5.1 @wraps loses the bound .to() signature.
    config = model.config
    if (config.num_hidden_layers, config.num_experts, config.num_experts_per_tok) != (OLMOE_LAYERS, OLMOE_EXPERTS, OLMOE_TOP_K):
        parser.error('expected public OLMoE 16-layer / 64-expert / top-8 architecture')
    if (ids < 0).any() or (ids >= config.vocab_size).any():
        parser.error('token ID outside model vocabulary')
    chunks: dict[PublicCaptureMetric, list[FloatArray | IntArray]] = {}
    for start in range(0, len(ids), OLMOE_CAPTURE_BATCH_SIZE):
        batch = slice(start, start + OLMOE_CAPTURE_BATCH_SIZE)
        width = int(lengths[batch].max())
        tokens = torch.as_tensor(ids[batch, :width], device='cuda', dtype=torch.long)
        mask = torch.arange(width, device='cuda')[None, :] < torch.as_tensor(lengths[batch], device='cuda')[:, None]
        for key, value in capture_olmoe(model, tokens, mask).items():
            if not torch.isfinite(value).all():
                raise ValueError(f'nonfinite {key}; no artifact written')
            chunks.setdefault(key, []).append(value.numpy())
        print(f'captured {min(start + OLMOE_CAPTURE_BATCH_SIZE, len(ids))}/{len(ids)} titles', flush=True)
    metadata = {'model': OLMOE_MODEL_ID, 'revision': revision, 'transformers': transformers.__version__,
                'torch': torch.__version__, 'attention_implementation': config._attn_implementation,
                'experts_implementation': config._experts_implementation,
                'batch_size': OLMOE_CAPTURE_BATCH_SIZE, 'padding': 'right_to_batch_max_length',
                'token_file_width': int(ids.shape[1]),
                'token_scope': 'last_nonpadding_token', 'forward_dtype': 'bfloat16',
                'response_projection_dtype': 'float32', 'score_definition': 'native_linear_logits',
                'tokens_sha256': hashlib.sha256(args.tokens.read_bytes()).hexdigest()}
    with args.out.open('xb') as stream:
        np.savez_compressed(
            stream,
            router_scores=np.concatenate(chunks['router_scores']),
            selection_scores=np.concatenate(chunks['selection_scores']),
            selected_experts=np.concatenate(chunks['selected_experts']),
            w1_l2=np.concatenate(chunks['w1_l2']),
            w3_l2=np.concatenate(chunks['w3_l2']),
            metadata_json=json.dumps(metadata),
        )
