"""Capture a paper checkpoint on fixed, already-tokenized Wikipedia prompts.

The token NPZ must contain right-padded integer input_ids and integer lengths.
Use the same token file for every checkpoint; no sampling occurs here.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import torch
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.transformer.model import MoETransformer

from src.capture.last_token import capture_last_tokens
from src.configs.paper_1b import model_config
from src.constants import (
    OLMO_CORE_REVISION,
    PAPER_CAPTURE_BATCH_SIZE,
    PAPER_ROUTINGS,
)
from src.types import CaptureMetric, FloatArray, IntArray, Routing


class Args:
    checkpoint: str
    routing: Routing
    tokens: Path
    out: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='OLMo-core model_and_optim directory')
    parser.add_argument('--routing', required=True, choices=PAPER_ROUTINGS)
    parser.add_argument('--tokens', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(namespace=Args())
    if args.out.exists():
        parser.error('output already exists')
    if not torch.cuda.is_available():
        parser.error('dropless inference requires GPU kernels')
    with np.load(args.tokens, allow_pickle=False) as data:
        ids, lengths = data['input_ids'], data['lengths']
    if ids.ndim != 2 or lengths.shape != ids.shape[:1] or not len(ids):
        parser.error('invalid input_ids/lengths dimensions or empty token set')
    if not np.issubdtype(ids.dtype, np.integer) or not np.issubdtype(lengths.dtype, np.integer):
        parser.error('input_ids and lengths must be integers')
    if np.any(lengths < 1) or np.any(lengths > ids.shape[1]):
        parser.error('invalid token lengths')
    config = model_config(args.routing)
    if np.any(ids < 0) or np.any(ids >= config.vocab_size):
        parser.error('token ID outside model vocabulary')

    model = cast(MoETransformer, config.build(init_device='cpu'))
    load_model_and_optim_state(args.checkpoint, model, strict=True)
    model.to(device='cuda', dtype=torch.bfloat16).eval()
    chunks: dict[CaptureMetric, list[FloatArray | IntArray]] = {}
    for start in range(0, len(ids), PAPER_CAPTURE_BATCH_SIZE):
        batch = slice(start, start + PAPER_CAPTURE_BATCH_SIZE)
        # Retained tokenize_and_batch pads each batch, not the entire title set.
        width = int(lengths[batch].max())
        values = capture_last_tokens(
            model, torch.as_tensor(ids[batch, :width], device='cuda', dtype=torch.long),
            torch.as_tensor(lengths[batch], device='cuda', dtype=torch.long),
        )
        for key, value in values.items():
            chunks.setdefault(key, []).append(value.numpy())
        print(f'captured {min(start + PAPER_CAPTURE_BATCH_SIZE, len(ids))}/{len(ids)} prompts', flush=True)

    # Preserve the original capture's FP16 storage quantization before analysis.
    with np.errstate(over='raise', invalid='raise'):
        arrays = {key: np.concatenate(values).astype(
            np.int64 if key == 'selected_experts' else np.float16
        ) for key, values in chunks.items()}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('capture contains nonfinite values; no artifact written')
    metadata = {
        'checkpoint': args.checkpoint, 'routing': args.routing,
        'tokens_sha256': hashlib.sha256(args.tokens.read_bytes()).hexdigest(),
        'token_scope': 'last_nonpadding_token', 'forward_dtype': 'bfloat16',
        'batch_size': PAPER_CAPTURE_BATCH_SIZE, 'padding': 'right_to_batch_max_length',
        'projection_dtype': 'float32', 'storage_dtype': 'float16',
        'olmo_core_commit': OLMO_CORE_REVISION,
    }
    with args.out.open('xb') as stream:
        np.savez_compressed(
            stream,
            router_scores=arrays['router_scores'],
            selected_experts=arrays['selected_experts'],
            w1_top10_l2=arrays['w1_top10_l2'],
            w1_silu_mean=arrays['w1_silu_mean'],
            w1_l2=arrays['w1_l2'],
            w3_l2=arrays['w3_l2'],
            metadata_json=json.dumps(metadata),
        )
