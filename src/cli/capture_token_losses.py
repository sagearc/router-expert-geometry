"""Capture the paper's per-token losses and boundary margins on fixed sequences.

Input NPZ: sequences (N, 2048), domain_idx (N,), domain_names (D,).
Use exactly the same input for K-means and loss-free checkpoints at step 21000.
Losses and margins share one output file so new captures cannot be mispaired.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.transformer.model import MoETransformer

from src.capture.token_losses import capture_token_losses
from src.configs.paper_1b import model_config
from src.constants import (
    ERROR_ROUTINGS,
    PAPER_EVAL_BATCH_SIZE,
    PAPER_SEQUENCE_LENGTH,
    PAPER_VOCAB_SIZE,
)
from src.sequences import domain_batches
from src.types import ErrorRouting, FloatArray, IntArray


class Args:
    checkpoint: str
    routing: ErrorRouting
    sequences: Path
    out: Path



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--routing', required=True, choices=ERROR_ROUTINGS)
    parser.add_argument('--sequences', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(namespace=Args())
    if args.out.exists() or not torch.cuda.is_available():
        parser.error('requires a new output path and CUDA')
    with np.load(args.sequences, allow_pickle=False) as data:
        sequences, domains, names = data['sequences'], data['domain_idx'], data['domain_names']
    if sequences.ndim != 2 or not len(sequences) or sequences.shape[1] != PAPER_SEQUENCE_LENGTH:
        parser.error('expected nonempty unpadded (N, 2048) sequences')
    if not np.issubdtype(sequences.dtype, np.integer) or (sequences < 0).any() or (sequences >= PAPER_VOCAB_SIZE).any():
        parser.error('invalid token IDs')
    if domains.shape != (len(sequences),) or not np.issubdtype(domains.dtype, np.integer):
        parser.error('expected one integer domain index per sequence')
    if names.ndim != 1 or names.dtype.kind != 'U' or (domains < 0).any() or (domains >= len(names)).any():
        parser.error('invalid domain names or indices')
    model = cast(MoETransformer, model_config(args.routing).build(init_device='cpu'))
    load_model_and_optim_state(args.checkpoint, model, strict=True)
    model.to(device='cuda', dtype=torch.float32).eval()
    chunks: dict[Literal['nll', 'targets', 'margin6'], list[FloatArray | IntArray]] = {}
    for rows in domain_batches(domains, PAPER_EVAL_BATCH_SIZE):
        batch = torch.as_tensor(sequences[rows], device='cuda', dtype=torch.long)
        for key, values in capture_token_losses(model, batch).items():
            # Retain the historical FP16 storage, while doing inference in FP32.
            with np.errstate(over='raise', invalid='raise'):
                stored: FloatArray | IntArray
                if key == 'targets':
                    stored = values.numpy().astype(np.uint16)
                else:
                    stored = values.numpy().astype(np.float16)
            if not np.isfinite(stored).all():
                raise ValueError(f'nonfinite {key} at rows {rows.start}:{rows.stop}; capture aborted')
            chunks.setdefault(key, []).append(stored)
        print(f'captured {rows.stop}/{len(sequences)} sequences', flush=True)
    arrays = {key: np.concatenate(values, axis=1 if key.startswith('margin') else 0)
              for key, values in chunks.items()}
    metadata = {'checkpoint': args.checkpoint, 'routing': args.routing,
                'sequences_sha256': hashlib.sha256(args.sequences.read_bytes()).hexdigest(),
                'forward_dtype': 'float32', 'storage_dtype': 'float16',
                'batch_size': PAPER_EVAL_BATCH_SIZE, 'batching': 'contiguous_domains_separately',
                'alignment': 'routing_at_t_predicts_target_at_t_plus_1'}
    with args.out.open('xb') as stream:
        np.savez_compressed(
            stream,
            nll=arrays['nll'],
            targets=arrays['targets'],
            margin6=arrays['margin6'],
            domain_idx=domains,
            domain_names=names,
            metadata_json=json.dumps(metadata),
        )
