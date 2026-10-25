"""Capture count and weight mass for the six-domain specialist analysis."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import torch
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.transformer.model import MoETransformer

from src.capture.domain_routing import routing_mass
from src.configs.paper_1b import model_config
from src.constants import (
    BALANCED_ROUTINGS,
    PAPER_EVAL_BATCH_SIZE,
    PAPER_SEQUENCE_LENGTH,
    PAPER_TOP_K,
    SPECIALIST_DOMAINS,
    SPECIALIST_SEQUENCE_COUNT,
)
from src.types import BalancedRouting


class Args:
    checkpoint: str
    routing: BalancedRouting
    sequences: Path
    out: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--routing', choices=BALANCED_ROUTINGS, required=True)
    parser.add_argument('--sequences', type=Path, required=True, help='output of prepare-error-sequences')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(namespace=Args())
    if args.out.exists() or not torch.cuda.is_available():
        parser.error('requires CUDA and a new output file')
    with np.load(args.sequences, allow_pickle=False) as data:
        names = data['domain_names'].tolist()
        sequences = [data['sequences'][data['domain_idx'] == names.index(name)][:SPECIALIST_SEQUENCE_COUNT] for name in SPECIALIST_DOMAINS]
    config = model_config(args.routing)
    if any(x.shape != (SPECIALIST_SEQUENCE_COUNT, PAPER_SEQUENCE_LENGTH) or not np.issubdtype(x.dtype, np.integer)
           or (x < 0).any() or (x >= config.vocab_size).any() for x in sequences):
        raise ValueError('requires 240 full valid sequences per specialist domain')
    model = cast(MoETransformer, config.build(init_device='cpu'))
    load_model_and_optim_state(args.checkpoint, model, strict=True)
    model.to(device='cuda', dtype=torch.float32).eval()
    counts, weights = [], []
    for name, tokens in zip(SPECIALIST_DOMAINS, sequences, strict=True):
        count, weight = routing_mass(model, tokens)
        if not np.all(count.sum(axis=1) == SPECIALIST_SEQUENCE_COUNT * PAPER_SEQUENCE_LENGTH * PAPER_TOP_K):
            raise ValueError('routing counts do not cover every token and top-6 slot')
        counts.append(count)
        weights.append(weight)
        print(f'captured {name}', flush=True)
    metadata = {'checkpoint': args.checkpoint, 'routing': args.routing,
                'sequences_per_domain': SPECIALIST_SEQUENCE_COUNT, 'tokens_per_sequence': PAPER_SEQUENCE_LENGTH,
                'batch_size': PAPER_EVAL_BATCH_SIZE, 'forward_dtype': 'float32', 'token_scope': 'all_including_final',
                'sequences_sha256': hashlib.sha256(args.sequences.read_bytes()).hexdigest()}
    with args.out.open('xb') as stream:
        np.savez_compressed(stream, counts=np.stack(counts, axis=-1),
                            counts_w=np.stack(weights, axis=-1), domains=np.array(SPECIALIST_DOMAINS),
                            metadata_json=json.dumps(metadata))
