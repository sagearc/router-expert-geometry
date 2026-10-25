"""Capture paired clean/r1/r4 specialist NLL on fixed code-domain sequences.

Matches the retained per-sequence experiment, with explicit masked-expert
exclusion checks. If selection bias reselects a specialist, stop rather than
silently changing the intervention or emitting a misleading result.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import torch
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.transformer.block import MoETransformerBlock
from olmo_core.nn.transformer.model import MoETransformer

from src.analysis.specialization import counts_routing
from src.capture.ablation import mask_specialist_logits, sequence_nll
from src.configs.paper_1b import model_config
from src.constants import (
    PAPER_EVAL_BATCH_SIZE,
    PAPER_EXPERTS,
    PAPER_LAYERS,
    PAPER_SEQUENCE_LENGTH,
    SPECIALIST_CODE_DOMAIN,
    SPECIALIST_EXPERT_COUNTS,
    SPECIALIST_SEQUENCE_COUNT,
)
from src.sequences import code_sequences


class Args:
    checkpoint: str
    sequences: Path
    counts: Path
    out: Path



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--sequences', required=True, type=Path, help='NPZ from prepare-error-sequences, shared by both models')
    parser.add_argument('--counts', required=True, type=Path, help='output of capture-domain-counts for this checkpoint')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(namespace=Args())
    if not torch.cuda.is_available():
        parser.error('GPU kernels are required')
    if args.out.exists():
        parser.error('output already exists')
    try:
        routing = counts_routing(args.counts, args.sequences)
    except ValueError as error:
        parser.error(str(error))
    sequences = code_sequences(args.sequences)
    with np.load(args.counts, allow_pickle=False) as data:
        domain = list(data['domains']).index(SPECIALIST_CODE_DOMAIN)
        counts = data['counts'][:, :, domain]
    if counts.shape != (PAPER_LAYERS, PAPER_EXPERTS) or not np.isfinite(counts).all() or (counts < 0).any():
        parser.error('expected finite nonnegative domain counts for nine layers and 64 experts')
    ranking = np.argsort(-counts, axis=1)
    config = model_config(routing)
    if (sequences < 0).any() or (sequences >= config.vocab_size).any():
        parser.error('token ID outside model vocabulary')
    model = cast(MoETransformer, config.build(init_device='cpu'))
    load_model_and_optim_state(args.checkpoint, model, strict=True)
    model.to(device='cuda', dtype=torch.float32).eval()
    routers = [cast(MoETransformerBlock, block).feed_forward_moe.router for block in model.blocks.values()]
    results = {'clean': sequence_nll(model, sequences, PAPER_EVAL_BATCH_SIZE, 'cuda')}
    for r in SPECIALIST_EXPERT_COUNTS:
        with mask_specialist_logits(routers, ranking[:, :r]):
            results[f'r{r}'] = sequence_nll(model, sequences, PAPER_EVAL_BATCH_SIZE, 'cuda')
        print(f'r{r}: mean NLL={results[f"r{r}"].mean():.6f}', flush=True)
    metadata = {'checkpoint': args.checkpoint, 'routing': routing, 'domain': SPECIALIST_CODE_DOMAIN,
                'sequences_per_domain': SPECIALIST_SEQUENCE_COUNT, 'tokens_per_sequence': PAPER_SEQUENCE_LENGTH,
                'sequences_sha256': hashlib.sha256(args.sequences.read_bytes()).hexdigest(),
                'counts_sha256': hashlib.sha256(args.counts.read_bytes()).hexdigest(),
                'forward_dtype': 'float32', 'intervention': 'raw_logit_mask_with_exclusion_assertion'}
    with args.out.open('xb') as stream:
        np.savez_compressed(
            stream,
            clean=results['clean'],
            r1=results['r1'],
            r4=results['r4'],
            metadata_json=json.dumps(metadata),
        )
