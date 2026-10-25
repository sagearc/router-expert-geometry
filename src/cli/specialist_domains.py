"""Top-four specialist ablation versus random non-specialists across domains."""

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

from src.analysis.specialization import counts_routing, expert_sets
from src.artifacts import SpecialistCondition, SpecialistResult
from src.capture.ablation import mask_specialist_logits, sequence_nll
from src.configs.paper_1b import model_config
from src.constants import (
    PAPER_EVAL_BATCH_SIZE,
    PAPER_EXPERTS,
    PAPER_LAYERS,
    PAPER_SEQUENCE_LENGTH,
    SPECIALIST_CONTROL_EXPERTS,
    SPECIALIST_CONTROL_SEED,
    SPECIALIST_CONTROL_SEQUENCES,
    SPECIALIST_DOMAINS,
)


class Args:
    checkpoint: str
    sequences: Path
    counts: Path
    out: Path



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--sequences', type=Path, required=True, help='output of prepare-error-sequences')
    parser.add_argument('--counts', type=Path, required=True, help='output of capture-domain-counts for this checkpoint')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(namespace=Args())
    if args.out.exists() or not torch.cuda.is_available():
        parser.error('requires CUDA and a new output file')
    try:
        routing = counts_routing(args.counts, args.sequences)
    except ValueError as error:
        parser.error(str(error))
    with np.load(args.sequences, allow_pickle=False) as data:
        source_domains = data['domain_names'].tolist()
        domains = list(SPECIALIST_DOMAINS)
        sequences = {name: data['sequences'][data['domain_idx'] == source_domains.index(name)][:SPECIALIST_CONTROL_SEQUENCES] for name in domains}
    if any(x.shape != (SPECIALIST_CONTROL_SEQUENCES, PAPER_SEQUENCE_LENGTH) for x in sequences.values()):
        raise ValueError('requires 64 full 2048-token sequences for each of six specialist domains')
    with np.load(args.counts, allow_pickle=False) as data:
        counts, count_domains = data['counts'], data['domains'].tolist()
    if counts.shape != (PAPER_LAYERS, PAPER_EXPERTS, len(count_domains)) or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('invalid nine-layer, 64-expert domain counts')
    config = model_config(routing)
    if any(not np.issubdtype(x.dtype, np.integer) or (x < 0).any() or (x >= config.vocab_size).any()
           for x in sequences.values()):
        raise ValueError('invalid token IDs')
    model = cast(MoETransformer, config.build(init_device='cpu'))
    load_model_and_optim_state(args.checkpoint, model, strict=True)
    model.to(device='cuda', dtype=torch.float32).eval()
    routers = [cast(MoETransformerBlock, block).feed_forward_moe.router for block in model.blocks.values()]

    def evaluate() -> dict[str, list[float]]:
        return {name: sequence_nll(model, x, PAPER_EVAL_BATCH_SIZE, 'cuda').tolist() for name, x in sequences.items()}

    baseline = evaluate()
    result: SpecialistResult = {'checkpoint': args.checkpoint, 'routing': routing, 'baseline': baseline,
              'conditions': {}, 'top_r': SPECIALIST_CONTROL_EXPERTS, 'random_seed': SPECIALIST_CONTROL_SEED,
              'sequences_sha256': hashlib.sha256(args.sequences.read_bytes()).hexdigest(),
              'counts_sha256': hashlib.sha256(args.counts.read_bytes()).hexdigest()}
    for target, top, random in expert_sets(counts, count_domains):
        for label, ids in [('specialists', top), ('random_non_specialists', random)]:
            with mask_specialist_logits(routers, ids):
                losses = evaluate()
            damage = {name: float(np.mean(losses[name]) - np.mean(baseline[name])) for name in domains}
            record: SpecialistCondition = {'experts': ids.tolist(), 'sequence_nll': losses, 'damage_by_domain': damage}
            if label == 'specialists':
                off = float(np.mean([damage[name] for name in domains if name != target]))
                record['damage_on'] = damage[target]
                record['damage_off'] = off
                record['specialization_index'] = damage[target] / max(off, 1e-9)
            result['conditions'][f'{target}/{label}'] = record
            print(target, label, damage, flush=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
