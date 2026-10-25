"""Reduce retained paired specialist-ablation losses with matching mass shares."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.analysis.specialization import load_damage_and_share, paired_damage_ratio
from src.constants import (
    SPECIALIST_CODE_DOMAIN,
    SPECIALIST_EXPERT_COUNTS,
)


class Args:
    aux_losses: Path
    loss_free_losses: Path
    aux_routing: Path
    loss_free_routing: Path
    experts: int



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aux-losses', required=True, type=Path)
    parser.add_argument('--loss-free-losses', required=True, type=Path)
    parser.add_argument('--aux-routing', required=True, type=Path)
    parser.add_argument('--loss-free-routing', required=True, type=Path)
    parser.add_argument('--experts', type=int, choices=SPECIALIST_EXPERT_COUNTS, default=1)
    args = parser.parse_args(namespace=Args())
    sequence_hashes = []
    for path in (args.aux_losses, args.loss_free_losses):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata_json'])) if 'metadata_json' in data else {}
            sequence_hashes.append(metadata.get('sequences_sha256'))
    pairing_verified = all(sequence_hashes) and sequence_hashes[0] == sequence_hashes[1]
    if any(sequence_hashes) and not pairing_verified:
        parser.error('sequence hashes differ or only one capture identifies its sequences')
    aux, aux_share = load_damage_and_share(args.aux_losses, args.aux_routing, SPECIALIST_CODE_DOMAIN, args.experts, 'aux_loss')
    lfb, lfb_share = load_damage_and_share(args.loss_free_losses, args.loss_free_routing, SPECIALIST_CODE_DOMAIN, args.experts, 'loss_free')
    result = paired_damage_ratio(aux, lfb, aux_share, lfb_share)
    print(json.dumps({'domain': SPECIALIST_CODE_DOMAIN, 'experts_per_layer': args.experts,
                      'sequence_pairing_hash_verified': bool(pairing_verified), **result}, indent=2))
