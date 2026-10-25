"""Router-axis steering with calibrated orthogonal controls and norm restoration.

Uses the upstream OLMo-core factory without overriding normalization.
The pilot steers the final layer; all_controls covers all 16 layers.
"""

import argparse
from pathlib import Path

from src.configs.public_olmoe import OLMOE_CHECKPOINT_REVISIONS
from src.hf.historical_causal import ConditionSet, run


class Args:
    checkpoint_step: int
    eval_root: Path
    out: Path
    condition_set: ConditionSet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-step', type=int, choices=OLMOE_CHECKPOINT_REVISIONS, required=True,
                        help='retained experiment checkpoint; its commit is fixed in configuration')
    parser.add_argument('--eval-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--condition-set', choices=('pilot', 'all_controls'), default='pilot')
    args = parser.parse_args(namespace=Args())
    run(OLMOE_CHECKPOINT_REVISIONS[args.checkpoint_step], args.eval_root, args.out, args.condition_set)
