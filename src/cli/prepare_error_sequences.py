"""Pack the exact eleven-domain v3-small suite used by the error analysis.

No sampling or shuffling. Keep complete 2048-token sequences from each domain's
first validation shard, dropping only that shard's incomplete trailing sequence.
The historical files may be raw uint16 despite their .npy extension.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.constants import PAPER_SEQUENCE_LENGTH, VALIDATION_DOMAINS
from src.sequences import pack_shard


class Args:
    eval_root: Path
    out: Path



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-root', required=True, type=Path, help='v3_small_gptneox20b directory')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(namespace=Args())
    if args.out.exists():
        parser.error('output already exists')
    sequences, domain_ids, sources = [], [], []
    for i, domain in enumerate(VALIDATION_DOMAINS):
        path = args.eval_root / domain / 'val' / 'part-0-00000.npy'
        packed, dropped = pack_shard(path)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        sequences.append(packed)
        domain_ids.append(np.full(len(packed), i, dtype=np.uint8))
        sources.append({'domain': domain, 'sha256': digest, 'sequences': len(packed),
                        'dropped_trailing_tokens': dropped})
    with args.out.open('xb') as stream:
        np.savez_compressed(stream, sequences=np.concatenate(sequences),
                            domain_idx=np.concatenate(domain_ids), domain_names=np.array(VALIDATION_DOMAINS),
                            metadata_json=json.dumps({'sources': sources, 'sequence_length': PAPER_SEQUENCE_LENGTH}))
