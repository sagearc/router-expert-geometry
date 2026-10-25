"""Show ranks for all four interventions at all 16 OLMoE layers."""

import argparse
import hashlib
import json
from pathlib import Path

from src.analysis.causal_summary import summarize


class Args:
    capture: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    args = parser.parse_args(namespace=Args())
    result = summarize(json.loads(args.capture.read_text()))
    result['source_sha256'] = hashlib.sha256(args.capture.read_bytes()).hexdigest()
    print(json.dumps(result, indent=2, allow_nan=False))
