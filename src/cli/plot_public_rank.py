"""Plot retained Qwen and DeepSeek-V4 expert-rank response captures."""

import argparse
import json
from pathlib import Path

from src.analysis.rank_curve import rank_curve
from src.plots.public_rank import plot


class Args:
    capture: Path
    out: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--out', type=Path, required=True, help='new PNG, PDF or SVG output')
    args = parser.parse_args(namespace=Args())
    if args.out.exists() or args.out.with_suffix('.json').exists():
        parser.error('choose new image and companion JSON output paths')
    if args.out.suffix.lower() not in ('.png', '.pdf', '.svg'):
        parser.error('expected PNG, PDF or SVG output')
    try:
        result = rank_curve(args.capture)
    except ValueError as error:
        parser.error(str(error))
    plot(result, args.out)
    args.out.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
