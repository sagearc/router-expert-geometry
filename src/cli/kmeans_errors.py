"""Reduce matched per-token NLL captures at the paper's step-21000 checkpoint."""

import argparse
import json

import numpy as np

from src.analysis.kmeans_errors import paired_error_analysis


class Args:
    kmeans: str
    reference: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kmeans', required=True, help='per_token_nll NPZ')
    parser.add_argument('--reference', required=True, help='matched loss-free per_token_nll NPZ')
    args = parser.parse_args(namespace=Args())
    with np.load(args.kmeans, allow_pickle=False) as km, \
            np.load(args.reference, allow_pickle=False) as ref:
        try:
            result = paired_error_analysis(km, ref)
        except ValueError as error:
            parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
