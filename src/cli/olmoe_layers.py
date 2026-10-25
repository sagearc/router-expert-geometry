"""Public OLMoE last-token score/response correlations, one point per layer.

Input contains router_scores, selection_scores, selected_experts and w1_l2.
Response must be full ||W_gate h||_2, not RMS, top-10 L2 or post-SiLU activation.
Rows must already be the final non-padding token of each Wikipedia prompt.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.analysis.olmoe_layers import layer_correlations


class Args:
    capture: Path



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    args = parser.parse_args(namespace=Args())
    with np.load(args.capture, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata_json']))
        if metadata.get('token_scope') != 'last_nonpadding_token':
            parser.error('capture must explicitly identify last-nonpadding-token rows')
        result = layer_correlations(
            data['router_scores'].astype(np.float32), data['selection_scores'].astype(np.float32),
            data['w1_l2'].astype(np.float32), data['selected_experts'],
        )
    print(json.dumps({**result, 'capture_metadata': metadata}, indent=2, allow_nan=False))
