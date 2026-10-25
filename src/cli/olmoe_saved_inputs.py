"""Recompute public OLMoE responses from the retained 256-title hidden inputs.

This reproduces the layer-table workflow, not a new model forward pass.
Run olmoe_layers.py on the resulting capture to obtain correlations.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
import transformers
from transformers import OlmoeForCausalLM

from src.artifacts import RetainedInputs
from src.constants import (
    OLMOE_EXPERTS,
    OLMOE_HIDDEN_SIZE,
    OLMOE_LAYERS,
    OLMOE_MODEL_ID,
    OLMOE_TOP_K,
    PUBLIC_TITLE_COUNT,
)
from src.hf.olmoe import expert_response_norms
from src.sequences import final_token_rows
from src.types import FloatArray


class Args:
    capture: Path
    out: Path



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(namespace=Args())
    if args.out.exists() or not torch.cuda.is_available():
        parser.error('requires CUDA and a new output path')
    with np.load(args.capture, allow_pickle=False) as data:
        rows = final_token_rows(cast(RetainedInputs, data))
        hidden = data['router_inputs'][rows]
        scores = data['router_scores'][rows]
        selection = data['selection_scores'][rows]
        selected = data['selected_experts'][rows]
        revision = str(data['resolved_revision'])
        model_id = str(data['model'])
    if model_id != OLMOE_MODEL_ID or not re.fullmatch('[0-9a-f]{40}', revision):
        parser.error('requires the retained public OLMoE model and immutable revision')
    if hidden.shape != (PUBLIC_TITLE_COUNT, OLMOE_LAYERS, OLMOE_HIDDEN_SIZE) or scores.shape != (PUBLIC_TITLE_COUNT, OLMOE_LAYERS, OLMOE_EXPERTS):
        parser.error('requires 256 retained last-token rows, 16 layers and 64 experts')
    if selection.shape != scores.shape or selected.shape != (PUBLIC_TITLE_COUNT, OLMOE_LAYERS, OLMOE_TOP_K):
        parser.error('invalid selection arrays')
    if not all(np.isfinite(value).all() for value in (hidden, scores, selection)):
        parser.error('non-finite retained inputs or scores')
    model = OlmoeForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16,
    ).to('cuda').eval()  # type: ignore[arg-type]  # Transformers 5.1 @wraps loses the bound .to() signature.
    responses: dict[Literal['w1_l2', 'w3_l2'], list[FloatArray]] = {'w1_l2': [], 'w3_l2': []}
    for index, layer in enumerate(model.model.layers):
        norms = expert_response_norms(layer.mlp.experts, torch.from_numpy(hidden[:, index]))
        for key, value in norms.items():
            responses[key].append(value.numpy())
        print(f'recomputed layer {index + 1}/16', flush=True)
    metadata = {
        'model': model_id, 'revision': revision,
        'token_scope': 'last_nonpadding_token',
        'workflow': 'reproject_retained_hidden_inputs_without_forward',
        'response_projection_dtype': 'float32', 'weight_load_dtype': 'bfloat16',
        'retained_hidden_dtype': str(hidden.dtype),
        'source_sha256': hashlib.sha256(args.capture.read_bytes()).hexdigest(),
        'torch': torch.__version__, 'transformers': transformers.__version__,
    }
    with args.out.open('xb') as stream:
        np.savez_compressed(
            stream, router_scores=scores, selection_scores=selection, selected_experts=selected,
            w1_l2=np.stack(responses['w1_l2'], axis=1),
            w3_l2=np.stack(responses['w3_l2'], axis=1),
            metadata_json=json.dumps(metadata),
        )
