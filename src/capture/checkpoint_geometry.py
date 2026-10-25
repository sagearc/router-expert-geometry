"""Read only native checkpoint router tensors and summarize off-diagonal cosines."""

import re
from pathlib import Path

from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys

from src.analysis.measurements import router_cosines
from src.artifacts import GeometryLayer, GeometryResult


def checkpoint_geometry(checkpoint: str | Path) -> GeometryResult:
    metadata = get_checkpoint_metadata(str(checkpoint))
    pattern = re.compile(r'model\.blocks\.(\d+)\.feed_forward_moe\.router\.(weight|_centroid)$')
    matches = [(int(match[1]), key, match[2]) for key in metadata.state_dict_metadata
               if (match := pattern.fullmatch(key))]
    matches.sort()
    if not matches or len({layer for layer, _, _ in matches}) != len(matches):
        raise ValueError('expected exactly one router matrix per MoE layer')
    rows: list[GeometryLayer] = []
    keys = [key for _, key, _ in matches]
    for (layer, key, kind), tensor in zip(matches, load_keys(str(checkpoint), keys), strict=True):
        # The paper's native learned routers flatten (64 experts, d_model).
        if kind == 'weight':
            if tensor.ndim != 1 or tensor.numel() % 64:
                raise ValueError(f'{key}: expected flattened 64-expert router')
            tensor = tensor.reshape(64, -1)
        elif tensor.ndim != 2 or tensor.shape[0] != 64:
            raise ValueError(f'{key}: expected one centroid per each of 64 experts')
        pairs = router_cosines(tensor.detach().float().cpu().numpy())
        rows.append({'layer': layer + 1, 'checkpoint_key': key,
                     'experts': 64, 'pairs': len(pairs), 'mean': float(pairs.mean()),
                     'std': float(pairs.std()), 'min': float(pairs.min()), 'max': float(pairs.max())})
    return {'checkpoint': str(checkpoint), 'statistic': 'unique_off_diagonal_router_cosines',
            'spread': 'population_standard_deviation_across_expert_pairs', 'layers': rows}
