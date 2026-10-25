"""Public OLMoE last-token score/response correlations, one point per layer.

Input contains router_scores, selection_scores, selected_experts and w1_l2.
Response must be full ||W_gate h||_2, not RMS, top-10 L2 or post-SiLU activation.
Rows must already be the final non-padding token of each Wikipedia prompt.
"""

import numpy as np

from src.analysis.measurements import per_token_correlations, selected_mask
from src.artifacts import CorrelationGroup, LayerCorrelationGroup, LayerCorrelations
from src.constants import OLMOE_EXPERTS, OLMOE_LAYERS, OLMOE_TOP_K
from src.types import FloatArray, IntArray


def layer_correlations(
    scores: FloatArray, selection_scores: FloatArray, response: FloatArray, selected: IntArray,
) -> LayerCorrelations:
    if scores.shape != selection_scores.shape or not np.isfinite(selection_scores).all():
        raise ValueError('selection scores must be finite and match router scores')
    if scores.ndim != 3 or scores.shape[1:] != (OLMOE_LAYERS, OLMOE_EXPERTS) or selected.shape != (*scores.shape[:2], OLMOE_TOP_K):
        raise ValueError('expected public OLMoE: 16 layers, 64 experts, native top-8')
    bottom = np.argsort(selection_scores, axis=-1, kind='stable')[..., :OLMOE_EXPERTS // 2]
    groups: dict[CorrelationGroup, IntArray] = {'routed_top8': selected, 'bottom8': bottom[..., :OLMOE_TOP_K], 'bottom_half': bottom}
    result: dict[CorrelationGroup, LayerCorrelationGroup] = {}
    for name, ids in groups.items():
        correlations = per_token_correlations(scores, response, selected_mask(ids, OLMOE_EXPERTS))
        valid = np.isfinite(correlations).sum(axis=0)
        if (valid == 0).any():
            raise ValueError(f'{name}: at least one layer has no defined correlations')
        result[name] = {
            'mean': np.nanmean(correlations, axis=0).tolist(),
            'title_std': np.nanstd(correlations, axis=0).tolist(),
            'valid_titles': valid.tolist(),
            'undefined_titles': (len(scores) - valid).tolist(),
        }
    return {'layers': list(range(1, OLMOE_LAYERS + 1)), 'n_titles': len(scores),
            'response': 'l2_norm_w_gate_h', 'aggregation': 'mean_of_per_title_spearman',
            'groups': result}
