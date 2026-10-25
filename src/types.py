"""Array dtypes and native router contracts used by the measurements.

Model dependencies stay optional for the NumPy-only analysis modules.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from torch import Tensor

# Captures retain FP16/FP32, while statistical reductions use FP64.
type FloatArray = NDArray[np.float16 | np.float32 | np.float64]
type IntArray = NDArray[np.int8 | np.uint8 | np.int16 | np.int32 | np.int64 | np.uint16 | np.uint32 | np.uint64]
type BoolArray = NDArray[np.bool_]
type Float64Array = NDArray[np.float64]
type Int64Array = NDArray[np.int64]
type ResponseMetric = Literal['w1_top10_l2', 'w1_silu_mean', 'w1_l2', 'w3_l2']
type CaptureMetric = ResponseMetric | Literal['router_scores', 'selected_experts']
type PublicCaptureMetric = Literal['w1_l2', 'w3_l2', 'router_scores', 'selection_scores', 'selected_experts']

# OLMo-core returns weights, IDs, dispatch counts, and optional auxiliary loss.
type RouterOutput = tuple[Tensor, Tensor, Tensor, Tensor | None]
type TopK = Callable[[Tensor], tuple[Tensor, Tensor]]
type ExpertLogits = Callable[[Tensor], Tensor]


type Routing = Literal['aux_loss', 'loss_free', 'seq_aux', 'kmeans']
type BalancedRouting = Literal['aux_loss', 'loss_free']
type ErrorRouting = Literal['aux_loss', 'loss_free', 'kmeans']
