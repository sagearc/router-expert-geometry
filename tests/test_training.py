import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('olmo_core')
from olmo_core.nn.moe.router import MoELinearRouter

from src.routers.centroid import PaperCentroidRouter
from src.routers.training import (
    RouterLearningRateCallback,
    set_router_learning_rate,
)


def test_learning_rate_reaches_only_kmeans_routers():
    model = torch.nn.ModuleDict({
        'kmeans': PaperCentroidRouter(d_model=2, num_experts=4, top_k=1, bias_lr_lambda=1., centroid_lr_lambda=10.),
        'linear': MoELinearRouter(d_model=2, num_experts=4, top_k=1),
    })
    assert set_router_learning_rate(model, .5) == 1
    assert model['kmeans'].learning_rate == .5
    assert set_router_learning_rate(torch.nn.Linear(2, 2), .5) == 0
    assert RouterLearningRateCallback().priority == 0
