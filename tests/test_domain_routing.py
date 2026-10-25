from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.capture.domain_routing import routing_mass


def test_routing_mass_counts_every_token_and_restores_hooks_on_failure():
    class Router(torch.nn.Module):
        num_experts = 3

        def forward(self, tokens):
            ids = torch.stack((tokens % 3, (tokens + 1) % 3), dim=-1)
            weights = torch.tensor([0.25, 0.75]).expand_as(ids)
            return weights, ids, None, None

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.router = Router()
            self.blocks = {'0': SimpleNamespace(feed_forward_moe=SimpleNamespace(router=self.router))}
            self.fail = False

        def forward(self, tokens):
            self.router(tokens)
            if self.fail:
                raise RuntimeError('forward failed')

    model = Model().eval()
    tokens = np.array([[0, 0, 2], [1, 2, 2], [0, 1, 2]])
    counts, weights = routing_mass(model, tokens, batch_size=2)
    # Token frequencies are 3, 2, 4. Each token selects itself and its successor.
    np.testing.assert_array_equal(counts, [[7, 5, 6]])
    np.testing.assert_array_equal(weights, [[3.75, 2.75, 2.5]])
    assert not model.router._forward_hooks
    model.fail = True
    with pytest.raises(RuntimeError, match='forward failed'):
        routing_mass(model, tokens)
    assert not model.router._forward_hooks
