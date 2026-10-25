import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('olmo_core')
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.moe import MoERouterGatingFunction

from src.configs.paper_1b import model_config
from src.routers.centroid import (
    PaperCentroidConfig,
    PaperCentroidRouter,
)


def test_cosine_scores_start_at_zero_and_ignore_scale():
    router = model_config('kmeans').block.feed_forward_moe.router.build(d_model=2, num_experts=64).eval()
    assert isinstance(router, PaperCentroidRouter)
    x = torch.tensor([[[1., 0.]]])
    assert torch.count_nonzero(router.get_expert_logits(x)) == 0
    state = router.state_dict()
    assert set(state) == {'_centroid', '_centroid_step', 'score_bias'}
    state['_centroid'][0] = torch.tensor([1., 0.])
    state['_centroid'][1] = torch.tensor([10., 10.])
    state['_centroid_step'].fill_(1)
    router.load_state_dict(state, strict=True)
    scores = router.get_expert_logits(x)
    assert scores.argmax(-1).item() == 0  # Dot products would prefer expert 1.
    torch.testing.assert_close(scores[..., :2], torch.tensor([[[1., 2**-0.5]]]))
    torch.testing.assert_close(router.get_expert_logits(x * 7), scores)
    router._centroid.mul_(3)
    torch.testing.assert_close(router.get_expert_logits(x), scores)
    router.reset_parameters()
    assert torch.count_nonzero(router._centroid) == 0 and router._centroid_step.item() == 0
    assert torch.count_nonzero(router.get_expert_logits(x)) == 0


def test_selection_adds_bias_but_weights_exclude_it():
    router = PaperCentroidRouter(d_model=2, num_experts=4, top_k=2, bias_lr_lambda=1., centroid_lr_lambda=10.).eval()
    with torch.no_grad():
        router._centroid.copy_(torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]))
        router.score_bias.copy_(torch.tensor([0., 0., 3., 0.]))
    weights, selected, counts, aux = router(torch.tensor([[[1., 0.]]]))
    assert selected.tolist() == [[[2, 0]]] and aux is None
    torch.testing.assert_close(weights, torch.tensor([[[-1., 1.]]]))
    assert counts.tolist() == [1, 0, 1, 0]


def test_batch_updates_scale_with_learning_rate():
    router = PaperCentroidRouter(d_model=2, num_experts=4, top_k=1, bias_lr_lambda=2., centroid_lr_lambda=10.).train()
    with torch.no_grad():
        router._centroid.copy_(torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]))
    x = torch.tensor([[[2., 0.], [4., 0.], [0., 1.]]])
    router(x)
    with pytest.raises(ValueError, match='learning rate'):
        router.post_batch()
    router.learning_rate = 1.
    with pytest.raises(OLMoConfigurationError, match='outside'):
        router.post_batch()
    router.learning_rate = .05  # Centroid step 0.5, bias step 0.1.
    router.post_batch(dry_run=True)
    assert router._centroid_step.item() == 0 and torch.count_nonzero(router.score_bias) == 0
    torch.testing.assert_close(router._centroid[0], torch.tensor([1., 0.]))
    router.post_batch()  # The dry run consumed the batch; nothing to apply.
    assert router._centroid_step.item() == 0
    router(x)
    router.post_batch()
    assert router._centroid_step.item() == 1
    torch.testing.assert_close(router._centroid, torch.tensor([[2., 0.], [0., 1.], [-1., 0.], [0., -1.]]))
    # Counts (2, 1, 0, 0) against their mean of 0.75: the two routed experts lose bias.
    torch.testing.assert_close(router.score_bias, torch.tensor([-.1, -.1, .1, .1]))
    with torch.no_grad():
        router(x)
    router.post_batch()
    router.eval()
    router(x)
    router.post_batch()
    assert router._centroid_step.item() == 1


def test_config_builds_only_paper_options_and_counts_no_parameters():
    config = PaperCentroidConfig(top_k=6)
    assert config.num_params(1024, 64) == 0
    router = config.build(1024, 64, init_device='meta')
    assert (router.bias_lr_lambda, router.centroid_lr_lambda) == (1., 10.)
    assert router._centroid.shape == (64, 1024) and router._centroid.dtype == torch.float32
    assert router.weight.numel() == 0 and not list(router.parameters())
    for option in ({'bias_gamma': 1e-3}, {'normalize_expert_weights': 1.}, {'jitter_eps': .1},
                   {'uniform_expert_assignment': True}, {'gating_function': MoERouterGatingFunction.sigmoid}):
        with pytest.raises(OLMoConfigurationError):
            PaperCentroidConfig(top_k=6, **option).build(4, 8)
    with pytest.raises(OLMoConfigurationError):
        config.build(4, 8, lb_loss_weight=.01)


def test_model_initialization_leaves_centroids_and_bias_at_zero():
    config = model_config('kmeans')
    config.d_model, config.n_layers, config.vocab_size = 8, 1, 16
    config.block.sequence_mixer.n_heads = 2
    moe = config.block.feed_forward_moe
    moe.num_experts, moe.hidden_size, moe.router.top_k = 4, 8, 2
    moe.shared_mlp.hidden_size = 8
    model = config.build(init_device='cpu')
    model.init_weights()
    router = model.blocks['0'].feed_forward_moe.router
    assert isinstance(router, PaperCentroidRouter)
    assert torch.count_nonzero(router._centroid) == 0 and torch.count_nonzero(router.score_bias) == 0
    assert torch.count_nonzero(model.blocks['0'].feed_forward_moe.experts.mlp.w1) > 0
