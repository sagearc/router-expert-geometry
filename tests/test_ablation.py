import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('olmo_core')

from olmo_core.nn.moe import MoERouterConfig, MoERouterGatingFunction

from src.capture.ablation import mask_specialist_logits


def test_biased_sigmoid_cannot_silently_reselect_masked_expert():
    router = MoERouterConfig(
        top_k=2, gating_function=MoERouterGatingFunction.sigmoid, bias_gamma=1e-3,
    ).build(2, 4).eval()
    with torch.no_grad():
        router.weight.zero_()
        router.score_bias.copy_(torch.tensor([1., 0., 0., 0.]))
    original = router.get_expert_logits
    with pytest.raises(RuntimeError, match='reselected'), mask_specialist_logits([router], [[0]]):
        logits = router.get_expert_logits(torch.ones(1, 2))
        router.get_top_k(logits.sigmoid() + 1e-7)
    assert router.get_expert_logits == original
    assert torch.isfinite(router.get_expert_logits(torch.ones(1, 2))).all()


def test_softmax_exclusion_preserves_remaining_native_selection():
    router = MoERouterConfig(top_k=2).build(2, 4).eval()
    with torch.no_grad():
        router.weight.copy_(torch.arange(8).float())
    with mask_specialist_logits([router], [[3]]):
        logits = router.get_expert_logits(torch.ones(1, 2))
        _, selected = router.get_top_k(logits.softmax(-1))
        assert selected.tolist() == [[2, 1]]
