import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('olmo_core')
from olmo_core.nn.moe import MoERouterGatingFunction

from src.configs.paper_1b import model_config
from src.routers.seq_aux import (
    PaperMoEConfig,
    PaperSeqAuxConfig,
    PaperSeqAuxRouter,
    sequence_auxiliary_loss,
)


def test_sequence_loss_matches_deepseek_formula_per_sequence():
    scores = torch.tensor([[[.5, .5, 0.], [1., 0., 0.]], [[0., 0., 1.], [0., 0., 1.]]])
    counts = torch.tensor([[2, 1, 1], [0, 0, 4]])  # top_k 2 over 2 tokens: 4 assignments each.
    # Sequence 0: f = (.5, .25, .25), P = (.75, .25, 0); sequence 1: f = P = (0, 0, 1).
    expected = (3 * (.5 * .75 + .25 * .25) + 3) / 2
    loss = sequence_auxiliary_loss(num_experts=3, top_k=2, expert_scores=scores, batched_batch_size_per_expert=counts)
    assert loss.item() == pytest.approx(expected)
    scaled = sequence_auxiliary_loss(num_experts=3, top_k=2, expert_scores=scores * 4, batched_batch_size_per_expert=counts)
    torch.testing.assert_close(scaled, loss)  # Scores are L1-normalized per token first.
    whole = sequence_auxiliary_loss(num_experts=3, top_k=2, expert_scores=scores,
                                    batched_batch_size_per_expert=counts, loss_div_factor=4.)
    halves = sum(sequence_auxiliary_loss(num_experts=3, top_k=2, expert_scores=scores[i:i + 1],
                                         batched_batch_size_per_expert=counts[i:i + 1], loss_div_factor=4.)
                 for i in range(2))
    assert whole.item() == pytest.approx(expected) and halves.item() == pytest.approx(expected)


def test_router_adds_weighted_loss_with_gradient_only_in_training():
    router = PaperSeqAuxConfig(
        top_k=2, gating_function=MoERouterGatingFunction.sigmoid, bias_gamma=1e-3,
        normalize_expert_weights=1., seq_aux_loss_weight=.5,
    ).build(4, 3)
    assert isinstance(router, PaperSeqAuxRouter)
    x = torch.randn(2, 3, 4, requires_grad=True)
    router.train()
    weights, indices, _counts, aux = router(x)
    assert aux is not None and aux.requires_grad
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 3))
    scores = torch.sigmoid(router.get_expert_logits(x)) + 1e-7
    per_sequence = torch.nn.functional.one_hot(indices, 3).sum(dim=(1, 2))
    expected = .5 * sequence_auxiliary_loss(num_experts=3, top_k=2, expert_scores=scores,
                                            batched_batch_size_per_expert=per_sequence)
    torch.testing.assert_close(aux, expected)
    aux.backward()
    assert router.weight.grad is not None and torch.count_nonzero(router.weight.grad) > 0
    metrics = router.compute_metrics()
    torch.testing.assert_close(metrics['seq aux loss'][0], expected.detach())
    torch.testing.assert_close(metrics['seq aux loss unscaled'][0], expected.detach() / .5)
    assert router.seq_aux_loss.item() == 0
    router.eval()
    assert router(x)[3] is None
    with pytest.raises(Exception, match='jitter'):
        PaperSeqAuxConfig(top_k=2, jitter_eps=.1).build(4, 3)


def test_moe_config_scales_sequence_loss_per_layer_without_changing_config():
    moe = model_config('seq_aux').block.feed_forward_moe
    assert isinstance(moe, PaperMoEConfig) and isinstance(moe.router, PaperSeqAuxConfig)
    moe.num_experts, moe.hidden_size, moe.shared_mlp.hidden_size = 4, 8, 8
    built = moe.build(8, n_layers=9)
    assert isinstance(built.router, PaperSeqAuxRouter)
    assert built.router.seq_aux_loss_weight == pytest.approx(1e-4 / 9)
    assert built.router.gating_function == MoERouterGatingFunction.sigmoid
    assert built.router.bias_gamma == 1e-3 and built.router.normalize_expert_weights == 1.
    assert moe.router.seq_aux_loss_weight == 1e-4
    moe.scale_loss_by_num_layers = False
    assert moe.build(8, n_layers=9).router.seq_aux_loss_weight == 1e-4
