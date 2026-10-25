import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('olmo_core')

from olmo_core.distributed.checkpoint import save_state_dict

from src.capture.checkpoint_geometry import checkpoint_geometry


def test_router_only_checkpoint_read_and_numeric_layer_order(tmp_path):
    path = tmp_path / 'checkpoint'
    save_state_dict(path, {'model': {
        'blocks.10.feed_forward_moe.router.weight': torch.ones(64 * 2),
        'blocks.2.feed_forward_moe.router._centroid': torch.ones(64, 2),
        'embeddings.weight': torch.ones(3, 2),
    }})
    result = checkpoint_geometry(path)
    assert [row['layer'] for row in result['layers']] == [3, 11]
    for row in result['layers']:
        assert row['pairs'] == 2016
        assert row['mean'] == pytest.approx(1)
        assert row['std'] == pytest.approx(0, abs=1e-15)
