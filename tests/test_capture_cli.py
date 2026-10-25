import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('olmo_core')

from src.cli import capture_1b


@pytest.mark.parametrize('response', [1.0, float('inf'), float('nan'), 1e8])
def test_capture_cli_padding_and_rejects_invalid_storage(tmp_path, monkeypatch, response):
    runner = vars(capture_1b)
    scope = runner['main'].__globals__
    monkeypatch.setitem(scope, 'PAPER_CAPTURE_BATCH_SIZE', 2)
    tokens, output = tmp_path / 'tokens.npz', tmp_path / 'capture.npz'
    ids = np.arange(24).reshape(3, 8)
    np.savez(tokens, input_ids=ids, lengths=np.array([3, 1, 5]))
    model = SimpleNamespace()
    model.to = lambda **kwargs: model
    model.eval = lambda: model
    monkeypatch.setitem(scope, 'model_config', lambda routing: SimpleNamespace(
        vocab_size=32, build=lambda **kwargs: model))
    monkeypatch.setitem(scope, 'load_model_and_optim_state', lambda *args, **kwargs: None)
    # Exercise CLI batching and serialization only; GPU forward is tested separately.
    tensor = torch.as_tensor
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch, 'as_tensor', lambda value, **kwargs: tensor(value, dtype=kwargs['dtype']))
    seen = []

    def capture(model, batch, lengths):
        seen.append((batch.clone(), lengths.clone()))
        return {'router_scores': torch.full((len(batch), 1, 2), response),
                'selected_experts': torch.zeros(len(batch), 1, 1, dtype=torch.long),
                **{key: torch.full((len(batch), 1, 2), response)
                   for key in ('w1_top10_l2', 'w1_silu_mean', 'w1_l2', 'w3_l2')}}

    monkeypatch.setitem(scope, 'capture_last_tokens', capture)
    monkeypatch.setattr('sys.argv', ['capture_1b.py', '--checkpoint', 'unused',
                                   '--routing', 'loss_free', '--tokens', str(tokens),
                                   '--out', str(output)])
    if response != 1.0:
        with pytest.raises((ValueError, FloatingPointError)):
            runner['main']()
        assert not output.exists()
        return
    runner['main']()
    assert [tuple(batch.shape) for batch, _ in seen] == [(2, 3), (1, 5)]
    np.testing.assert_array_equal(seen[0][0], ids[:2, :3])
    np.testing.assert_array_equal(seen[1][0], ids[2:, :5])
    with np.load(output, allow_pickle=False) as data:
        assert data['router_scores'].shape == (3, 1, 2)
        metadata = json.loads(str(data['metadata_json']))
        assert metadata['batch_size'] == 2
        assert metadata['padding'] == 'right_to_batch_max_length'
