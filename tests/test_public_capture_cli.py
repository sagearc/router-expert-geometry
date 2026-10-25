import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')

from src.cli import capture_public_olmoe


@pytest.mark.parametrize('response', [1.0, float('inf'), float('nan')])
@pytest.mark.parametrize('step,revision', [
    (5000, '37c50a5f4d30fba036185f57168ee8df0003f2e0'),
    (1000000, '33d21fb2f1382acb824dcdf25e893f44f3ad57ed'),
])
def test_public_capture_preserves_historical_batch_width_and_attention(tmp_path, monkeypatch, response, step, revision):
    runner = vars(capture_public_olmoe)
    scope = runner['main'].__globals__
    monkeypatch.setitem(scope, 'OLMOE_CAPTURE_BATCH_SIZE', 2)
    tokens, output = tmp_path / 'tokens.npz', tmp_path / 'out.npz'
    np.savez(tokens, input_ids=np.arange(24).reshape(3, 8), lengths=np.array([3, 1, 5]))
    config = SimpleNamespace(num_hidden_layers=16, num_experts=64, num_experts_per_tok=8,
                             vocab_size=32, _attn_implementation='sdpa', _experts_implementation='eager')
    model = SimpleNamespace(config=config)
    model.to = lambda *args: model
    model.eval = lambda: model
    loaded = []

    def load(*args, **kwargs):
        loaded.append(kwargs)
        return model

    monkeypatch.setitem(scope, 'OlmoeForCausalLM', SimpleNamespace(from_pretrained=load))
    tensor, arange = torch.as_tensor, torch.arange
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch, 'as_tensor', lambda x, **kw: tensor(x, dtype=kw.get('dtype')))
    monkeypatch.setattr(torch, 'arange', lambda n, **kw: arange(n))
    batches = []

    def capture(model, ids, mask):
        batches.append((tuple(ids.shape), mask.tolist()))
        return {**{key: torch.full((len(ids), 16, 64), response)
                   for key in ('router_scores', 'selection_scores', 'w1_l2', 'w3_l2')},
                'selected_experts': torch.arange(8).expand(len(ids), 16, 8)}

    monkeypatch.setitem(scope, 'capture_olmoe', capture)
    monkeypatch.setattr('sys.argv', ['capture_public_olmoe.py', '--checkpoint-step', str(step),
                                   '--tokens', str(tokens), '--out', str(output)])
    if response != 1.0:
        with pytest.raises(ValueError, match='nonfinite'):
            runner['main']()
        assert not output.exists()
        return
    runner['main']()
    assert loaded[0]['revision'] == revision
    assert loaded[0]['attn_implementation'] == 'sdpa'
    assert batches == [((2, 3), [[True, True, True], [True, False, False]]),
                       ((1, 5), [[True] * 5])]
    with np.load(output, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata_json']))
        assert metadata['revision'] == revision
        assert metadata['padding'] == 'right_to_batch_max_length'
        assert metadata['token_file_width'] == 8
