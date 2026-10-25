import json
import sys

import numpy as np
import pytest

pytest.importorskip('transformers')
from src.cli import prepare_public_titles as runner
from src.hf import titles as title_tokenizer


@pytest.mark.parametrize('model,revision', [
    ('qwen', '1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9'),
    ('olmoe', '37c50a5f4d30fba036185f57168ee8df0003f2e0'),
])
def test_public_title_preparation_pins_tokenizer_and_rejects_empty(tmp_path, monkeypatch, model, revision):
    titles = tmp_path / 'titles.json'
    titles.write_text(json.dumps(['A title'] * 256))
    output = tmp_path / 'tokens.npz'
    monkeypatch.setattr(sys, 'argv', ['prepare', '--model', model, '--out', str(output)])

    class Tokenizer:
        pad_token_id = 0
        empty = False

        def __call__(self, text, **kwargs):
            assert len(text) == 256 and kwargs['add_special_tokens'] is False
            assert kwargs['max_length'] == 64 and self.padding_side == 'right'
            shape = (256, 0 if self.empty else 2)
            return {'input_ids': np.ones(shape, dtype=np.int64),
                    'attention_mask': np.ones(shape, dtype=np.int64)}

    monkeypatch.setattr(runner, 'PUBLIC_TITLES_PATH', titles)
    tokenizer = Tokenizer()

    def load(model_id, **kwargs):
        assert kwargs == {'revision': revision}
        assert model_id == ('Qwen/Qwen3-30B-A3B-Base' if model == 'qwen' else 'allenai/OLMoE-1B-7B-0924')
        return tokenizer

    monkeypatch.setattr(title_tokenizer.AutoTokenizer, 'from_pretrained', load)
    tokenizer.empty = True
    with pytest.raises(ValueError, match='nonempty'):
        runner.main()
    assert not output.exists()
    tokenizer.empty = False
    runner.main()
    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_array_equal(data['lengths'], np.full(256, 2))
        assert json.loads(str(data['metadata_json']))['revision'] == revision
