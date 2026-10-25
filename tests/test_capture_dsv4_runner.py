import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.mark.parametrize('failure', [None, 'generate', 'coverage'])
def test_runner_closes_engine_and_only_writes_complete_capture(tmp_path, monkeypatch, failure):
    events = []

    class Engine:
        def __init__(self, **kwargs):
            assert kwargs['tensor_parallel_size'] == 1
            assert not kwargs['enable_prefix_caching']
            assert kwargs['revision'] == kwargs['tokenizer_revision']
            assert len(kwargs['revision']) == 40
            self.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(
                shutdown=lambda: events.append('shutdown')))

        def get_tokenizer(self):
            return SimpleNamespace(encode=lambda title, **kwargs: [1, 2], eos_token_id=0)

        def apply_model(self, function):
            if hasattr(function, 'keywords'):
                assert function.keywords['total_prompt_tokens'] == 512
                events.append('install')
                return [None]
            events.append('finish')
            count = 511 if failure == 'coverage' else 512
            return [(np.full((40, 256, 2), 1024.), np.full((40, 256), count))]

        def generate(self, prompts, params):
            assert len(prompts) == 256 and params.max_tokens == 1
            events.append('generate')
            if failure == 'generate':
                raise RuntimeError('generation failed')

    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(
        __version__='0.25.0', LLM=Engine, SamplingParams=lambda **kw: SimpleNamespace(**kw)))
    monkeypatch.setattr(torch.version, 'hip', 'test')
    titles, output = tmp_path / 'titles.json', tmp_path / 'capture.npz'
    titles.write_text(json.dumps(['title'] * 256))
    monkeypatch.setattr(sys, 'argv', ['capture', '--out', str(output)])
    from src.cli.capture_dsv4_rank import main
    monkeypatch.setitem(main.__globals__, 'PUBLIC_TITLES_PATH', titles)
    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            main()
        assert not output.exists()
    else:
        main()
        from src.analysis.rank_curve import rank_curve
        result = rank_curve(output)
        np.testing.assert_array_equal(result['response_mean'], np.full((2, 256), 2.))
        assert result['capture_metadata']['n_tokens'] == 512
    assert events[-1] == 'shutdown' and events.count('shutdown') == 1
