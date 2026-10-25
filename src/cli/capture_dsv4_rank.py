"""DeepSeek-V4-Flash all-token rank responses on the retained ROCm backend.

Run inside a pinned public ROCm/vLLM container under srun --kill-on-bad-exit=1.
The required container/kernel combination is not yet runtime-validated here.
"""

import argparse
import hashlib
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from src.constants import (
    DSV4_EXPERTS,
    DSV4_HASH_LAYERS,
    DSV4_MODEL_ID,
    DSV4_RESPONSE_METRICS,
    DSV4_REVISION,
    DSV4_SCORE_LAYERS,
    DSV4_TOP_K,
    DSV4_VLLM_VERSION,
    PUBLIC_MAX_TOKENS,
    PUBLIC_TITLE_COUNT,
    PUBLIC_TITLES_PATH,
)
from src.hf.dsv4 import finish_capture, install_capture
from src.types import Float64Array, Int64Array

if TYPE_CHECKING:
    from torch.nn import Module
    from vllm.inputs import TokensPrompt


class Args:
    out: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(namespace=Args())
    titles = json.loads(PUBLIC_TITLES_PATH.read_text())
    if (args.out.exists() or not isinstance(titles, list) or len(titles) != PUBLIC_TITLE_COUNT
            or any(not isinstance(x, str) or not x for x in titles)):
        parser.error('requires 256 nonempty title strings and a new output file')
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    if not torch.version.hip or vllm.__version__ != DSV4_VLLM_VERSION:
        raise ValueError('requires the retained ROCm vLLM 0.25.0 API; container validation pending')
    llm = LLM(
        model=DSV4_MODEL_ID, revision=DSV4_REVISION, tokenizer_revision=DSV4_REVISION,
        tensor_parallel_size=1, distributed_executor_backend='mp',
        worker_cls='src.hf.dsv4_worker.FixedCacheWorker',
        trust_remote_code=True, tokenizer_mode='deepseek_v4', max_model_len=4096,
        max_num_batched_tokens=256, max_num_seqs=256, gpu_memory_utilization=0.85,
        kv_cache_memory_bytes=10_000_000_000, kv_cache_dtype='fp8_e4m3',
        enforce_eager=True, disable_log_stats=True, enable_prefix_caching=False,
        moe_backend='triton_unfused', linear_backend='triton',
    )
    try:
        tokenizer = llm.get_tokenizer()
        tokens = [tokenizer.encode(title, add_special_tokens=False)[:PUBLIC_MAX_TOKENS] for title in titles]
        tokens = [row or [tokenizer.eos_token_id] for row in tokens]
        count = sum(map(len, tokens))
        llm.apply_model(partial(install_capture, total_prompt_tokens=count))
        prompts: list[TokensPrompt] = [{'prompt_token_ids': row} for row in tokens]
        llm.generate(prompts,
                     SamplingParams(max_tokens=1, temperature=0., detokenize=False))
        results = llm.apply_model(cast('Callable[[Module], tuple[Float64Array, Int64Array]]', finish_capture))
        if len(results) != 1:
            raise ValueError('expected exactly one model worker')
        sums, counts = results[0]
        if (sums.shape != (len(DSV4_SCORE_LAYERS), DSV4_EXPERTS, 2) or counts.shape != (len(DSV4_SCORE_LAYERS), DSV4_EXPERTS)
                or not np.isfinite(sums).all() or (sums < 0).any()
                or not np.issubdtype(counts.dtype, np.integer) or not np.all(counts == count)):
            raise ValueError('incomplete or invalid worker response summaries')
        metadata = {'model': DSV4_MODEL_ID, 'revision': DSV4_REVISION,
                    'top_k': DSV4_TOP_K, 'token_scope': 'all', 'excluded_hash_moe_layers': DSV4_HASH_LAYERS,
                    'response_metrics': list(DSV4_RESPONSE_METRICS),
                    'rank_definition': 'descending sigmoid(router_logits) + correction_bias',
                    'vllm': vllm.__version__, 'torch': torch.__version__,
                    'n_tokens': count, 'prefix_caching': False,
                    'tokens_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                    'titles_sha256': hashlib.sha256(PUBLIC_TITLES_PATH.read_bytes()).hexdigest()}
        with args.out.open('xb') as stream:
            np.savez_compressed(stream, response_mean=sums / counts[..., None], rank_count=counts,
                                expert_rank=np.arange(1, DSV4_EXPERTS + 1), model_layer_indices=np.asarray(DSV4_SCORE_LAYERS),
                                metadata_json=json.dumps(metadata))
    finally:
        llm.llm_engine.engine_core.shutdown()
