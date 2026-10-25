"""Retained ROCm worker workaround for fixed-cache, eager DeepSeek inference."""

from vllm.model_executor.layers import mhc
from vllm.v1.worker.gpu_worker import Worker

# The retained MI325X run used the Torch/Triton fallback, not AITER mHC.
mhc.HAS_AITER_MHC = False


class FixedCacheWorker(Worker):
    def determine_available_memory(self) -> int:
        size = self.cache_config.kv_cache_memory_bytes
        if not size or not self.model_config.enforce_eager:
            raise ValueError('this experiment requires explicit KV memory and eager execution')
        return int(self._reserve_mm_ipc_gpu_memory(size))
