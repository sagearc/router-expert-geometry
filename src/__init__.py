"""Measurements for Routers Learn the Geometry of Their Experts.

The package is layered by dependency so the analysis code installs without
model libraries:

- ``constants``, ``types`` and ``artifacts``: shared identities, array types and
  artifact schemas (NumPy only).
- ``analysis``: the statistical reductions and plots' inputs (NumPy and SciPy).
- ``routers``: the paper's router variants on OLMo-core's router base and
  their trainer glue (``models`` extra).
- ``configs``: model, optimizer and trainer configurations for the 1B routers
  and the 6.9B recipes (``models`` extra).
- ``capture``: measurements on the paper's OLMo-core checkpoints: last-token
  captures, losses, ablations and router geometry (``models`` extra).
- ``hf``: captures and interventions on public OLMoE, Qwen and DeepSeek
  checkpoints through Transformers or vLLM (``public-models`` extra).
"""
