# Router–Expert Geometry

Code for “Routers Learn the Geometry of Their Experts: Geometric Coupling in Sparse Mixture-of-Experts.”

Model weights and evaluation datasets are not bundled. The [validation ledger](validation/result-comparison.md)
distinguishes retained measurements, fresh reproductions and unresolved results.
A passing unit test is not a full-model reproduction. A repository license has not yet been selected.

## Setup

Use Python 3.12 and uv from the repository root.

```bash
uv sync
uv sync --extra models --extra public-models --extra plots
uv run --extra models --extra public-models --extra plots pytest -q
```

Extras select dependencies per experiment:

| Extra | Use |
|---|---|
| `models` | Paper models and native OLMo-core checkpoints |
| `public-models` | Public models through Transformers |
| `steering` | OLMo-core steering with public OLMoE weights |
| `plots` | Figure rendering |

OLMo-core is unmodified upstream commit `60930ef14c8e4fb93eec071d44e104324d5c1c30`,
pinned through `tool.uv.sources`. Do not use `--no-sources`.
Paper-specific routers extend upstream classes in `src/routers/`; no personal fork is installed.

On the cluster, reuse a public Docker image rather than creating a virtual environment.
For steering, use
`pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel@sha256:53ab3de62f6101d1e42f9be28623ab7a468a24c070d632f211ed576e30b6abd3`.
Install into the container with
`uv pip install --system --break-system-packages -e '.[steering]'`.
Under Slurm, launch with `srun --kill-on-bad-exit=1`. Request only the GPUs needed for the experiment.

## Steering

`historical-causal` loads public OLMoE weights into the upstream
`TransformerConfig.olmoe_1B_7B()` factory. Normalization is not overridden:
the factory uses post-attention/post-MoE RMSNorm with epsilon `1e-6`.
RoPE theta is the factory default `500000`; neither RoPE nor normalization has a CLI override.
Evaluation uses FP32 weights and expert arithmetic with TF32 disabled.
Weights are read directly from the checkpoint's indexed Safetensors shards.
Upstream conversion handles packing; local mappings correct OLMoE norm naming
and dropless gate/up layout. No temporary Transformers model is constructed.

This is the **OLMo-core preset with public OLMoE weights**, not the released
checkpoint's original architecture. The separate `public-causal` command
evaluates the native Transformers model.

```bash
uv run --locked --extra steering historical-causal \
  --checkpoint-step 50000 --eval-root /data/causal-eval \
  --condition-set all_controls --out /results/steering.json
```

The evaluation root contains eleven corpus directories, each with
`val/part-0-00000.npy`. Each corpus contributes four calibration sequences and
sixteen evaluation sequences of 512 next-token predictions. These are corpus
validation sequences, not Wikipedia-title prompts.

- `all_controls` evaluates router removal and all 32 orthogonal controls at every layer (0–15).
- `pilot` (default) evaluates half/full removal, calibrated controls and input-norm restoration at layer 15.
- Directions are unit vectors. Controls are constructed from gate/up response energy in the router's orthogonal complement.
- Steering changes dispatched expert inputs after routing. Target-layer expert IDs and mixture weights must remain unchanged.
- Unchanged prefixes are cached; cached/full-forward parity is checked on fixed samples.
- Rank 1 means the largest mean loss increase among the 33 interventions. Norm-restored comparisons are separate.
- Results record per-sequence losses, token hashes, model configuration and checks. Existing outputs are not overwritten.

Do not combine results from different configurations into one claim.
See [steering evidence and current status](validation/STEERING_CLAIM.md).

```bash
uv run --locked --extra steering pytest -q tests/test_historical_causal.py tests/test_public_causal.py
```

## Original 1B Figure 2 and training dynamics

```bash
uv run --extra models capture-1b \
  --checkpoint /checkpoints/model_and_optim --routing loss_free \
  --tokens fixed-title-tokens.npz --out capture.npz
uv run figure2 capture.npz --response w1_top10_l2
uv run figure2-dynamics --captures raw-top10 --out dynamics.json
uv run --extra plots plot-figure2-dynamics dynamics.json --out dynamics.pdf
```

Capture uses the last non-padding token. `w1_top10_l2` is the norm of the ten
largest signed gate preactivations, not the full gate-vector norm. The original
reduction standardizes within layer/expert over tokens before pooling pairs for
Spearman correlation. Routed and non-routed pairs are separate.
Dynamics expects retained captures at steps 0, 1k, 5k, 10k, 20k and 39750.
See [measurement definitions and validation](validation/figure2-dynamics.md).

## Public-model responses

Prepare the [fixed title set](data/README.md), then capture with a pinned model revision.

```bash
uv run --extra public-models prepare-public-titles --model olmoe --out titles.npz
uv run --extra public-models capture-public-olmoe \
  --checkpoint-step 5000 --tokens titles.npz --out olmoe.npz
uv run olmoe-layers olmoe.npz
```

Public OLMoE uses native Transformers normalization and routing. Responses are
full gate/up projection L2 norms. Its layer reducer computes correlations across
the selected experts for each token/layer; this is not the original 1B pooled
statistic. Undefined correlations remain NaN and require valid-count reporting.
See [OLMoE validation](validation/public-olmoe.md).

To reproject saved hidden inputs without rerunning the transformer:

```bash
uv run --extra public-models olmoe-saved-inputs retained-capture.npz --out responses.npz
uv run olmoe-layers responses.npz
```

For Qwen and DeepSeek capture/plot commands, use `capture-qwen-rank --help`,
`capture-dsv4-rank --help` and `plot-public-rank --help`.
DeepSeek requires the GPU runtime described in [public-rank validation](validation/public-rank.md).
Saved-input reprojection and fresh full-model inference have different validation scopes.

## Native public-model causal evaluation

```bash
uv run --extra public-models public-causal \
  --checkpoint-step 50000 --eval-root /data/causal-eval --out public-causal.json
uv run summarize-public-causal public-causal.json
```

This uses native Transformers layers, preserves target-layer routing, and
compares router removal with 32 orthogonal controls at five layers.
It is a different forward configuration from the OLMo-core steering experiment.
The bootstrap comparison is conditional on the strongest observed control, not
a simultaneous 32-control interval. See [results and duplicate-input sensitivity](validation/public-intervention.md).

## Paper-model training and router geometry

The four routing recipes are `aux_loss`, `loss_free`, `seq_aux` and `kmeans`.
They use OLMo-core factories; K-means and sequence-wise auxiliary routing are
implemented in this package.

```bash
uv run --extra models torchrun --standalone --nproc_per_node=4 --no-python \
  train-1b --routing loss_free --data-root /data/tokenized \
  --eval-root /data --cache-dir /local/cache --save-folder /checkpoints/loss-free
uv run --extra models router-geometry /checkpoints/model_and_optim
```

Use `train-1b --print-config` to inspect the complete recipe.
K-means uses cosine scores and learning-rate-scaled centroid/bias updates.
Optimizer choices and resume checks are documented in [training validation](validation/training.md).
Small-model tests do not establish paper-scale training-trajectory parity.

`router-geometry` reports cosine statistics over unique unordered router-row
pairs, excluding the diagonal. Standard deviation is pairwise spread, not a
confidence interval. The separate `train-geometry-7b` comparison uses different
optimizer/LR recipes; see [7B geometry validation](validation/router-geometry-7b.md).

## Expert specialization

```bash
uv run --extra models capture-domain-counts \
  --checkpoint /checkpoints/step22000/model_and_optim --routing loss_free \
  --sequences eval-sequences.npz --out domain-counts.npz
uv run --extra models capture-ablation \
  --checkpoint /checkpoints/step22000/model_and_optim \
  --sequences eval-sequences.npz --counts domain-counts.npz --out code-ablation.npz
```

`specialist-ablation` reduces code-domain clean/ablated losses;
`specialist-domains` runs the cross-domain top-four-versus-random comparison.
Routing-file and sequence hashes must agree. A masked expert being reselected
aborts capture: masking logits does not guarantee exclusion under selection bias.
The historical damage ratio is an NLL ratio, not a perplexity ratio.
See [verified results and the remaining loss-free exclusion issue](validation/specialist.md).

## K-means error breakdown

```bash
uv run prepare-error-sequences --eval-root /data/validation --out sequences.npz
uv run --extra models capture-token-losses \
  --checkpoint /checkpoints/kmeans/step21000/model_and_optim --routing kmeans \
  --sequences sequences.npz --out kmeans-losses.npz
# Repeat capture with the loss-free checkpoint and --routing loss_free.
uv run kmeans-errors --kmeans kmeans-losses.npz --reference loss-free-losses.npz
```

The comparison uses identical next-token targets. Frequency is measured in the
evaluation corpus; boundary margin is the biased selection-score gap between
ranks 6 and 7, standardized per layer and averaged over layers. Ties remain
together. This measures loss differences, not ground-truth routing mistakes.
See [K-means validation](validation/kmeans-errors.md) and the [comparison ledger](validation/result-comparison.md).
