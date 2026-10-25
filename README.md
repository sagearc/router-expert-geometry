# Router–Expert Geometry

Code for “Routers Learn the Geometry of Their Experts: Geometric Coupling in
Sparse Mixture-of-Experts.”

This repository contains the paper/rebuttal experiments and requested
camera-ready extensions. [Scope](validation/scope.md) maps them to the claims.
[Result verification](validation/result-comparison.md) distinguishes reproduced
measurements from unresolved results. Weights and evaluation corpora are not
bundled; a repository license has not yet been selected.

## Setup

Use Python 3.12 and uv from the repository root.

```bash
uv sync --locked --extra models --extra public-models --extra plots
uv run --locked --extra models --extra public-models --extra plots pytest -q
```

Extras are `models` (paper checkpoints/training), `public-models`
(Transformers response and steering experiments), and `plots`.
OLMo-core is unmodified upstream commit
`60930ef14c8e4fb93eec071d44e104324d5c1c30`; local routers extend it.
Do not install a personal fork or use `--no-sources`.

On clusters, reuse public Docker images, not new virtual environments.
The steering runtime is
`pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel@sha256:53ab3de62f6101d1e42f9be28623ab7a468a24c070d632f211ed576e30b6abd3`.
Inside that container, install with
`uv pip install --system --break-system-packages -e '.[public-models]'`.
Under Slurm, use `srun --kill-on-bad-exit=1` and request only needed GPUs.
DeepSeek has separate [runtime requirements](validation/public-rank.md).

## Router-axis steering

```bash
uv run --locked --extra public-models public-causal \
  --checkpoint-step 50000 --eval-root /data/causal-eval --out steering.json
```

Four fixed interventions versus 32 weight-derived orthogonal controls,
after dispatch and before expert gate/up projections. Rank 1 is
the highest mean CE among 33 interventions. All 16 layers use the same 176
validation sequences and unchanged target-layer routing; cached/full-forward
parity is checked.

The four edits are full projection removal, 5%/10% input-norm displacement,
and a 0.1-radian norm-preserving rotation. They are fixed in the protocol,
not optional architecture or exploratory-strength flags. Use
`summarize-public-causal steering.json` for the 16-layer result table.

Load the public checkpoint directly through Transformers, without OLMo-core
conversion or architecture overrides. Read normalization, RoPE and epsilon
from the checkpoint. See [protocol and results](validation/STEERING_CLAIM.md).
Historical factory-forward measurements remain in the external audit archive,
not as a second released model configuration.

## Figure 2, dynamics and identity controls

```bash
uv run --extra models capture-1b \
  --checkpoint /checkpoints/model_and_optim --routing loss_free \
  --tokens fixed-title-tokens.npz --out capture.npz
uv run figure2 capture.npz --response w1_top10_l2
uv run figure2-dynamics --captures raw-top10 --out dynamics.json
uv run --extra plots plot-figure2-dynamics dynamics.json --out dynamics.pdf
```

Capture uses last non-padding tokens. Signed top-10 gate L2 and mean-SiLU
responses are distinct measurements. Dynamics uses within-layer/expert
standardization followed by pooled Spearman, with expert-identity and token
shuffle controls. See [definitions and evidence](validation/figure2-dynamics.md).

## Public-model responses and depth

```bash
uv run --extra public-models prepare-public-titles --model olmoe --out titles.npz
uv run --extra public-models capture-public-olmoe \
  --checkpoint-step 5000 --tokens titles.npz --out olmoe.npz
uv run olmoe-layers olmoe.npz
```

Use the [fixed title set](data/README.md). OLMoE measures full gate/up L2;
its layer reducer averages per-title correlations across selected experts,
not the original pooled 1B statistic. `olmoe-saved-inputs` reprojects retained
hidden inputs without a fresh forward pass.
[OLMoE methods and validation](validation/public-olmoe.md).

`capture-qwen-rank`, `capture-dsv4-rank` and `plot-public-rank`
produce gate/up magnitude versus per-token expert rank.
[Rank definitions, runtimes and limits](validation/public-rank.md).

## Training and router geometry

`train-1b` implements `aux_loss`, `loss_free`, `seq_aux` and `kmeans`.
`train-geometry-7b` implements the two retained 6.9B recipes.
Use `--print-config` to inspect settings and `--help` for data/output paths.

```bash
uv run --extra models torchrun --standalone --nproc_per_node=4 --no-python \
  train-1b --routing loss_free --data-root /data/tokenized --eval-root /data \
  --cache-dir /local/cache --save-folder /checkpoints/loss-free
uv run --extra models router-geometry /checkpoints/model_and_optim
```

Router geometry uses unique off-diagonal cosine pairs.
[Training validation](validation/training.md) and
[6.9B recipe differences](validation/router-geometry-7b.md).

## Specialist ablation and K-means errors

Prepare common validation sequences with `prepare-error-sequences`.

- Specialization: `capture-domain-counts` → `capture-ablation` /
  `specialist-domains` → `specialist-ablation`.
  [Protocol and unresolved loss-free exclusion](validation/specialist.md).
- K-means errors: run `capture-token-losses` on both step-21k checkpoints,
  then `kmeans-errors --kmeans kmeans.npz --reference loss-free.npz`.
  [Pairing, boundary definition and results](validation/kmeans-errors.md).

These report NLL changes, not perplexity ratios or known routing mistakes.
Failed validation aborts; do not skip failing batches.
