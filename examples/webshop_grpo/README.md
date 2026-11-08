# Qwen3.5 WebShop GRPO baseline

Local integration based on official verl commit `b17218ba58a6fbcdfb0de32772e3c2a79f88529d`.
It trains original Qwen3.5-9B with free-form WebShop actions, **no thinking, no Jev readout, and no latent reasoning**.

## Changes

- Custom official asynchronous AgentLoop with one causal serialized trajectory per episode. Generated action tokens have loss mask 1; environment observations and inserted template context have mask 0.
- Separate CPU HTTP service loads the verified original WebShop catalog once. UUID browser sessions are pinned to canonical train/dev tasks. Held-out test tasks are rejected, including initial and terminal resets.
- Explicit `VERL_QWEN35_NATIVE_GDN=1` opt-in uses unwrapped native Transformers PyTorch DeltaNet kernels and independently splits packed samples. Sequence parallelism greater than 1 is rejected. Default kernel selection is unchanged when unset.
- Qwen3.5 fused LM-head forward uses public FSDP2 forward-method registration, so CPU-offloaded child weights participate in normal unshard/reshard and backward hooks. Torch and Triton fused-head paths are covered.
- Strict custom sampler requires 16 task groups with 8 completed transport-success trajectories each. Missing, failed, empty, duplicated, or padded outputs abort before training; no replacement/resampling is used.
- Independent tmux launchers, local per-episode JSONL traces, checkpoints, timestamped logs, and completion status files.

## Runtime and assets

Validated installed stack: Python 3.12, Torch 2.13.0+cu130, Transformers 5.16.1, vLLM 0.32.0, FlashAttention 2.8.3. Eight A100-SXM4-80GB GPUs; FSDP2 actor/reference with SP=1 and four TP=2 vLLM replicas.

Use `uv` and the upstream lockfile for the trainer environment:

```bash
uv sync --frozen --extra fsdp --extra vllm --python 3.12
```

The CPU WebShop service uses a separate original-WebShop-compatible environment, Java/Lucene, and spaCy. This example depends on the existing workspace's vendored WebShop, verified assets, evaluation helpers, and frozen manifest; it does not download or bundle them. See `run_environment.sh` and `prepare_data.py` for explicit inputs. Defaults assume workspace `/home/aiscuser/shiqiyan`; override `WS` and `ROOT` to relocate.

Fixed manifest ID: `d1f421e9d7fc6f9af1223dca054391f039e62e572324b264d64f3068419223b2`.
Train: 11,286 goals; dev: 100; test: 500 (not consumed by training or validation). The manifest and data remain outside this repository.

## Protocol

- 150 global updates, 16 tasks × 8 trajectories = 128 actual episodes/update.
- 64-trajectory optimizer minibatches; in this upstream trainer the configured minibatch is 8 task groups, multiplied by rollout.n=8.
- Learning rate 1e-6, KL loss coefficient 0.01, entropy coefficient 0.001.
- Maximum 15 actions and 512 generated tokens/action.
- Initial prompt budget 4096; serialized response budget 24576, including inserted observations; vLLM context ceiling 28672.
- No silent observation truncation. Episodes stop before the serialized budget is exceeded; traces record the stop reason.
- Original terminal exact-success reward 10, otherwise 0, minus 0.1 per malformed action.
- Validate on all 100 dev tasks before training and every 5 updates; n=1, temperature=0.4. Save checkpoints every 5 updates.

**History differs from the legacy verl-agent run:** this integration retains the full causal transcript rather than last-two-only history, preserving agreement between rollout prefixes and training. It adds no separate public-memory augmentation. Compare evaluations only after accounting for this protocol difference.

## Launch

Prepare parquet once using `prepare_data.py --manifest PATH --output-dir PATH`. Existing outputs are not overwritten. The launcher expects `webshop-runtime/grpo-data-upstream-b17218ba/{train,dev}.parquet` under `WS`.

```bash
tmux new-session -d -s webshop-env-upstream 'bash examples/webshop_grpo/run_environment.sh'
# Wait for GET http://127.0.0.1:8093/health to report ready.
tmux new-session -d -s webshop-grpo-upstream 'bash examples/webshop_grpo/run_persistent.sh smoke trainer.test_freq=-1'
# After successful smoke exit and checkpoint verification, launch from base weights:
tmux new-session -d -s webshop-grpo-formal 'bash examples/webshop_grpo/run_persistent.sh train'
```

Alternatively, `run_formal_after_smoke.sh` takes the **exact active smoke status file** and gates formal launch on successful exit and a complete 8-rank checkpoint. Run it in its own tmux session. An engineering smoke checkpoint is not used to initialize the formal experiment.

Logs/status: `WS/runs/webshop_grpo/launch-upstream-b17218ba/`.
Trajectories: `WS/runs/webshop_grpo/EXPERIMENT/trajectories/`.
Checkpoints: `WS/checkpoints/webshop_grpo/EXPERIMENT/`.
All training telemetry is local (console, offline W&B environment); no remote inference calls.

## Tests and observed execution

```bash
.venv/bin/python -m unittest examples.webshop_grpo.test_webshop_agent examples.webshop_grpo.test_strict_sampler -q
.venv/bin/python -m pytest -q tests/models/test_qwen35_native_gdn_on_cpu.py tests/models/test_fused_kernels_ulysses_sp_on_cpu.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q tests/models/test_qwen35_fused_lm_head_fsdp_on_gpu.py
```

Results: 25 environment/sampler tests, 44 model CPU tests, and 2 real GPU FSDP2 CPU-offload fused-head tests passed. GPU tests cover no-grad log-probs plus repeated backward microbatches against an unsharded reference. A real 128-client original-environment create/close check passed after increasing the HTTP accept backlog to 256.

Engineering smoke completed an optimizer update and saved all model/optimizer/extra-state shards, dataloader state, config/tokenizer, and atomic completion tracker. That smoke encountered HTTP resets before the backlog fix, so it is **not** evidence of a full 128-episode scientific update. The formal run starts original weights with the strict sampler enabled. Smoke step duration was approximately 336 seconds; this is not a guarantee of formal-run throughput.

Native reference DeltaNet may be slow for long sequences. Modern HF cached generation is outside the patch's verified scope; actor/reference training uses no cache and rollout uses vLLM. Model weights, catalog, indexes, manifests, trajectories, logs, checkpoints, and virtual environments are intentionally not committed.
