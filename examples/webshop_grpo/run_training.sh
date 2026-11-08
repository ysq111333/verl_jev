#!/usr/bin/env bash
# Official upstream AgentLoop + vLLM; original WebShop environment via CPU HTTP service.
set -euo pipefail
ROOT=${ROOT:-/home/aiscuser/shiqiyan/verl-upstream-latest}
WS=${WS:-/home/aiscuser/shiqiyan}
PY="$ROOT/.venv/bin/python"
export PATH="$ROOT/.venv/bin:$WS/.venv-webshop-download/bin:$PATH"
export VERL_QWEN35_NATIVE_GDN=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export WEBSHOP_ENV_URL=http://127.0.0.1:8093
MODE=${1:-smoke}
shift || true
EXPERIMENT=upstream_webshop_qwen35_native_gdn_v1
STEPS=150
VAL=True
if [[ "$MODE" == smoke ]]; then
 EXPERIMENT=upstream_webshop_qwen35_native_gdn_smoke_v1
 STEPS=1
 VAL=False
elif [[ "$MODE" != train ]]; then
 printf 'Usage: %s smoke|train [Hydra overrides]\n' "$0" >&2
 exit 1
fi
export WEBSHOP_TRACE_DIR="$WS/runs/webshop_grpo/$EXPERIMENT/trajectories"
"$PY" -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
 data.train_files="$WS/webshop-runtime/grpo-data-upstream-b17218ba/train.parquet" \
 data.val_files="$WS/webshop-runtime/grpo-data-upstream-b17218ba/dev.parquet" \
 data.train_batch_size=16 data.val_batch_size=16 \
 data.max_prompt_length=4096 data.max_response_length=24576 \
 data.filter_overlong_prompts=False data.truncation=error data.shuffle=True \
 +data.apply_chat_template_kwargs.enable_thinking=False \
 actor_rollout_ref.model.path="$WS/downloads/webshop-choice-20261105/qwen35-9b" \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.model.use_fused_kernels=True \
 actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
 actor_rollout_ref.actor.strategy=fsdp2 \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=8 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.actor.use_dynamic_bsz=True \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=28672 \
 actor_rollout_ref.actor.use_torch_compile=False \
 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.01 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0.001 \
 actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
 actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
 actor_rollout_ref.actor.fsdp_config.param_offload=True \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
 actor_rollout_ref.actor.fsdp_config.offload_policy=True \
 actor_rollout_ref.ref.strategy=fsdp2 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
 actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=28672 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 actor_rollout_ref.ref.fsdp_config.offload_policy=True \
 actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=1 \
 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
 actor_rollout_ref.rollout.n=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
 actor_rollout_ref.rollout.max_model_len=28672 \
 actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
 actor_rollout_ref.rollout.max_num_seqs=32 \
 actor_rollout_ref.rollout.enable_chunked_prefill=True \
 actor_rollout_ref.rollout.enable_prefix_caching=False \
 actor_rollout_ref.rollout.free_cache_engine=True \
 actor_rollout_ref.rollout.enforce_eager=False \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.rollout.multi_turn.enable=True \
 actor_rollout_ref.rollout.agent.num_workers=8 \
 actor_rollout_ref.rollout.agent.default_agent_loop=webshop \
 actor_rollout_ref.rollout.agent.agent_loop_config_path="$ROOT/examples/webshop_grpo/agent_loop.yaml" \
 actor_rollout_ref.rollout.val_kwargs.n=1 \
 actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
 actor_rollout_ref.rollout.val_kwargs.do_sample=True \
 trainer.project_name=webshop_grpo trainer.experiment_name="$EXPERIMENT" \
 trainer.n_gpus_per_node=8 trainer.nnodes=1 \
 trainer.total_epochs=150 trainer.total_training_steps="$STEPS" \
 trainer.v1.trainer_mode=sync \
 +trainer.v1.sync.parameter_sync_step=1 \
 trainer.v1.sampler.sync_refill_failed_groups=False \
 'trainer.v1.sampler.sampler_kwargs={}' \
 algorithm.filter_groups.enable=False \
 trainer.v1.sampler.custom_sampler.path="$ROOT/examples/webshop_grpo/strict_sampler.py" \
 trainer.v1.sampler.custom_sampler.name=StrictWebShopReplayBuffer \
 trainer.val_before_train="$VAL" trainer.test_freq=5 trainer.save_freq=5 \
 trainer.logger='[console]' trainer.resume_mode=auto \
 trainer.default_local_dir="$WS/checkpoints/webshop_grpo/$EXPERIMENT" \
 trainer.rollout_data_dir="$WS/runs/webshop_grpo/$EXPERIMENT/rollouts" \
 ray_kwargs.ray_init.num_cpus=48 \
 ray_kwargs.ray_init.runtime_env.py_executable="$PY" \
 '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_QWEN35_NATIVE_GDN="1"' \
 +ray_kwargs.ray_init.runtime_env.env_vars.WEBSHOP_ENV_URL="$WEBSHOP_ENV_URL" \
 +ray_kwargs.ray_init.runtime_env.env_vars.WEBSHOP_TRACE_DIR="$WEBSHOP_TRACE_DIR" \
 '+ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE="1"' \
 "$@"
