"""Opt-in fail-closed WebShop sampler for formal 16 x 8 synchronous GRPO.

Use the official trainer.v1.sampler.custom_sampler.path/name hook. This sampler
never replaces a failed group or filters a reward. Validation remains unchanged.
A completed transport rollout can have zero shopping reward and is still valid.
"""

from __future__ import annotations

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer


class StrictWebShopReplayBuffer(ReplayBuffer):
    EXPECTED_GROUPS = 16
    EXPECTED_ROLLOUTS = 8

    def __init__(
        self,
        trainer_mode,
        trainer_config,
        max_off_policy_threshold,
        max_off_policy_strategy,
        sampler_kwargs,
        poll_interval=2.0,
        refill_fn=None,
        filter_groups_metric=None,
        train_batch_size=None,
        gen_batch_size=None,
        max_inflight_gen_batches=1,
        sync_refill_failed_groups=False,
    ):
        # Match the upstream ReplayBuffer constructor, but prohibit semantics
        # that could silently replace/drop the originally dispatched task groups.
        if trainer_mode != "sync":
            raise ValueError("Strict WebShop sampling requires trainer.v1.trainer_mode=sync")
        if filter_groups_metric is not None or sync_refill_failed_groups:
            raise ValueError("Strict WebShop sampling forbids filtering and failed-group refill")
        if sampler_kwargs:
            raise ValueError("Strict WebShop sampling is fixed at 16 groups x 8 rollouts; sampler_kwargs must be empty")
        if train_batch_size not in (None, self.EXPECTED_GROUPS):
            raise ValueError("Strict WebShop sampling requires 16 task groups per training batch")
        super().__init__(
            trainer_mode=trainer_mode,
            trainer_config=trainer_config,
            max_off_policy_threshold=max_off_policy_threshold,
            max_off_policy_strategy=max_off_policy_strategy,
            sampler_kwargs=sampler_kwargs,
            poll_interval=poll_interval,
            refill_fn=refill_fn,
            filter_groups_metric=None,
            train_batch_size=train_batch_size,
            gen_batch_size=gen_batch_size,
            max_inflight_gen_batches=max_inflight_gen_batches,
            sync_refill_failed_groups=False,
        )

    def sample(self, global_steps, partition_id, batch_size):
        if partition_id == "train" and batch_size != self.EXPECTED_GROUPS:
            raise RuntimeError("Strict WebShop requires 16 groups in each sample; use parameter_sync_step=1")
        return super().sample(global_steps, partition_id, batch_size)

    def _materialize_batch(self, partition_id, selected_prompt_uids, partition_snapshot):
        if partition_id == "train":
            selected = set(selected_prompt_uids)
            if len(selected_prompt_uids) != self.EXPECTED_GROUPS or len(selected) != self.EXPECTED_GROUPS:
                raise RuntimeError("Incomplete WebShop GRPO batch: require 16 unique task groups")
            if not selected <= self.finished_keys[partition_id] or selected & self.failure_keys[partition_id]:
                raise RuntimeError("Incomplete WebShop GRPO batch: every selected group must finish without errors")
            expected = {f"{uid}_{session}_0" for uid in selected for session in range(self.EXPECTED_ROLLOUTS)}
            actual = {key for key in partition_snapshot if key.split("_")[0] in selected}
            if actual != expected:
                raise RuntimeError(
                    "Incomplete WebShop GRPO batch: require exactly 8 single-output rollouts per group "
                    f"(missing={len(expected - actual)}, extra={len(actual - expected)})"
                )
            for key in expected:
                tag = partition_snapshot[key]
                if (
                    tag.get("status") != "success"
                    or tag.get("is_padding", False)
                    or not isinstance(tag.get("response_len"), int)
                    or tag["response_len"] <= 0
                ):
                    raise RuntimeError("Incomplete WebShop GRPO batch: transport failure, padding, or empty response")
        # Validate BEFORE upstream clears prompt status markers. No retry,
        # eviction, padding, filtering, refill, or replacement occurs here.
        return super()._materialize_batch(partition_id, selected_prompt_uids, partition_snapshot)
