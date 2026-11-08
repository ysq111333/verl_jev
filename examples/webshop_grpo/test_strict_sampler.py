"""stdlib tests using actual upstream ReplayBuffer code and stubbed transports."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]


def load_sampler():
    """Run upstream materialization/sampling without importing Ray, Torch or TQ."""
    queue = ModuleType("transfer_queue")
    queue.kv_clear = Mock()
    queue.KVBatchMeta = lambda **kwargs: SimpleNamespace(**kwargs)
    numpy = ModuleType("numpy")
    omega = ModuleType("omegaconf")
    omega.DictConfig = dict
    skip = ModuleType("verl.utils.skip")
    skip.SkipManager = SimpleNamespace(annotate_tq=lambda **kwargs: lambda function: function)
    name = "verl.trainer.ppo.v1.replay_buffer"
    spec = importlib.util.spec_from_file_location(name, ROOT / "verl/trainer/ppo/v1/replay_buffer.py")
    replay = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {"transfer_queue": queue, "numpy": numpy, "omegaconf": omega, "verl.utils.skip": skip, name: replay},
    ):
        spec.loader.exec_module(replay)
        spec = importlib.util.spec_from_file_location("_strict_webshop_sampler", DIRECTORY / "strict_sampler.py")
        custom = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom)
    return custom.StrictWebShopReplayBuffer, replay, queue


StrictSampler, replay, queue = load_sampler()


class StrictSamplerTests(unittest.TestCase):
    def setUp(self):
        queue.kv_clear.reset_mock()
        self.refill = Mock(side_effect=AssertionError("refill/resampling is prohibited"))
        self.sampler = self.make_sampler()
        self.uids = [f"group{index:02d}" for index in range(16)]
        self.tags = {
            f"{uid}_{session}_0": {"status": "success", "response_len": 10, "seq_len": 20, "global_steps": 1}
            for uid in self.uids
            for session in range(8)
        }
        self.sampler.finished_keys["train"].update(self.uids)

    def make_sampler(self, **overrides):
        kwargs = dict(
            trainer_mode="sync",
            trainer_config={},
            max_off_policy_threshold=None,
            max_off_policy_strategy="drop",
            sampler_kwargs={},
            refill_fn=self.refill,
        )
        kwargs.update(overrides)
        return StrictSampler(**kwargs)

    def check_rejected(self, selected=None, tags=None):
        with self.assertRaises(RuntimeError):
            self.sampler._materialize_batch(
                "train", selected if selected is not None else self.uids, tags if tags is not None else self.tags
            )
        queue.kv_clear.assert_not_called()
        self.refill.assert_not_called()

    def test_complete_16_by_8_materializes_exactly_once(self):
        result = self.sampler._materialize_batch("train", self.uids, self.tags)
        self.assertEqual(len(result.keys), 128)
        self.assertEqual(set(result.keys), set(self.tags))
        self.assertEqual(len(result.tags), 128)
        queue.kv_clear.assert_called_once_with(partition_id="train", keys=self.uids)
        self.refill.assert_not_called()

    def test_partial_seven_of_eight_rejected_before_clear(self):
        del self.tags[f"{self.uids[0]}_7_0"]
        self.check_rejected()

    def test_failed_group_rejected_even_with_all_eight_outputs(self):
        self.sampler.finished_keys["train"].remove(self.uids[0])
        self.sampler.failure_keys["train"].add(self.uids[0])
        self.check_rejected()

    def test_padding_rejected(self):
        self.tags[f"{self.uids[0]}_0_0"]["is_padding"] = True
        self.check_rejected()

    def test_extra_session_and_extra_output_rejected(self):
        for extra_key in (f"{self.uids[0]}_8_0", f"{self.uids[0]}_0_1"):
            with self.subTest(extra_key=extra_key):
                tags = dict(self.tags)
                tags[extra_key] = {"status": "success", "response_len": 2}
                self.check_rejected(tags=tags)

    def test_wrong_or_duplicate_group_count_rejected(self):
        for selected in (self.uids[:-1], self.uids + ["extra"], self.uids[:-1] + [self.uids[0]]):
            with self.subTest(selected=selected):
                self.check_rejected(selected=selected)

    def test_transport_failure_or_empty_response_rejected(self):
        key = f"{self.uids[0]}_0_0"
        for changes in (
            {"status": "failure"},
            {"response_len": 0},
            {"response_len": -1},
            {"response_len": None},
            {"response_len": "10"},
        ):
            with self.subTest(changes=changes):
                tags = dict(self.tags)
                tags[key] = {**tags[key], **changes}
                self.check_rejected(tags=tags)

    def test_validation_unchanged_even_for_partial_failed_group(self):
        tags = {"dev_0_0": {"status": "failure", "response_len": 0, "is_padding": True}}
        result = self.sampler._materialize_batch("val", ["dev"], tags)
        self.assertEqual(result.keys, ["dev_0_0"])
        self.assertEqual(result.tags, [tags["dev_0_0"]])
        queue.kv_clear.assert_called_once_with(partition_id="val", keys=["dev"])
        self.refill.assert_not_called()

    def test_sample_rejects_wrong_groups_before_any_upstream_poll(self):
        with patch.object(replay.ReplayBuffer, "sample") as upstream:
            with self.assertRaises(RuntimeError):
                self.sampler.sample(global_steps=1, partition_id="train", batch_size=4)
            upstream.assert_not_called()
            self.sampler.sample(global_steps=1, partition_id="val", batch_size=100)
            upstream.assert_called_once_with(1, "val", 100)
        queue.kv_clear.assert_not_called()
        self.refill.assert_not_called()

    def test_upstream_sample_fails_partial_batch_without_resampling(self):
        del self.tags[f"{self.uids[0]}_7_0"]

        def sync_metadata():
            self.sampler.partitions["train"] = dict(self.tags)
            self.sampler.prompt_global_steps["train"] = {uid: 1 for uid in self.uids}

        with patch.object(self.sampler, "_sync_metadata_from_transfer_queue", side_effect=sync_metadata):
            with self.assertRaises(RuntimeError):
                self.sampler.sample(global_steps=1, partition_id="train", batch_size=16)
        queue.kv_clear.assert_not_called()
        self.refill.assert_not_called()

    def test_constructor_forbids_async_filtering_refill_and_shape_changes(self):
        for overrides in (
            {"trainer_mode": "separate_async"},
            {"filter_groups_metric": "reward"},
            {"sync_refill_failed_groups": True},
            {"sampler_kwargs": {"n": 7}},
            {"train_batch_size": 15},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    self.make_sampler(**overrides)
        queue.kv_clear.assert_not_called()
        self.refill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
