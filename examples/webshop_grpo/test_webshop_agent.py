"""CPU/stdlib tests: python -m unittest discover -s examples/webshop_grpo -v.

Heavy verl/model modules are stubbed only while importing the custom loop; no
GPU, model runtime, original catalog, or dependency installs are needed.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
sys.path.insert(0, str(ROOT))
from examples.webshop_grpo import prepare_data  # noqa: E402
from examples.webshop_grpo import webshop_env_service as service_module  # noqa: E402


class FakeBase:
    def __init__(self, trainer_config, server_manager, tokenizer, **kwargs):
        self.rollout_config = trainer_config.config.actor_rollout_ref.rollout
        self.apply_chat_template_kwargs = trainer_config.config.data.apply_chat_template_kwargs
        self.tokenizer, self.server_manager = tokenizer, server_manager
        self.loop = asyncio.get_running_loop()

    async def ct_build_initial_tokens(self, messages):
        return self._cap_text_prompt_length(self.tokenizer.encode(json.dumps(messages)))

    async def ct_merge_assistant_token(self, runtime_ids, assistant_ids, mask, logprobs, assistant_logprobs=None):
        return (
            SimpleNamespace(token_ids=runtime_ids + assistant_ids),
            mask + [1] * len(assistant_ids),
            None if logprobs is None else logprobs + list(assistant_logprobs),
        )

    async def ct_merge_context_msg(self, previous_messages, messages, runtime_ids, mask, logprobs):
        assert messages[: len(previous_messages)] == previous_messages
        assert len(messages) == len(previous_messages) + 1
        assert previous_messages[-1]["role"] == "assistant"
        assert messages[-1]["role"] == "user"
        # As in CT: assistant text is already present as raw generated tokens;
        # append only the observation/boundary suffix, never retokenize actions.
        context = self.tokenizer.encode("\nUSER:" + messages[-1]["content"] + "\nASSISTANT:")
        return (
            SimpleNamespace(token_ids=runtime_ids + context),
            mask + [0] * len(context),
            None if logprobs is None else logprobs + [0.0] * len(context),
        )


class FakeOutput:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def load_agent():
    module = ModuleType("verl.experimental.agent_loop.agent_loop")
    module.AgentLoopBase, module.AgentLoopOutput = FakeBase, FakeOutput
    module.registry = {}

    def register(name):
        def decorate(cls):
            module.registry[name] = cls
            return cls

        return decorate

    module.register = register
    spec = importlib.util.spec_from_file_location("_stdlib_webshop_agent", DIRECTORY / "webshop_agent.py")
    agent = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {module.__name__: module}):
        spec.loader.exec_module(agent)
    return agent, module.registry


agent, registry = load_agent()


def manifest_fixture():
    tasks = [
        {"task_id": f"{index:064x}", "instruction": f"public task {index}", "split": split}
        for index, split in enumerate(["train", "train", "dev", "test"], 1)
    ]
    result = {
        "protocol": "webshop-choice-v1",
        "tasks": tasks,
        "goals_hash": service_module.digest([task["task_id"] for task in tasks]),
        "provenance": {"seed": 0},
    }
    result["manifest_id"] = service_module.digest(result)
    return result


class TextTokenizer:
    encode = staticmethod(lambda text: [ord(character) for character in text])
    decode = staticmethod(lambda ids, **kwargs: "".join(chr(token) for token in ids))


class FakeModel:
    def __init__(self, texts, logprobs=True):
        self.texts, self.logprobs, self.calls = list(texts), logprobs, []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        text = self.texts.pop(0)
        ids = TextTokenizer.encode(text)
        return SimpleNamespace(
            token_ids=ids,
            log_probs=[-0.25] * len(ids) if self.logprobs else None,
            num_preempted=0,
            extra_fields={"min_global_steps": 3, "max_global_steps": 3},
        )


class FakeClient:
    def __init__(self, transitions, split="train", mismatch=False):
        self.transitions, self.split, self.mismatch = list(transitions), split, mismatch
        self.calls = []

    async def post(self, path, payload):
        self.calls.append((path, payload))
        if path == "/create":
            return {
                "session_id": "session",
                "task_id": "wrong" if self.mismatch else payload["task_id"],
                "split": self.split,
                "instruction": "public goal",
                "manifest_id": "frozen",
                "observation": "ORIGINAL [SEP] INITIAL",
            }
        if path == "/step":
            observation, reward, done = self.transitions.pop(0)
            return {
                "session_id": "session",
                "task_id": payload.get("task_id", "canonical"),
                "observation": observation,
                "raw_reward": reward,
                "done": done,
            }
        return {"closed": True}


class AgentTests(unittest.IsolatedAsyncioTestCase):
    def make_loop(self, texts, transitions, directory, response_length=24576, logprobs=True):
        config = SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(
                rollout=SimpleNamespace(prompt_length=4096, response_length=response_length)
            ),
            data=SimpleNamespace(apply_chat_template_kwargs={"enable_thinking": False}),
        )
        model = FakeModel(texts, logprobs)
        loop = agent.WebShopAgentLoop(
            trainer_config=SimpleNamespace(config=config),
            server_manager=model,
            tokenizer=TextTokenizer(),
            trace_dir=directory,
        )
        loop.env_client = FakeClient(transitions)
        return loop, model

    async def run_loop(self, loop, split="train"):
        return await loop.run(
            {"temperature": 1.0, "max_new_tokens": 8192},
            priority=7,
            raw_prompt=[{"role": "user", "content": "ignored placeholder"}],
            extra_info={"task_id": "canonical", "instruction": "public goal", "split": split, "manifest_id": "frozen"},
        )

    async def test_registry_masks_full_prefix_and_direct_episode_reward(self):
        self.assertIs(registry["webshop"], agent.WebShopAgentLoop)
        texts = ["<action>search[My unrestricted query words]</action>", "<action>click[buy now]</action>"]
        with tempfile.TemporaryDirectory() as directory:
            loop, model = self.make_loop(
                texts, [("ORIGINAL [SEP] RESULTS", 0, False), ("TERMINAL", 1, True)], directory
            )
            output = await self.run_loop(loop)
            self.assertEqual(output.reward_score, 10.0)
            self.assertEqual(sum(output.response_mask), sum(len(text) for text in texts))
            self.assertIn(0, output.response_mask)
            self.assertEqual(len(output.response_mask), len(output.response_ids))
            self.assertEqual(len(output.response_logprobs), len(output.response_ids))
            self.assertTrue(
                all(
                    lp == 0 for lp, mask in zip(output.response_logprobs, output.response_mask, strict=True) if not mask
                )
            )
            full = output.prompt_ids + output.response_ids
            self.assertEqual(full[: len(model.calls[1]["prompt_ids"])], model.calls[1]["prompt_ids"])
            prefix = TextTokenizer.decode(model.calls[1]["prompt_ids"])
            self.assertIn(texts[0], prefix)
            self.assertIn("ORIGINAL [SEP] RESULTS", prefix)
            self.assertNotIn("public_memory", prefix)
            self.assertTrue(all(call["sampling_params"]["max_tokens"] == 512 for call in model.calls))
            self.assertTrue(all("max_new_tokens" not in call["sampling_params"] for call in model.calls))
            self.assertEqual(model.calls[0]["priority"], 7)
            self.assertEqual(output.extra_fields["tool_rewards"], [])
            self.assertEqual(loop.env_client.calls[1][1]["action"], "search[My unrestricted query words]")
            self.assertEqual(loop.env_client.calls[-1][0], "/close")
            records = [json.loads(line) for line in Path(output.extra_fields["trace_path"]).read_text().splitlines()]
            transitions = [record for record in records if record["event"] == "transition"]
            self.assertEqual([record["step_reward"] for record in transitions], [0.0, 10.0])
            self.assertEqual(transitions[-1]["next_observation"], "TERMINAL")

    async def test_format_penalty_once_and_nonperfect_reward_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            loop, _ = self.make_loop(
                ["thinking <action>search[x]</action>", "<action>click[buy now]</action>"],
                [("TERMINAL", 0.999, True)],
                directory,
            )
            result = await self.run_loop(loop)
            self.assertAlmostEqual(result.reward_score, -0.1)
            self.assertEqual(sum(path == "/step" for path, _ in loop.env_client.calls), 1)
            self.assertEqual(result.extra_fields["reward_extra_info"]["invalid_actions"], 1)

    async def test_fifteen_steps_no_logprobs(self):
        with tempfile.TemporaryDirectory() as directory:
            loop, model = self.make_loop(
                ["<action>search[x]</action>"] * 15, [("PAGE", 0, False)] * 15, directory, logprobs=False
            )
            result = await self.run_loop(loop)
            self.assertEqual(len(model.calls), 15)
            self.assertEqual(result.reward_score, 0)
            self.assertIsNone(result.response_logprobs)
            self.assertEqual(result.extra_fields["reward_extra_info"]["steps"], 15)

    async def test_serialized_budget_never_slices_observation_or_action(self):
        with tempfile.TemporaryDirectory() as directory:
            loop, model = self.make_loop(
                ["<action>search[x]</action>"], [("X" * 2000, 0, False)], directory, response_length=1024
            )
            result = await self.run_loop(loop)
            self.assertEqual(len(model.calls), 1)
            self.assertEqual(result.extra_fields["stop_reason"], "serialized_token_budget")
            self.assertEqual(TextTokenizer.decode(result.response_ids), "<action>search[x]</action>")
            records = Path(result.extra_fields["trace_path"]).read_text()
            self.assertIn("X" * 2000, records)

    async def test_test_split_rejected_before_create(self):
        with tempfile.TemporaryDirectory() as directory:
            loop, _ = self.make_loop([], [], directory)
            with self.assertRaises(ValueError):
                await self.run_loop(loop, "test")
            self.assertEqual(loop.env_client.calls, [])

    async def test_identity_mismatch_and_generation_failure_close_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            loop, _ = self.make_loop([], [], directory)
            loop.env_client.mismatch = True
            with self.assertRaises(ValueError):
                await self.run_loop(loop)
            self.assertEqual(loop.env_client.calls[-1][0], "/close")
            loop.env_client.mismatch = False
            with self.assertRaises(IndexError):
                await self.run_loop(loop)
            self.assertEqual(loop.env_client.calls[-1][0], "/close")

    async def test_no_think_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            loop, _ = self.make_loop([], [], directory)
            config = SimpleNamespace(
                actor_rollout_ref=SimpleNamespace(rollout=loop.rollout_config),
                data=SimpleNamespace(apply_chat_template_kwargs={"enable_thinking": True}),
            )
            with self.assertRaises(ValueError):
                agent.WebShopAgentLoop(
                    trainer_config=SimpleNamespace(config=config),
                    server_manager=None,
                    tokenizer=None,
                    trace_dir=directory,
                )


class FakeOriginalEnvironment:
    def __init__(self, server, **kwargs):
        self.server, self.session = server, None
        self.reset()

    def reset(self, session=None, instruction_text=None):
        if session is None:
            raise AssertionError("random resets are forbidden")
        self.session = session
        self.server.user_sessions[session]["actions"] = []
        self.observation = "ORIGINAL " + self.get_instruction_text()
        return self.observation, None

    def get_instruction_text(self):
        return self.server.user_sessions[self.session]["goal"]["instruction_text"]

    def step(self, action):
        self.server.user_sessions[self.session]["actions"].append(action)
        if action == "click[buy now]":
            self.reset()  # emulate original terminal auto-reset
            return "ORIGINAL TERMINAL", 1.0, True, {"hidden": "not exported"}
        return "ORIGINAL SEARCH", 0, False, None


class ServiceAndDataTests(unittest.TestCase):
    def make_service(self):
        manifest = manifest_fixture()
        goals = {
            task["task_id"]: {"instruction_text": task["instruction"], "secret": "private", "weight": 1}
            for task in manifest["tasks"]
            if task["split"] in {"train", "dev"}
        }
        catalog = SimpleNamespace(
            server=SimpleNamespace(user_sessions={}), file_path="products", attr_path="attrs", _seed=0
        )
        return service_module.EnvironmentService(catalog, FakeOriginalEnvironment, manifest, goals), manifest

    def test_action_format_free_search_and_rewards(self):
        query = "Unlisted Brand X red 20 inch bench"
        self.assertEqual(service_module.parse_action(f"<action>search[{query}]</action>"), f"search[{query}]")
        for text in [
            "search[x]",
            "<think>x</think><action>search[x]</action>",
            "<action>click[]</action>",
            "<action>search[x]</action><action>click[y]</action>",
            "<action>search[\n]</action>",
        ]:
            self.assertIsNone(service_module.parse_action(text), text)
        self.assertEqual(agent.episode_reward(True, 1, 2), 9.8)
        self.assertEqual(agent.episode_reward(False, 1, 0), 0)
        self.assertEqual(agent.episode_reward(True, 0.9, 0), 0)

    def test_canonical_sessions_are_unique_pinned_and_cleanup_terminal(self):
        service, manifest = self.make_service()
        task_id = manifest["tasks"][0]["task_id"]
        first, second = service.create(task_id), service.create(task_id)
        self.assertNotEqual(first["session_id"], second["session_id"])
        self.assertNotIn("secret", json.dumps(first))
        self.assertIsNot(service.catalog.server.user_sessions[first["session_id"]]["goal"], service.goals[task_id])
        result = service.step(first["session_id"], "click[buy now]")
        self.assertEqual(result["observation"], "ORIGINAL TERMINAL")
        self.assertTrue(result["done"])
        self.assertNotIn(first["session_id"], service.catalog.server.user_sessions)
        self.assertIn(second["session_id"], service.sessions)
        service.close(second["session_id"])
        self.assertEqual(service.catalog.server.user_sessions, {})
        with self.assertRaises(ValueError):
            service.create(manifest["tasks"][-1]["task_id"])

    def test_session_limit_and_expiry(self):
        service, manifest = self.make_service()
        session = service.create(manifest["tasks"][0]["task_id"])["session_id"]
        for _ in range(15):
            result = service.step(session, "search[query]")
        self.assertTrue(result["truncated"])
        self.assertEqual(service.sessions, {})
        session = service.create(manifest["tasks"][0]["task_id"])["session_id"]
        service.sessions[session].touched -= 2000
        self.assertEqual(service.health()["active_sessions"], 0)
        self.assertNotIn(session, service.catalog.server.user_sessions)

    def test_http_parallel_sessions_and_test_rejection(self):
        service, manifest = self.make_service()
        server = service_module.make_http_server(service, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = agent.EnvironmentClient(f"http://127.0.0.1:{server.server_port}")
        try:
            self.assertGreaterEqual(server.request_queue_size, 128)
            with ThreadPoolExecutor(max_workers=4) as executor:
                records = list(
                    executor.map(
                        lambda _: client._post("/create", {"task_id": manifest["tasks"][0]["task_id"]}), range(8)
                    )
                )
            self.assertEqual(len({record["session_id"] for record in records}), 8)
            with self.assertRaises(urllib.error.HTTPError):
                client._post("/create", {"task_id": manifest["tasks"][-1]["task_id"]})
            for record in records:
                client._post("/close", {"session_id": record["session_id"]})
            with urllib.request.urlopen(client.url + "/health") as response:
                self.assertEqual(json.load(response)["active_sessions"], 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_manifest_hash_split_integrity_and_full_training_rows(self):
        manifest = manifest_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            read = service_module.read_manifest(path)
            rows = prepare_data.make_rows(read, "train")
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["agent_name"], "webshop")
            self.assertEqual(rows[0]["extra_info"]["task_id"], manifest["tasks"][0]["task_id"])
            self.assertNotIn("secret", json.dumps(rows))
            self.assertEqual(len(prepare_data.make_rows(read, "dev", 1)), 1)
            with self.assertRaises(ValueError):
                prepare_data.make_rows(read, "test")
            with self.assertRaises(ValueError):
                prepare_data.make_rows(read, "dev", 100)
            manifest["tasks"][0]["instruction"] = "tampered"
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                service_module.read_manifest(path)

    def test_verified_catalog_suppresses_first_reset_and_excludes_test_goals(self):
        manifest = manifest_fixture()
        goals = [
            (task["task_id"], {"instruction_text": task["instruction"], "weight": 1}) for task in manifest["tasks"]
        ]
        original_module = ModuleType("web_agent_site.envs.web_agent_text_env")
        utils = ModuleType("web_agent_site.utils")
        resets = []

        # Execute in the fake source module's globals: the original WebShop
        # constructor explicitly resolves its module-level WebAgentTextEnv name
        # inside super(WebAgentTextEnv, self). Rebinding that name to a subclass
        # must fail this regression rather than silently passing a simpler mock.
        original_module.__dict__.update(SimpleNamespace=SimpleNamespace, goals=goals, resets=resets)
        exec(
            "class CatalogBase:\n"
            "    def __init__(self):\n"
            "        self.base_initializations = 1\n"
            "class WebAgentTextEnv(CatalogBase):\n"
            "    def __init__(self):\n"
            "        super(WebAgentTextEnv, self).__init__()\n"
            "        self.server = SimpleNamespace(goals=[goal for _, goal in goals], user_sessions={})\n"
            "        self.reset()\n"
            "    def reset(self):\n"
            "        resets.append('forbidden random reset')\n",
            original_module.__dict__,
        )
        OriginalCatalog = original_module.WebAgentTextEnv
        original_reset = OriginalCatalog.reset
        installed = []

        def install(catalog, ordered):
            installed.extend(task_id for task_id, _ in ordered)
            catalog.server.goals = [goal for _, goal in ordered]

        helpers = SimpleNamespace(
            load_environment=lambda args: (original_module.WebAgentTextEnv(), "provenance"),
            verify_manifest=lambda manifest, goals, provenance: goals_fixture,
            install_manifest_goals=install,
        )
        goals_fixture = goals
        args = SimpleNamespace(
            eval_helpers="helpers",
            manifest="manifest",
            webshop_root="/trusted/source",
            human_instructions="/assets/human.json",
        )
        with (
            patch.object(service_module, "load_eval_helpers", return_value=helpers),
            patch.object(service_module, "read_manifest", return_value=manifest),
            patch.object(
                service_module.importlib,
                "import_module",
                side_effect=lambda name: utils if name == "web_agent_site.utils" else original_module,
            ),
            patch.object(sys, "path", list(sys.path)),
        ):
            catalog, env_class, result, allowed = service_module.load_verified_environment(args)
        self.assertEqual(resets, [])
        self.assertIs(env_class, OriginalCatalog)
        self.assertIs(original_module.WebAgentTextEnv, OriginalCatalog)
        self.assertIs(OriginalCatalog.reset, original_reset)
        self.assertEqual(catalog.base_initializations, 1)
        catalog.reset()
        self.assertEqual(resets, ["forbidden random reset"])
        self.assertEqual(len(allowed), 3)
        self.assertNotIn(manifest["tasks"][-1]["task_id"], installed)
        self.assertEqual(utils.HUMAN_ATTR_PATH, "/assets/human.json")
        self.assertEqual(result["manifest_id"], manifest["manifest_id"])

        # Restoration is mandatory even if catalog/asset loading raises.
        with (
            patch.object(service_module, "load_eval_helpers", return_value=helpers),
            patch.object(service_module, "read_manifest", return_value=manifest),
            patch.object(helpers, "load_environment", side_effect=RuntimeError("asset loading failed")),
            patch.object(
                service_module.importlib,
                "import_module",
                side_effect=lambda name: utils if name == "web_agent_site.utils" else original_module,
            ),
            patch.object(sys, "path", list(sys.path)),
        ):
            with self.assertRaisesRegex(RuntimeError, "asset loading failed"):
                service_module.load_verified_environment(args)
        self.assertIs(OriginalCatalog.reset, original_reset)
        self.assertIs(original_module.WebAgentTextEnv, OriginalCatalog)

    def test_endpoint_rejects_remote_hosts_and_credentials(self):
        for endpoint in [
            "https://localhost",
            "http://example.com",
            "http://user:password@localhost",
            "http://localhost/path",
        ]:
            with self.assertRaises(ValueError):
                agent.local_endpoint(endpoint)


if __name__ == "__main__":
    unittest.main()
