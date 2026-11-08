"""CPU-only original WebShop HTTP environment; never imports a model API.

Launch this file with the isolated WebShop Python environment and explicit asset
paths. Only public instructions/observations and original scalar rewards cross
HTTP. The verified catalog is loaded once; every episode has its own UUID.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

MAX_STEPS = 15
_ACTION = re.compile(r"<action>((?:search|click)\[[^\[\]\r\n]+\])</action>")


def parse_action(text):
    """Require just one action tag; leave free-form search arguments unchanged."""
    if not isinstance(text, str):
        return None
    match = _ACTION.fullmatch(text.strip())
    if not match:
        return None
    action = match.group(1)
    argument = action[action.index("[") + 1 : -1]
    if not argument.strip() or "<action>" in argument or "</action>" in argument:
        return None
    return action


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_manifest(path):
    """Validate manifest identity and public task metadata without loading assets."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("protocol") != "webshop-choice-v1" or manifest.get("manifest_id") != digest(
        {key: value for key, value in manifest.items() if key != "manifest_id"}
    ):
        raise ValueError("invalid frozen WebShop manifest")
    seen, groups = set(), {}
    for task in manifest["tasks"]:
        task_id, instruction, split = task["task_id"], task["instruction"], task["split"]
        if not isinstance(task_id, str) or not re.fullmatch(r"[0-9a-f]{64}", task_id) or task_id in seen:
            raise ValueError("invalid or duplicate canonical task ID")
        if not isinstance(instruction, str) or not instruction.strip() or split not in {"train", "dev", "test"}:
            raise ValueError("invalid public task metadata")
        group = " ".join(instruction.split()).casefold()
        if group in groups and groups[group] != split:
            raise ValueError("the same public instruction crosses dataset splits")
        groups[group] = split
        seen.add(task_id)
    if [task["task_id"] for task in manifest["tasks"]] != sorted(seen):
        raise ValueError("manifest IDs are not in canonical order")
    if manifest.get("goals_hash") != digest(sorted(seen)):
        raise ValueError("manifest canonical goals hash differs")
    return manifest


def load_eval_helpers(path):
    spec = importlib.util.spec_from_file_location("_webshop_frozen_eval", Path(path).resolve())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_verified_environment(args):
    """Use existing asset/source/goal verification, without an initial random task.

    The original constructor calls reset before the manifest can be verified.
    Suppress only that first reset, so no test goal is ever executed. All episode
    transitions subsequently use the unmodified original WebShop implementation.
    """
    helpers = load_eval_helpers(args.eval_helpers)
    manifest = read_manifest(args.manifest)
    # Import the trusted source checkout, not any downloaded asset directory.
    import sys

    sys.path.insert(0, str(Path(args.webshop_root).resolve()))
    utils = importlib.import_module("web_agent_site.utils")
    utils.HUMAN_ATTR_PATH = str(Path(args.human_instructions).resolve())
    module = importlib.import_module("web_agent_site.envs.web_agent_text_env")
    original_class = module.WebAgentTextEnv

    # The original __init__ uses super(WebAgentTextEnv, self), so rebinding
    # that module-level class to a subclass recurses. Suppress reset in place.
    original_reset = original_class.reset
    original_class.reset = lambda self, *args, **kwargs: ("", None)
    try:
        catalog, provenance = helpers.load_environment(args)
    finally:
        original_class.reset = original_reset
    ordered = helpers.verify_manifest(manifest, catalog.server.goals, provenance)
    task_map = {task["task_id"]: task for task in manifest["tasks"]}
    for task_id, goal in ordered:
        if task_map[task_id]["instruction"] != goal["instruction_text"]:
            raise ValueError("manifest public instruction differs from canonical goal")
    allowed = [(task_id, goal) for task_id, goal in ordered if task_map[task_id]["split"] in {"train", "dev"}]
    if not allowed:
        raise ValueError("manifest has no train/dev tasks")
    helpers.install_manifest_goals(catalog, allowed)
    return catalog, original_class, manifest, dict(allowed)


class EnvironmentService:
    """All catalog/session access is serialized, including automatic resets."""

    def __init__(self, catalog, env_class, manifest, goals, session_ttl=1800):
        self.catalog, self.env_class, self.goals = catalog, env_class, goals
        self.manifest_id = manifest["manifest_id"]
        self.tasks = {task["task_id"]: task for task in manifest["tasks"] if task["split"] in {"train", "dev"}}
        self.sessions = {}
        self.lock = threading.RLock()
        self.session_ttl = session_ttl

    def _close(self, session_id):
        self.sessions.pop(session_id, None)
        self.catalog.server.user_sessions.pop(session_id, None)

    def _expire(self):
        now = time.monotonic()
        for session_id, record in list(self.sessions.items()):
            if now - record.touched > self.session_ttl:
                self._close(session_id)

    def health(self):
        with self.lock:
            self._expire()
            return {
                "status": "ok",
                "manifest_id": self.manifest_id,
                "splits": ["train", "dev"],
                "max_steps": MAX_STEPS,
                "active_sessions": len(self.sessions),
                "task_counts": {
                    split: sum(task["split"] == split for task in self.tasks.values()) for split in ("train", "dev")
                },
            }

    def create(self, task_id):
        with self.lock:
            self._expire()
            if task_id not in self.tasks or task_id not in self.goals:
                raise ValueError("task ID is not an allowed train/dev canonical task")
            session_id = uuid4().hex
            server = self.catalog.server
            # A deep copy protects canonical goals from original session mutation.
            server.user_sessions[session_id] = {"goal": copy.deepcopy(self.goals[task_id]), "done": False}
            original_class = self.env_class

            class SessionEnvironment(original_class):
                def reset(self, session=None, instruction_text=None):
                    # Constructor and terminal auto-reset both stay on this exact goal.
                    return super().reset(session=session_id, instruction_text=instruction_text)

            try:
                env = SessionEnvironment(
                    observation_mode="text",
                    server=server,
                    file_path=self.catalog.file_path,
                    attr_path=self.catalog.attr_path,
                    seed=self.catalog._seed,
                    num_prev_obs=0,
                    num_prev_actions=0,
                )
                observation = env.observation
                if env.get_instruction_text().removeprefix("Instruction: ") != self.tasks[task_id]["instruction"]:
                    raise ValueError("session instruction differs from requested canonical task")
                self.sessions[session_id] = SimpleNamespace(env=env, task_id=task_id, steps=0, touched=time.monotonic())
            except BaseException:
                self._close(session_id)
                raise
            task = self.tasks[task_id]
            return {
                "session_id": session_id,
                "task_id": task_id,
                "split": task["split"],
                "instruction": task["instruction"],
                "observation": str(observation),
                "manifest_id": self.manifest_id,
                "done": False,
                "raw_reward": 0.0,
            }

    def step(self, session_id, action):
        with self.lock:
            self._expire()
            record = self.sessions.get(session_id)
            if record is None:
                raise ValueError("unknown, closed, or expired session")
            # HTTP actions are executable search[...] / click[...], not model tags.
            if not isinstance(action, str) or parse_action(f"<action>{action}</action>") != action:
                raise ValueError("invalid executable action format")
            try:
                observation, reward, done, _ = record.env.step(action)
                record.steps += 1
                record.touched = time.monotonic()
                result = {
                    "session_id": session_id,
                    "task_id": record.task_id,
                    "observation": str(observation),
                    "raw_reward": float(reward),
                    "done": bool(done),
                    "steps": record.steps,
                    "truncated": record.steps >= MAX_STEPS and not done,
                }
                if done or record.steps >= MAX_STEPS:
                    self._close(session_id)
                return result
            except BaseException:
                self._close(session_id)
                raise

    def close(self, session_id):
        with self.lock:
            self._close(session_id)
            return {"closed": True}


def make_http_server(service, host="127.0.0.1", port=8765):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # No payloads, goals or tracebacks in HTTP access logs.

        def reply(self, status, payload):
            data = canonical(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, service.health())
            else:
                self.reply(404, {"error": "unknown endpoint"})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("request body must be 1..65536 bytes")
                payload = json.loads(self.rfile.read(length))
                if self.path == "/create":
                    result = service.create(payload["task_id"])
                elif self.path == "/step":
                    result = service.step(payload["session_id"], payload["action"])
                elif self.path == "/close":
                    result = service.close(payload["session_id"])
                else:
                    self.reply(404, {"error": "unknown endpoint"})
                    return
                self.reply(200, result)
            except (ValueError, KeyError, TypeError) as error:
                print(f"WebShop request rejected: {type(error).__name__}: {error}", flush=True)
                self.reply(400, {"error": "invalid request or unavailable train/dev session"})
            except Exception as error:
                print(f"WebShop transition failed: {type(error).__name__}: {error}", flush=True)
                self.reply(500, {"error": "WebShop environment transition failed"})

    class WebShopHTTPServer(ThreadingHTTPServer):
        # All 128 rollout sessions may connect together; Python 3.10 defaults to 5.
        request_queue_size = 256

    return WebShopHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "eval-helpers", "webshop-root", "products", "attributes", "human-instructions", "index"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--goal-type", choices=["human", "synthetic"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--host", choices=["127.0.0.1", "localhost"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--session-ttl", type=float, default=1800)
    args = parser.parse_args()
    if args.session_ttl <= 0:
        parser.error("session TTL must be positive")
    catalog, env_class, manifest, goals = load_verified_environment(args)
    service = EnvironmentService(catalog, env_class, manifest, goals, args.session_ttl)
    server = make_http_server(service, args.host, args.port)
    print(canonical(service.health()), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
