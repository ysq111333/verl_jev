"""Official asynchronous WebShop agent loop for verl commit b17218ba.

Registry target: examples.webshop_grpo.webshop_agent.WebShopAgentLoop.
Use rollout.agent.agent_loop_config_path pointing to a YAML list with name:
webshop and that _target_. Constructor options: env_url, trace_dir,
request_timeout (or WEBSHOP_ENV_URL / WEBSHOP_TRACE_DIR environment variables).
Set data.apply_chat_template_kwargs.enable_thinking=false, rollout.multi_turn.
enable=true, prompt_length=4096, and response_length=24576 (includes observations).

The official worker trains one causal serialized trajectory. Consequently this
implementation retains the full transcript, NOT the legacy last-two-only memory.
All generated tokens are mask1; environment and template context is mask0.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from examples.webshop_grpo.webshop_env_service import MAX_STEPS, parse_action
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register

MAX_NEW_TOKENS = 512
SYSTEM_PROMPT = (
    "You are shopping in WebShop. Follow the user's shopping instruction using the original page observations. "
    "Return exactly one action, and no explanation or reasoning: <action>search[your search query]</action> "
    "or <action>click[visible button, option, or product ID]</action>. "
    "Choose any free-form search query; use clickable text from the current page for clicks. "
    "Do not invent a purchase or describe an action: execute it using its action tag."
)


def local_endpoint(url):
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("WebShop service must be a credential-free local HTTP origin")
    return url.rstrip("/")


class EnvironmentClient:
    """stdlib HTTP runs in CPU threads, never blocking sibling rollout coroutines."""

    def __init__(self, url, timeout=120):
        self.url, self.timeout = local_endpoint(url), timeout

    def _post(self, path, payload):
        request = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload, allow_nan=False).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # No retries: a lost step response must never cause a second transition.
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    async def post(self, path, payload):
        return await asyncio.to_thread(self._post, path, payload)


def episode_reward(done, raw_reward, invalid_actions):
    """Original exact-success reward, with the format penalty applied once."""
    return (10.0 if done and float(raw_reward) == 1.0 else 0.0) - 0.1 * invalid_actions


class RolloutTrace:
    """Unique local JSONL per rollout. Only explicitly public fields are logged."""

    def __init__(self, directory, rollout_id, task_id, split, instruction, manifest_id):
        directory = Path(directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = directory / f"{rollout_id}.jsonl"
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.file = os.fdopen(fd, "w", encoding="utf-8")
        self.write(
            {
                "event": "start",
                "rollout_id": rollout_id,
                "task_id": task_id,
                "split": split,
                "instruction": instruction,
                "manifest_id": manifest_id,
                "history_mode": "official_full_causal_transcript",
                "max_steps": MAX_STEPS,
                "max_new_tokens_per_step": MAX_NEW_TOKENS,
            }
        )

    def write(self, record):
        self.file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


@register("webshop")
class WebShopAgentLoop(AgentLoopBase):
    merges_context_messages = True

    def __init__(self, *args, env_url=None, trace_dir=None, request_timeout=120, **kwargs):
        super().__init__(*args, **kwargs)
        if self.apply_chat_template_kwargs.get("enable_thinking") is not False:
            raise ValueError("WebShop requires data.apply_chat_template_kwargs.enable_thinking=false")
        self.response_length = int(self.rollout_config.response_length)
        if self.response_length < MAX_NEW_TOKENS:
            raise ValueError(
                "rollout.response_length must allow at least 512 generated tokens plus observation context"
            )
        env_url = env_url or os.environ.get("WEBSHOP_ENV_URL", "http://127.0.0.1:8765")
        trace_dir = trace_dir or os.environ.get("WEBSHOP_TRACE_DIR")
        if not trace_dir:
            raise ValueError("set WEBSHOP_TRACE_DIR or the loop's trace_dir for mandatory local rollout traces")
        self.trace_dir = trace_dir
        self.env_client = EnvironmentClient(env_url, request_timeout)

    def _cap_text_prompt_length(self, prompt_ids):
        # Original observations must not be silently shortened by the base helper.
        if len(prompt_ids) > int(self.rollout_config.prompt_length):
            raise ValueError("original WebShop initial prompt exceeds rollout.prompt_length; increase it")
        return prompt_ids

    async def run(self, sampling_params, priority=0, **kwargs):
        extra_info = kwargs.get("extra_info") or {}
        task_id = kwargs.get("task_id", extra_info.get("task_id"))
        split = kwargs.get("split", extra_info.get("split"))
        instruction = extra_info.get("instruction")
        if split not in {"train", "dev"} or not isinstance(task_id, str) or not instruction:
            raise ValueError("WebShop requires canonical task_id, instruction, and train/dev split in extra_info")
        rollout_id = uuid4().hex
        session_id, trace = None, None
        metrics = {"generate_sequences": 0.0, "tool_calls": 0.0, "num_preempted": -1}
        invalid_actions, steps, raw_reward, done = 0, 0, 0.0, False
        stop_reason = "step_limit"
        mask, logprobs, runtime_ids = [], None, []
        num_turns = 1
        extra_fields = {}
        try:
            started = time.perf_counter()
            created = await self.env_client.post("/create", {"task_id": task_id})
            metrics["tool_calls"] += time.perf_counter() - started
            session_id = created["session_id"]
            if (
                created["task_id"] != task_id
                or created["split"] != split
                or created["instruction"] != instruction
                or (extra_info.get("manifest_id") and created["manifest_id"] != extra_info["manifest_id"])
            ):
                raise ValueError("environment canonical task/split/instruction/manifest identity mismatch")
            observation = created["observation"]
            if not isinstance(observation, str):
                raise ValueError("WebShop must return original text observations")
            trace = RolloutTrace(self.trace_dir, rollout_id, task_id, split, instruction, created["manifest_id"])
            trace.write({"event": "initial_observation", "observation": observation})
            # Do not trust arbitrary dataset content as instructions or hidden state.
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction + "\n\n" + observation},
            ]
            runtime_ids = await self.ct_build_initial_tokens(messages)
            params = dict(sampling_params)
            params.pop("max_new_tokens", None)
            params["max_tokens"] = MAX_NEW_TOKENS
            for step in range(MAX_STEPS):
                if self.response_length - len(mask) < MAX_NEW_TOKENS:
                    stop_reason = "serialized_token_budget"
                    break
                started = time.perf_counter()
                generated = await self.server_manager.generate(
                    request_id=rollout_id, prompt_ids=runtime_ids, sampling_params=dict(params), priority=int(priority)
                )
                metrics["generate_sequences"] += time.perf_counter() - started
                if generated.num_preempted is not None:
                    metrics["num_preempted"] = max(0, metrics["num_preempted"]) + int(generated.num_preempted)
                if not generated.token_ids or len(generated.token_ids) > MAX_NEW_TOKENS:
                    raise ValueError("WebShop generation must return 1..512 action tokens")
                text = await asyncio.to_thread(self.tokenizer.decode, generated.token_ids, skip_special_tokens=True)
                raw_text = await asyncio.to_thread(
                    self.tokenizer.decode, generated.token_ids, skip_special_tokens=False
                )
                action = parse_action(text)
                trace.write(
                    {
                        "event": "generation",
                        "step": step + 1,
                        "observation": observation,
                        "generated_text": raw_text,
                        "parsed_text": text,
                        "action": action,
                        "generated_token_ids": generated.token_ids,
                    }
                )
                has_logprobs = generated.log_probs is not None and len(generated.log_probs) > 0
                if step and (logprobs is not None) != has_logprobs:
                    raise ValueError("inconsistent rollout log-prob availability across WebShop turns")
                if has_logprobs and logprobs is None:
                    logprobs = [0.0] * len(mask)
                merged, mask, logprobs = await self.ct_merge_assistant_token(
                    runtime_ids,
                    generated.token_ids,
                    mask,
                    logprobs,
                    assistant_logprobs=generated.log_probs if has_logprobs else None,
                )
                runtime_ids = merged.token_ids
                if len(mask) > self.response_length:
                    raise ValueError("assistant merge exceeded the serialized response budget")
                for key in ("min_global_steps", "max_global_steps"):
                    value = generated.extra_fields.get(key)
                    if value is not None:
                        reducer = min if key == "min_global_steps" else max
                        extra_fields[key] = reducer(extra_fields.get(key, value), value)
                steps += 1
                num_turns += 1
                previous_observation = observation
                if action is None:
                    invalid_actions += 1
                    raw_reward, done = 0.0, False
                    step_reward = -0.1
                    # Malformed output is a no-op; do not send arbitrary text to WebShop.
                else:
                    started = time.perf_counter()
                    transition = await self.env_client.post("/step", {"session_id": session_id, "action": action})
                    metrics["tool_calls"] += time.perf_counter() - started
                    if transition["task_id"] != task_id or transition["session_id"] != session_id:
                        raise ValueError("WebShop transition changed task/session identity")
                    observation = transition["observation"]
                    if not isinstance(observation, str):
                        raise ValueError("WebShop must return original text observations")
                    raw_reward, done = float(transition["raw_reward"]), bool(transition["done"])
                    step_reward = episode_reward(done, raw_reward, 0)
                trace.write(
                    {
                        "event": "transition",
                        "step": steps,
                        "observation": previous_observation,
                        "action": action,
                        "generated_text": raw_text,
                        "next_observation": observation,
                        "raw_env_reward": raw_reward,
                        "step_reward": step_reward,
                        "done": done,
                        "invalid_format": action is None,
                    }
                )
                if done:
                    stop_reason = "environment_done"
                    break
                if steps == MAX_STEPS:
                    break
                # The assistant already exists in runtime_ids. Include it in BOTH
                # message lists so CT appends only the new user observation.
                messages.append({"role": "assistant", "content": text})
                previous_messages = list(messages)
                messages.append({"role": "user", "content": observation})
                merged, new_mask, new_logprobs = await self.ct_merge_context_msg(
                    previous_messages,
                    messages,
                    runtime_ids,
                    mask,
                    logprobs,
                )
                # Never train on a truncated action or generate from a prefix that
                # differs from the serialized trajectory sent to the trainer.
                if len(new_mask) + MAX_NEW_TOKENS > self.response_length:
                    stop_reason = "serialized_token_budget"
                    break
                runtime_ids, mask, logprobs = merged.token_ids, new_mask, new_logprobs
                num_turns += 1
            if not mask or not any(mask):
                raise ValueError("WebShop rollout contains no generated action tokens")
            reward = episode_reward(done, raw_reward, invalid_actions)
            reward_info = {
                "episode_reward": reward,
                "success": int(done and raw_reward == 1.0),
                "invalid_actions": invalid_actions,
                "steps": steps,
                "truncated": int(not done),
            }
            trace.write(
                {
                    "event": "end",
                    **reward_info,
                    "raw_env_reward": raw_reward,
                    "done": done,
                    "stop_reason": stop_reason,
                    "response_tokens": len(mask),
                }
            )
            extra_fields.update(
                {
                    "rollout_id": rollout_id,
                    "task_id": task_id,
                    "split": split,
                    "trace_path": str(trace.path),
                    "history_mode": "official_full_causal_transcript",
                    "stop_reason": stop_reason,
                    "reward_extra_info": reward_info,
                    "turn_scores": [],
                    "tool_rewards": [],
                }
            )
            # reward_score becomes rm_scores in the official worker; do not also
            # register a format/reward function or populate tool_rewards penalties.
            return AgentLoopOutput(
                prompt_ids=runtime_ids[: -len(mask)],
                response_ids=runtime_ids[-len(mask) :],
                response_mask=mask,
                response_logprobs=logprobs,
                reward_score=reward,
                num_turns=num_turns,
                metrics=metrics,
                extra_fields=extra_fields,
            )
        except BaseException as error:
            if trace:
                trace.write({"event": "error", "error_type": type(error).__name__, "steps": steps})
            raise
        finally:
            try:
                if session_id:
                    await asyncio.shield(self.env_client.post("/close", {"session_id": session_id}))
            finally:
                if trace:
                    trace.close()
