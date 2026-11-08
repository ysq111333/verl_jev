# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native GDN routing and tiny numerical checks; no GPU or FLA required.

AST loading keeps the routing checks independent of verl/Transformers runtime
imports. Numerical checks use the installed Transformers torch reference.
"""

import ast
import os
from functools import wraps
from importlib import import_module
from inspect import signature, unwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

MODELS_DIR = Path(__file__).resolve().parents[2] / "verl" / "models" / "transformers"
FUNCTIONS = {
    "_call_accepts_kwarg",
    "_use_native_gdn",
    "_native_gdn_kernels",
    "_report_native_gdn",
    "_split_packed_args",
    "_packed_causal_conv1d_fallback",
    "_packed_chunk_gated_delta_rule",
    "qwen3_5_gated_delta_net_forward",
}


@pytest.fixture
def adapter(monkeypatch):
    tree = ast.parse((MODELS_DIR / "qwen3_5.py").read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    assert {node.name for node in functions} == FUNCTIONS
    tree.body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions]
    ast.fix_missing_locations(tree)
    namespace = {
        "os": os,
        "signature": signature,
        "unwrap": unwrap,
        "import_module": import_module,
        "get_ulysses_sequence_parallel_world_size": lambda: 1,
    }
    exec(compile(tree, str(MODELS_DIR / "qwen3_5.py"), "exec"), namespace)
    monkeypatch.delenv("VERL_QWEN35_NATIVE_GDN", raising=False)
    return namespace


def _forbidden(*args, **kwargs):
    raise AssertionError("external/fused kernel was selected")


def _dispatcher(native):
    @wraps(native)
    def dispatch(*args, **kwargs):
        return _forbidden(*args, **kwargs)

    return dispatch


@pytest.mark.parametrize("flag", [None, "", "0", "true", "1"])
def test_flag_is_explicit_opt_in(adapter, monkeypatch, flag):
    if flag is not None:
        monkeypatch.setenv("VERL_QWEN35_NATIVE_GDN", flag)
    assert adapter["_use_native_gdn"]() == (flag == "1")


@pytest.mark.parametrize("legacy_conv_name", [False, True])
def test_native_kernels_unwrap_dispatchers(adapter, legacy_conv_name):
    module = ModuleType("fake_transformers")

    def chunk(*args, **kwargs):
        return "chunk"

    def recurrent(*args, **kwargs):
        return "recurrent"

    def update(*args, **kwargs):
        return "conv"

    update.__module__ = module.__name__
    module.torch_chunk_gated_delta_rule = _dispatcher(_dispatcher(chunk))
    module.torch_recurrent_gated_delta_rule = _dispatcher(recurrent)
    module.causal_conv1d_update = _forbidden if legacy_conv_name else _dispatcher(update)
    if legacy_conv_name:
        module.torch_causal_conv1d_update = update
    kernels = adapter["_native_gdn_kernels"](module)
    assert kernels == (chunk, recurrent, update)
    assert [kernel() for kernel in kernels] == ["chunk", "recurrent", "conv"]


def test_missing_native_kernel_fails_closed(adapter):
    with pytest.raises(RuntimeError, match="torch_chunk_gated_delta_rule"):
        adapter["_native_gdn_kernels"](ModuleType("unsupported_transformers"))


def test_external_conv_update_fails_closed(adapter):
    module = ModuleType("unsupported_transformers")
    module.torch_chunk_gated_delta_rule = _forbidden
    module.torch_recurrent_gated_delta_rule = _forbidden
    module.causal_conv1d_update = _forbidden
    with pytest.raises(RuntimeError, match="Cannot identify a native causal conv update"):
        adapter["_native_gdn_kernels"](module)


def test_startup_report_and_sp_guard(adapter, monkeypatch, capsys):
    dense = type("Qwen3_5GatedDeltaNet", (), {})()
    moe = type("Qwen3_5MoeGatedDeltaNet", (), {})()
    model = SimpleNamespace(modules=lambda: iter((dense, moe)))
    adapter["_report_native_gdn"](model, 2)  # Unset must not reject the existing FLA path.
    assert capsys.readouterr().out == ""
    monkeypatch.setenv("VERL_QWEN35_NATIVE_GDN", "1")
    adapter["_native_gdn_kernels"] = lambda module: (_forbidden,) * 3
    adapter["_report_native_gdn"](model, 1)
    report = capsys.readouterr().out
    assert "2 DeltaNet layers" in report
    assert "SP=1 only" in report
    assert "norm and LM-head backend unchanged" in report
    with pytest.raises(NotImplementedError, match="requires ulysses_sequence_parallel_size=1"):
        adapter["_report_native_gdn"](model, 2)


def test_native_hook_is_independent_of_fused_head_backend():
    tree = ast.parse((MODELS_DIR / "monkey_patch.py").read_text())
    apply_patch = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "apply_monkey_patch"
    )
    branch = next(
        node
        for node in ast.walk(apply_patch)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "model.config.model_type in ['qwen3_5', 'qwen3_5_moe']"
    )
    assert any(
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "_report_native_gdn"
        for node in branch.body
    )
    assignments = {
        ast.unparse(target) for node in branch.body if isinstance(node, ast.Assign) for target in node.targets
    }
    assert {"Qwen3_5GatedDeltaNet.forward", "Qwen3_5MoeGatedDeltaNet.forward"} <= assignments


@pytest.fixture
def torch_adapter(adapter):
    torch = pytest.importorskip("torch")
    adapter.update(torch=torch, F=torch.nn.functional)
    return adapter, torch


def test_force_split_even_when_native_accepts_kwargs(torch_adapter):
    adapter, torch = torch_adapter
    lengths = []

    def native(q, k, v, **kwargs):
        assert "cu_seqlens" not in kwargs
        assert kwargs["initial_state"] is None
        lengths.append(q.shape[1])
        return v.cumsum(dim=1), None

    x = torch.ones(1, 5, 1, 2, requires_grad=True)
    g = torch.ones(1, 5, 1)
    cu_seqlens = torch.tensor([0, 2, 5])
    out, state = adapter["_packed_chunk_gated_delta_rule"](
        native, x, x, x, g, g, cu_seqlens, cu_seqlens, force_split=True
    )
    assert lengths == [2, 3]
    assert state is None
    torch.testing.assert_close(out[0, :, 0, 0], torch.tensor([1.0, 2.0, 1.0, 2.0, 3.0]))
    out.sum().backward()
    torch.testing.assert_close(x.grad[0, :, 0, 0], torch.tensor([2.0, 1.0, 3.0, 2.0, 1.0]))
    with pytest.raises(NotImplementedError, match="cp_context"):
        adapter["_packed_chunk_gated_delta_rule"](
            native, x, x, x, g, g, cu_seqlens, cu_seqlens, cp_context=object(), force_split=True
        )


def test_default_still_selects_instance_kernels(torch_adapter):
    adapter, torch = torch_adapter
    calls = []

    def chunk(q, k, v, **kwargs):
        calls.append("instance chunk")
        return v, None

    def conv(x, *args, **kwargs):
        calls.append("instance conv")
        return x

    layer = _tiny_fake_layer(torch)
    layer.chunk_gated_delta_rule = chunk
    layer.causal_conv1d_fn = conv
    adapter["import_module"] = lambda name: SimpleNamespace(apply_mask_to_padding_states=lambda x, mask: x)
    adapter["qwen3_5_gated_delta_net_forward"](layer, torch.ones(1, 3, 4))
    assert calls == ["instance conv", "instance chunk"]


def _tiny_fake_layer(torch):
    layer = torch.nn.Module()
    layer.head_k_dim = layer.head_v_dim = layer.key_dim = layer.value_dim = 2
    layer.num_k_heads = layer.num_v_heads = 1
    layer.conv_kernel_size = 3
    layer.layer_idx = 0
    layer.activation = "silu"
    layer.in_proj_qkv = torch.nn.Linear(4, 6, bias=False)
    layer.in_proj_z = torch.nn.Linear(4, 2, bias=False)
    layer.in_proj_b = torch.nn.Linear(4, 1, bias=False)
    layer.in_proj_a = torch.nn.Linear(4, 1, bias=False)
    layer.conv1d = torch.nn.Conv1d(6, 6, 3, groups=6, padding=2, bias=False)
    layer.out_proj = torch.nn.Linear(2, 4, bias=False)
    layer.norm = lambda x, z: x * z.sigmoid()
    layer.A_log = torch.nn.Parameter(torch.zeros(1))
    layer.dt_bias = torch.nn.Parameter(torch.zeros(1))
    layer.chunk_gated_delta_rule = _forbidden
    layer.recurrent_gated_delta_rule = _forbidden
    layer.causal_conv1d_fn = _forbidden
    layer.causal_conv1d_update = _forbidden
    return layer


@pytest.mark.parametrize("cached", [False, True])
def test_exact_forced_forward_ignores_instance_and_module_dispatchers(torch_adapter, monkeypatch, cached):
    adapter, torch = torch_adapter
    monkeypatch.setenv("VERL_QWEN35_NATIVE_GDN", "1")
    layer = _tiny_fake_layer(torch)
    module = ModuleType("fake_transformers")
    module.apply_mask_to_padding_states = lambda x, mask: x
    calls = []

    def chunk(q, k, v, **kwargs):
        calls.append("native chunk")
        return v, None

    def recurrent(q, k, v, **kwargs):
        calls.append("native recurrent")
        assert kwargs["initial_state"] is cache.recurrent_states[0]
        return v, kwargs["initial_state"]

    def update(x, state, weight, bias, activation):
        calls.append("native conv update")
        return x

    module.torch_chunk_gated_delta_rule = _dispatcher(chunk)
    module.torch_recurrent_gated_delta_rule = _dispatcher(recurrent)
    module.torch_causal_conv1d_update = update
    module.causal_conv1d_fn = module.causal_conv1d_update = _forbidden
    adapter["import_module"] = lambda name: module
    cache = SimpleNamespace(
        has_previous_state=True,
        conv_states=[torch.zeros(1, 6, 3)],
        recurrent_states=[torch.zeros(1, 1, 2, 2)],
    )
    out = adapter["qwen3_5_gated_delta_net_forward"](
        layer, torch.ones(1, 1 if cached else 3, 4), cache_params=cache if cached else None
    )
    assert out.shape == (1, 1 if cached else 3, 4)
    assert calls == (["native conv update", "native recurrent"] if cached else ["native chunk"])
    adapter["get_ulysses_sequence_parallel_world_size"] = lambda: 2
    with pytest.raises(NotImplementedError, match="requires ulysses_sequence_parallel_size=1"):
        adapter["qwen3_5_gated_delta_net_forward"](layer, torch.ones(1, 3, 4))


def test_actual_native_recurrent_and_conv_update(torch_adapter):
    adapter, torch = torch_adapter
    module = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    chunk, recurrent, conv_update = adapter["_native_gdn_kernels"](module)
    torch.manual_seed(456)
    q, k, v = (torch.randn(1, 3, 1, 2) for _ in range(3))
    g = -torch.rand(1, 3, 1)
    beta = torch.rand(1, 3, 1)
    kwargs = dict(g=g, beta=beta, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
    chunk_out, chunk_state = chunk(q, k, v, **kwargs)
    recurrent_out, recurrent_state = recurrent(q, k, v, **kwargs)
    torch.testing.assert_close(chunk_out, recurrent_out, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(chunk_state, recurrent_state, atol=1e-6, rtol=1e-5)
    x, state, weight = torch.randn(1, 6, 1), torch.randn(1, 6, 3), torch.randn(6, 3)
    combined = torch.cat((state, x), dim=-1)
    expected = torch.nn.functional.silu(
        torch.nn.functional.conv1d(combined, weight.unsqueeze(1), groups=6)[..., -1:]
    )
    out = conv_update(x, state, weight, None, "silu")
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(state, combined[..., -3:])


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
def test_actual_transformers_native_packed_output_and_gradients(torch_adapter, monkeypatch, model_type):
    adapter, torch = torch_adapter
    module = pytest.importorskip(f"transformers.models.{model_type}.modeling_{model_type}")
    config_module = pytest.importorskip(f"transformers.models.{model_type}.configuration_{model_type}")
    prefix = "Qwen3_5" if model_type == "qwen3_5" else "Qwen3_5Moe"
    config = getattr(config_module, f"{prefix}TextConfig")(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=3,
        layer_types=["linear_attention"],
    )
    torch.manual_seed(123)
    layer = getattr(module, f"{prefix}GatedDeltaNet")(config, 0).float()
    # Simulate pre-existing FLA attributes AND newer Transformers dispatchers.
    for name in ("chunk_gated_delta_rule", "recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update"):
        setattr(layer, name, _forbidden)
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_update"):
        monkeypatch.setattr(module, name, _dispatcher(unwrap(getattr(module, name))))
    monkeypatch.setenv("VERL_QWEN35_NATIVE_GDN", "1")
    forward = adapter["qwen3_5_gated_delta_net_forward"]
    x = torch.randn(1, 5, 8, requires_grad=True)
    offsets = torch.tensor([0, 2, 5])
    packed = forward(layer, x, cu_seqlens=offsets, cu_seqlens_cpu=offsets)
    independent = torch.cat((forward(layer, x[:, :2]), forward(layer, x[:, 2:])), dim=1)
    torch.testing.assert_close(packed, independent, atol=1e-6, rtol=1e-5)
    parameters = (x, *layer.parameters())
    packed_grad = torch.autograd.grad(packed.square().sum(), parameters)
    independent_grad = torch.autograd.grad(independent.square().sum(), parameters)
    for actual, expected in zip(packed_grad, independent_grad, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
