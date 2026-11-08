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
"""Tiny one-GPU FSDP2 CPU-offloaded child LM-head regression.

Run: CUDA_VISIBLE_DEVICES=0 python -m pytest -q <this file>
"""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard

from verl.models.transformers.qwen3_5 import forward_with_torch_backend, forward_with_triton_backend

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires one CUDA GPU")


class _Body(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(128, 128)

    def forward(self, input_ids):
        hidden = self.embed(input_ids)
        return _Output(hidden)


class _Output:
    def __init__(self, hidden):
        self.hidden_states = hidden

    def __getitem__(self, index):
        assert index == 0
        return self.hidden_states


class _LM(torch.nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.model = _Body()
        self.lm_head = torch.nn.Linear(128, 128, bias=False)
        self.backend = backend
        self._verl_fused_kernels_backend = "torch"

    def forward(self, input_ids, shift_labels):
        fn = forward_with_triton_backend if self.backend == "triton" else forward_with_torch_backend
        out = fn(self, input_ids=input_ids, shift_labels=shift_labels, temperature=0.8)
        return out.log_probs, out.entropy


@pytest.fixture
def mesh(tmp_path):
    created = not dist.is_initialized()
    torch.cuda.set_device(0)
    if created:
        dist.init_process_group("nccl", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1)
    try:
        yield init_device_mesh("cuda", (1,))
    finally:
        if created:
            dist.destroy_process_group()


@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_fused_child_lm_head_cpu_offload_forward_backward(mesh, backend):
    torch.manual_seed(314)
    model = _LM(backend).cuda()
    reference = _LM(backend).cuda()
    reference.load_state_dict(model.state_dict())
    offload = CPUOffloadPolicy()
    # Match apply_fsdp2: lm_head is independently sharded under the root.
    fully_shard(model.lm_head, mesh=mesh, offload_policy=offload, reshard_after_forward=True)
    fully_shard(model, mesh=mesh, offload_policy=offload, reshard_after_forward=True)
    assert model.lm_head.weight.device.type == "cpu"
    input_ids = torch.tensor([[1, 3, 5, 7]], device="cuda")
    labels = torch.tensor([[3, 5, 7, 9]], device="cuda")
    with torch.no_grad():
        old_log_probs, old_entropy = model(input_ids, labels)
        logits = reference.lm_head(reference.model(input_ids)[0]).float() / 0.8
        log_distribution = logits.log_softmax(dim=-1)
        expected_log_probs = log_distribution.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        expected_entropy = -(log_distribution.exp() * log_distribution).sum(-1)
        # Triton uses lower-precision matmul; isolate FSDP correctness by also
        # comparing against the identical kernel on an unsharded reference.
        torch.testing.assert_close(
            old_log_probs.reshape_as(expected_log_probs), expected_log_probs, atol=1e-3, rtol=2e-4
        )
        torch.testing.assert_close(
            old_entropy.reshape_as(expected_entropy), expected_entropy, atol=5e-3, rtol=5e-4
        )
        ref_log_probs, ref_entropy = reference(input_ids, labels)
        torch.testing.assert_close(old_log_probs, ref_log_probs, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(old_entropy, ref_entropy, atol=1e-6, rtol=1e-6)
    assert model.lm_head.weight.device.type == "cpu"
    wrapped_method = model.lm_head._verl_fused_lm_head_forward
    # Two successive training microbatches test idempotent registration and hooks.
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        log_probs, entropy = model(input_ids, labels)
        assert model.lm_head._verl_fused_lm_head_forward is wrapped_method
        assert model.lm_head.weight.device.type == "cpu"
        ref_log_probs, ref_entropy = reference(input_ids, labels)
        # Tokenwise loss supplies contiguous upstream gradients, as required by
        # the existing Triton kernel (a bare sum supplies an expanded view).
        coefficients = torch.tensor([0.7, 0.9, 1.1, 1.3], device="cuda")
        loss = -(log_probs * coefficients).sum() - (0.03 * entropy * coefficients).sum()
        ref_loss = -(ref_log_probs * coefficients).sum() - (0.03 * ref_entropy * coefficients).sum()
        loss.backward()
        ref_loss.backward()
        assert model.lm_head.weight.device.type == "cpu"
        for name, parameter in model.named_parameters():
            gradient = parameter.grad.full_tensor()
            assert gradient.device.type == "cpu"
            assert torch.isfinite(gradient).all()
            torch.testing.assert_close(
                gradient.cuda(), dict(reference.named_parameters())[name].grad, atol=2e-5, rtol=2e-5
            )
