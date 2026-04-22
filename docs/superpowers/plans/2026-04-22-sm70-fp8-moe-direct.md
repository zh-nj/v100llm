# SM70 FP8 Direct MoE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SM70 FP8 direct MoE expert inference for `/mnt/data6/models/Qwen3.6-35B-A3B-FP8` while keeping expert weights resident as compressed FP8.

**Architecture:** Reuse the existing AWQ SM70 MoE runtime skeleton for token permutation, StridedPtr tables, workspace, warmup, and CUDA graph safety. Add FP8-specific expert prepare and grouped/batched GEMM ops that consume TurboMind packed `float8_e4m3fn` weights plus half scale layout. Add `Fp8SM70DirectMoEMethod` for serialized block-FP8 `FusedMoE` layers on capability 70.

**Tech Stack:** C++/CUDA custom ops, TurboMind SM70 GEMM, PyTorch custom op bindings, vLLM FP8 quantization, vLLM FusedMoE, pytest CUDA tests.

---

## File Structure

Modify `csrc/quantization/awq/awq_sm70_gemm.cu` to add FP8 MoE 3D prepare and grouped GEMM implementations next to the existing dense FP8 direct and AWQ MoE code.

Modify `csrc/ops.h` and `csrc/torch_bindings.cpp` to expose `sm70_fp8_moe_direct_prepare` and `sm70_fp8_moe_gemm_out`.

Modify `vllm/_custom_ops.py` to add Python wrappers and fake registrations for compile/fake tensor support.

Modify `vllm/model_executor/layers/quantization/fp8.py` to select and implement `Fp8SM70DirectMoEMethod`.

Modify `vllm/model_executor/warmup/awq_sm70_warmup.py` to discover and warm up direct FP8 MoE layers before CUDA graph capture.

Modify `tests/quantization/test_fp8_sm70.py` for Python quant method and lifecycle tests.

Modify `tests/kernels/quantization/test_sm70_fp8_runtime_decode.py` for CUDA numeric and CUDA graph tests.

Modify `tests/model_executor/test_sm70_runtime_warmup.py` for warmup coverage.

---

### Task 1: Add Python RED Tests For SM70 FP8 MoE Selection And Lifecycle

**Files:**
- Modify: `tests/quantization/test_fp8_sm70.py`

- [ ] **Step 1: Replace the old unsupported-MoE expectation with selection behavior**

Edit `test_fp8_sm70_serialized_moe_raises_clear_error` into:

```python
def test_fp8_sm70_serialized_moe_uses_direct_moe_method(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_single_rank_params(monkeypatch)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8._get_current_capability_int",
        lambda: 70,
        raising=False,
    )
    config = _make_fp8_config()
    layer = FusedMoE(
        num_experts=2,
        top_k=1,
        hidden_size=256,
        intermediate_size=128,
        tp_size=1,
        dp_size=1,
        pcp_size=1,
        params_dtype=torch.float16,
        quant_config=config,
        prefix="model.layers.0.mlp.experts",
    )

    assert layer.quant_method.__class__.__name__ == "Fp8SM70DirectMoEMethod"
    assert tuple(layer.w13_weight.shape) == (2, 256, 256)
    assert tuple(layer.w2_weight.shape) == (2, 256, 128)
    assert layer.w13_weight.dtype == torch.float8_e4m3fn
    assert layer.w2_weight.dtype == torch.float8_e4m3fn
    assert tuple(layer.w13_weight_scale_inv.shape) == (2, 2, 2)
    assert tuple(layer.w2_weight_scale_inv.shape) == (2, 2, 1)
```

- [ ] **Step 2: Add a lifecycle test that uses fake custom ops**

Append this test near the other SM70 FP8 tests:

```python
def test_fp8_sm70_moe_process_prepares_fp8_resident_weights(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_single_rank_params(monkeypatch)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8._get_current_capability_int",
        lambda: 70,
        raising=False,
    )
    ops_module = importlib.import_module("vllm._custom_ops")
    config = _make_fp8_config()
    layer = FusedMoE(
        num_experts=2,
        top_k=1,
        hidden_size=256,
        intermediate_size=128,
        tp_size=1,
        dp_size=1,
        pcp_size=1,
        params_dtype=torch.float16,
        quant_config=config,
        prefix="model.layers.0.mlp.experts",
    )
    layer.w13_weight.data.zero_()
    layer.w2_weight.data.zero_()
    layer.w13_weight_scale_inv.data.fill_(1)
    layer.w2_weight_scale_inv.data.fill_(1)

    prepare_calls = []

    def fake_prepare(weight, scale, block_n, block_k, interleave_gated_silu):
        prepare_calls.append(
            (
                tuple(weight.shape),
                tuple(scale.shape),
                block_n,
                block_k,
                interleave_gated_silu,
            )
        )
        prepared_weight = torch.empty_like(weight)
        prepared_scale = torch.empty(
            (weight.size(0), weight.size(2) // block_k, weight.size(1)),
            dtype=torch.float16,
            device=weight.device,
        )
        prepared_meta = torch.tensor(
            [weight.size(1), weight.size(2), block_k, 4096, weight.size(1)],
            dtype=torch.int64,
            device=weight.device,
        )
        return [prepared_weight, prepared_scale, prepared_meta]

    def fake_ptrs(weight, scale, k_ld, q_ld, num_experts):
        return [
            torch.empty((num_experts * 16,), dtype=torch.uint8, device=weight.device),
            torch.empty((num_experts * 16,), dtype=torch.uint8, device=weight.device),
        ]

    monkeypatch.setattr(
        ops_module,
        "sm70_fp8_moe_direct_prepare",
        fake_prepare,
        raising=False,
    )
    monkeypatch.setattr(
        ops_module,
        "awq_moe_build_strided_ptrs",
        fake_ptrs,
        raising=False,
    )

    layer.quant_method.process_weights_after_loading(layer)

    assert prepare_calls == [
        ((2, 256, 256), (2, 2, 2), 128, 128, True),
        ((2, 256, 128), (2, 2, 1), 128, 128, False),
    ]
    assert layer._sm70_fp8_moe_direct_prepared is True
    assert layer.sm70_batched_ready is True
    assert layer.w13_tm_weight.dtype == torch.float8_e4m3fn
    assert layer.w2_tm_weight.dtype == torch.float8_e4m3fn
    assert layer.w13_tm_scales.dtype == torch.float16
    assert layer.w2_tm_scales.dtype == torch.float16
    assert not hasattr(layer, "w13_weight")
    assert not hasattr(layer, "w2_weight")
    assert not hasattr(layer, "w13_weight_scale_inv")
    assert not hasattr(layer, "w2_weight_scale_inv")
```

- [ ] **Step 3: Add an apply test using fake permute/unpermute and fake FP8 MoE GEMM**

Append this test:

```python
def test_fp8_sm70_moe_apply_calls_fp8_moe_gemm(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fp8_module = importlib.import_module("vllm.model_executor.layers.quantization.fp8")
    ops_module = importlib.import_module("vllm._custom_ops")
    method_cls = getattr(fp8_module, "Fp8SM70DirectMoEMethod")
    dummy_moe_layer = torch.nn.Module()
    dummy_moe_layer.moe_config = type("MoeCfg", (), {"experts_per_token": 1})()
    method = method_cls(_make_fp8_config(), dummy_moe_layer)

    layer = torch.nn.Module()
    layer.sm70_batched_ready = True
    layer.sm70_num_experts = 2
    layer.sm70_hidden_logical_size = 256
    layer.sm70_w13_k_dim = 256
    layer.sm70_w13_n_dim = 256
    layer.sm70_w2_k_dim = 128
    layer.sm70_w2_n_dim = 256
    layer.sm70_intermediate_size = 128
    layer._buf_max_tokens = 32
    layer._buf_max_slots = 32
    layer._buf_top_k = 1
    layer._buf_output = torch.empty(32, 256, dtype=torch.float16)
    layer._buf_permuted_input = torch.empty(32, 256, dtype=torch.float16)
    layer._buf_sorted_output = torch.empty(32, 256, dtype=torch.float16)
    layer._buf_gate_up = torch.empty(32, 256, dtype=torch.float16)
    layer._buf_intermediate = torch.empty(32, 128, dtype=torch.float16)
    layer._buf_expert_offsets = torch.empty(3, dtype=torch.int32)
    layer._buf_expert_offsets64 = torch.empty(3, dtype=torch.int64)
    layer._buf_inv_permuted_idx = torch.empty(32, 1, dtype=torch.int32)
    layer._buf_topk_ids_i32 = torch.empty(32, 1, dtype=torch.int32)
    layer._buf_token_expert_indices = torch.arange(32, dtype=torch.int32).view(32, 1)
    layer._buf_permuted_idx = torch.empty(32, dtype=torch.int32)
    layer._buf_m_indices = torch.empty(32, dtype=torch.int32)
    layer.w13_strided_ptrs_w = torch.empty(32, dtype=torch.uint8)
    layer.w13_strided_ptrs_s = torch.empty(32, dtype=torch.uint8)
    layer.w2_strided_ptrs_w = torch.empty(32, dtype=torch.uint8)
    layer.w2_strided_ptrs_s = torch.empty(32, dtype=torch.uint8)

    def fake_permute(
        x, topk_ids, token_expert_indices, scales, num_experts,
        padded_num_experts, top_k, maybe_unused, permuted_input,
        expert_offsets64, inv_permuted_idx, permuted_idx, m_indices,
    ):
        permuted_input[: x.size(0)].copy_(x)
        expert_offsets64.copy_(torch.tensor([0, x.size(0), x.size(0)]))
        inv_permuted_idx.zero_()

    def fake_unpermute(sorted_output, topk_weights, inv_idx, offsets, top_k, output):
        output.copy_(sorted_output[: output.size(0)])

    gemm_calls = []

    def fake_fp8_moe_gemm_out(
        out, sorted_input, expert_offsets, ptrs_w, ptrs_s,
        num_experts, k, n, group_size, gated_silu=False,
    ):
        gemm_calls.append((tuple(out.shape), k, n, group_size, gated_silu))
        out.fill_(3 if gated_silu else 5)

    monkeypatch.setattr(torch.ops._moe_C, "moe_permute", fake_permute)
    monkeypatch.setattr(torch.ops._moe_C, "moe_unpermute", fake_unpermute)
    monkeypatch.setattr(
        fp8_module,
        "_moe_permute_accepts_scale_and_m_indices",
        lambda: True,
    )
    monkeypatch.setattr(
        ops_module,
        "sm70_fp8_moe_gemm_out",
        fake_fp8_moe_gemm_out,
        raising=False,
    )

    x = torch.ones(2, 256, dtype=torch.float16)
    topk_weights = torch.ones(2, 1, dtype=torch.float32)
    topk_ids = torch.zeros(2, 1, dtype=torch.int64)
    out = method.apply(layer, x, topk_weights, topk_ids, None)

    assert gemm_calls == [
        ((2, 128), 256, 256, 128, True),
        ((2, 256), 128, 256, 128, False),
    ]
    assert tuple(out.shape) == (2, 256)
    assert torch.all(out == 5)
```

- [ ] **Step 4: Run tests and verify RED**

Run:

```bash
pytest tests/quantization/test_fp8_sm70.py::test_fp8_sm70_serialized_moe_uses_direct_moe_method tests/quantization/test_fp8_sm70.py::test_fp8_sm70_moe_process_prepares_fp8_resident_weights tests/quantization/test_fp8_sm70.py::test_fp8_sm70_moe_apply_calls_fp8_moe_gemm -q
```

Expected: FAIL because `Fp8Config.get_quant_method()` still raises for SM70 serialized FP8 MoE and `Fp8SM70DirectMoEMethod` does not exist.

---

### Task 2: Implement Python `Fp8SM70DirectMoEMethod`

**Files:**
- Modify: `vllm/model_executor/layers/quantization/fp8.py`

- [ ] **Step 1: Add helper imports**

Add these imports near the top of `fp8.py`:

```python
from vllm.model_executor.layers.quantization.awq_sm70_moe import (
    _DEFAULT_PERSISTENT_MAX_TOKENS as _SM70_FP8_MOE_PERSISTENT_TOKENS,
    _moe_permute_accepts_scale_and_m_indices,
)
```

- [ ] **Step 2: Select the new method for SM70 serialized block FP8 MoE**

Replace the SM70 MoE `ValueError` branch in `Fp8Config.get_quant_method()` with:

```python
            if (
                self.is_checkpoint_fp8_serialized
                and _get_current_capability_int() == 70
            ):
                if self.supports_sm70_checkpoint_fallback():
                    return Fp8SM70DirectMoEMethod(self, layer)
                raise ValueError(
                    "sm70 serialized FP8 MoE requires dynamic block FP8 "
                    "with weight_block_size=[128, 128]."
                )
```

- [ ] **Step 3: Add `Fp8SM70DirectMoEMethod` class**

Add the class before `Fp8MoEMethod` so it is available when `get_quant_method()` runs:

```python
class Fp8SM70DirectMoEMethod(FusedMoEMethodBase):
    """SM70 direct FP8 MoE method using TurboMind grouped GEMM."""

    def __init__(self, quant_config: Fp8Config, layer: torch.nn.Module):
        super().__init__(layer.moe_config)
        self.quant_config = quant_config
        self.weight_block_size = quant_config.weight_block_size
        if self.weight_block_size != [128, 128]:
            raise ValueError(
                "SM70 FP8 MoE requires weight_block_size=[128, 128], "
                f"got {self.weight_block_size}."
            )
        if quant_config.activation_scheme != "dynamic":
            raise ValueError("SM70 FP8 MoE requires dynamic activation scheme.")

    @property
    def supports_eplb(self) -> bool:
        return True
```

- [ ] **Step 4: Implement `create_weights()`**

Continue the class with:

```python
    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = self.weight_block_size
        block_n, block_k = self.weight_block_size
        if hidden_size % block_k != 0:
            raise ValueError(
                "SM70 FP8 MoE requires hidden_size divisible by 128, "
                f"got {hidden_size}."
            )
        if intermediate_size_per_partition % block_n != 0:
            raise ValueError(
                "SM70 FP8 MoE requires intermediate_size_per_partition "
                f"divisible by 128, got {intermediate_size_per_partition}."
            )

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * (intermediate_size_per_partition // block_n),
                hidden_size // block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w2_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size // block_n,
                intermediate_size_per_partition // block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
        )
        layer.register_parameter("w13_weight_scale_inv", w13_scale)
        layer.register_parameter("w2_weight_scale_inv", w2_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        layer.w13_input_scale = None
        layer.w2_input_scale = None
```

- [ ] **Step 5: Implement post-load prepare and persistent buffers**

Add methods:

```python
    def _prepare_matrix(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        interleave_gated_silu: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_n, block_k = self.weight_block_size
        return ops.sm70_fp8_moe_direct_prepare(
            weight,
            scale.to(torch.float32),
            block_n,
            block_k,
            interleave_gated_silu,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        w13, w13_scale, w13_meta = self._prepare_matrix(
            layer.w13_weight,
            layer.w13_weight_scale_inv,
            True,
        )
        w2, w2_scale, w2_meta = self._prepare_matrix(
            layer.w2_weight,
            layer.w2_weight_scale_inv,
            False,
        )
        layer.w13_tm_weight = torch.nn.Parameter(w13, requires_grad=False)
        layer.w13_tm_scales = torch.nn.Parameter(w13_scale, requires_grad=False)
        layer.w2_tm_weight = torch.nn.Parameter(w2, requires_grad=False)
        layer.w2_tm_scales = torch.nn.Parameter(w2_scale, requires_grad=False)
        del layer.w13_weight, layer.w2_weight
        del layer.w13_weight_scale_inv, layer.w2_weight_scale_inv

        num_experts = int(w13.shape[0])
        w13_k_ld, w13_q_ld = int(w13_meta[3].item()), int(w13_meta[4].item())
        w2_k_ld, w2_q_ld = int(w2_meta[3].item()), int(w2_meta[4].item())
        w13_ptrs = ops.awq_moe_build_strided_ptrs(
            w13, w13_scale, w13_k_ld, w13_q_ld, num_experts
        )
        w2_ptrs = ops.awq_moe_build_strided_ptrs(
            w2, w2_scale, w2_k_ld, w2_q_ld, num_experts
        )
        layer.w13_strided_ptrs_w = torch.nn.Parameter(w13_ptrs[0], requires_grad=False)
        layer.w13_strided_ptrs_s = torch.nn.Parameter(w13_ptrs[1], requires_grad=False)
        layer.w2_strided_ptrs_w = torch.nn.Parameter(w2_ptrs[0], requires_grad=False)
        layer.w2_strided_ptrs_s = torch.nn.Parameter(w2_ptrs[1], requires_grad=False)
        layer.w13_strided_ptrs_w_rows = layer.w13_strided_ptrs_w.view(num_experts, -1)
        layer.w13_strided_ptrs_s_rows = layer.w13_strided_ptrs_s.view(num_experts, -1)
        layer.w2_strided_ptrs_w_rows = layer.w2_strided_ptrs_w.view(num_experts, -1)
        layer.w2_strided_ptrs_s_rows = layer.w2_strided_ptrs_s.view(num_experts, -1)

        layer.sm70_num_experts = num_experts
        layer.sm70_w13_n_dim = int(w13_meta[0].item())
        layer.sm70_w13_k_dim = int(w13_meta[1].item())
        layer.sm70_w2_n_dim = int(w2_meta[0].item())
        layer.sm70_w2_k_dim = int(w2_meta[1].item())
        layer.sm70_hidden_logical_size = layer.sm70_w2_n_dim
        layer.sm70_intermediate_size = layer.sm70_w2_k_dim
        layer.sm70_batched_ready = True
        layer._sm70_fp8_moe_direct_prepared = True
        self._allocate_buffers(layer, w13.device)
```

Add `_allocate_buffers()` using the same tensor names as `AWQSM70MoEMethod`:

```python
    def _allocate_buffers(self, layer: Module, device: torch.device) -> None:
        top_k = self.moe.experts_per_token
        persistent_tokens = _SM70_FP8_MOE_PERSISTENT_TOKENS
        max_slots = persistent_tokens * top_k
        hidden_size = layer.sm70_hidden_logical_size
        layer._buf_max_tokens = persistent_tokens
        layer._buf_max_slots = max_slots
        layer._buf_top_k = top_k
        layer._buf_expert_offsets = torch.empty(
            layer.sm70_num_experts + 1, dtype=torch.int32, device=device
        )
        layer._buf_expert_offsets64 = torch.empty(
            layer.sm70_num_experts + 1, dtype=torch.int64, device=device
        )
        layer._buf_gate_up = torch.empty(
            max_slots, layer.sm70_w13_n_dim, dtype=torch.float16, device=device
        )
        layer._buf_intermediate = torch.empty(
            max_slots, layer.sm70_intermediate_size, dtype=torch.float16, device=device
        )
        layer._buf_permuted_input = torch.empty(
            max_slots, hidden_size, dtype=torch.float16, device=device
        )
        layer._buf_sorted_output = torch.empty(
            max_slots, hidden_size, dtype=torch.float16, device=device
        )
        layer._buf_inv_permuted_idx = torch.empty(
            persistent_tokens, top_k, dtype=torch.int32, device=device
        )
        layer._buf_topk_ids_i32 = torch.empty(
            persistent_tokens, top_k, dtype=torch.int32, device=device
        )
        layer._buf_token_expert_indices = torch.arange(
            max_slots, dtype=torch.int32, device=device
        ).view(persistent_tokens, top_k)
        layer._buf_permuted_idx = torch.empty(max_slots, dtype=torch.int32, device=device)
        layer._buf_m_indices = torch.empty(max_slots, dtype=torch.int32, device=device)
        layer._buf_output = torch.empty(
            persistent_tokens, hidden_size, dtype=torch.float16, device=device
        )
```

- [ ] **Step 6: Implement apply path**

Add `_get_buffers()`, `_permute_tokens_by_expert()`, and `apply()` by copying the AWQ SM70 MoE logic and changing only the GEMM op:

```python
    def _get_buffers(self, layer: Module, total_slots: int, num_tokens: int):
        if total_slots <= layer._buf_max_slots and num_tokens <= layer._buf_max_tokens:
            return {
                "output": layer._buf_output[:num_tokens],
                "permuted_input": layer._buf_permuted_input[:total_slots],
                "sorted_output": layer._buf_sorted_output[:total_slots],
                "intermediate": layer._buf_intermediate[:total_slots],
                "expert_offsets": layer._buf_expert_offsets,
                "expert_offsets64": layer._buf_expert_offsets64,
                "inv_permuted_idx": layer._buf_inv_permuted_idx[:num_tokens],
                "topk_ids_i32": layer._buf_topk_ids_i32[:num_tokens],
                "token_expert_indices": layer._buf_token_expert_indices[:num_tokens],
                "permuted_idx": layer._buf_permuted_idx[:total_slots],
                "m_indices": layer._buf_m_indices[:total_slots],
            }
        device = layer._buf_output.device
        top_k = layer._buf_top_k
        hidden_size = layer.sm70_hidden_logical_size
        return {
            "output": torch.empty(num_tokens, hidden_size, dtype=torch.float16, device=device),
            "permuted_input": torch.empty(total_slots, hidden_size, dtype=torch.float16, device=device),
            "sorted_output": torch.empty(total_slots, hidden_size, dtype=torch.float16, device=device),
            "intermediate": torch.empty(total_slots, layer.sm70_intermediate_size, dtype=torch.float16, device=device),
            "expert_offsets": torch.empty(layer.sm70_num_experts + 1, dtype=torch.int32, device=device),
            "expert_offsets64": torch.empty(layer.sm70_num_experts + 1, dtype=torch.int64, device=device),
            "inv_permuted_idx": torch.empty(num_tokens, top_k, dtype=torch.int32, device=device),
            "topk_ids_i32": torch.empty(num_tokens, top_k, dtype=torch.int32, device=device),
            "token_expert_indices": torch.arange(total_slots, dtype=torch.int32, device=device).view(num_tokens, top_k),
            "permuted_idx": torch.empty(total_slots, dtype=torch.int32, device=device),
            "m_indices": torch.empty(total_slots, dtype=torch.int32, device=device),
        }

    def apply(
        self,
        layer: Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts_input
        if not getattr(layer, "sm70_batched_ready", False):
            raise RuntimeError("SM70 FP8 MoE batched runtime is not prepared.")
        num_tokens = x.shape[0]
        top_k = topk_ids.shape[1]
        total_slots = num_tokens * top_k
        buffers = self._get_buffers(layer, total_slots, num_tokens)
        output = buffers["output"]
        output.zero_()
        if total_slots == 0:
            return output

        topk_ids_i32 = buffers["topk_ids_i32"]
        topk_ids_i32.copy_(topk_ids, non_blocking=True)
        if _moe_permute_accepts_scale_and_m_indices():
            torch.ops._moe_C.moe_permute(
                x,
                topk_ids_i32,
                buffers["token_expert_indices"],
                None,
                layer.sm70_num_experts,
                layer.sm70_num_experts,
                top_k,
                None,
                buffers["permuted_input"],
                buffers["expert_offsets64"],
                buffers["inv_permuted_idx"],
                buffers["permuted_idx"],
                buffers["m_indices"],
            )
        else:
            torch.ops._moe_C.moe_permute(
                x,
                topk_ids_i32,
                buffers["token_expert_indices"],
                None,
                layer.sm70_num_experts,
                layer.sm70_num_experts,
                top_k,
                buffers["permuted_input"],
                buffers["expert_offsets64"],
                buffers["inv_permuted_idx"],
                buffers["permuted_idx"],
            )
        buffers["expert_offsets"].copy_(buffers["expert_offsets64"], non_blocking=True)

        ops.sm70_fp8_moe_gemm_out(
            buffers["intermediate"],
            buffers["permuted_input"],
            buffers["expert_offsets"],
            layer.w13_strided_ptrs_w,
            layer.w13_strided_ptrs_s,
            layer.sm70_num_experts,
            layer.sm70_w13_k_dim,
            layer.sm70_w13_n_dim,
            128,
            True,
        )
        ops.sm70_fp8_moe_gemm_out(
            buffers["sorted_output"],
            buffers["intermediate"],
            buffers["expert_offsets"],
            layer.w2_strided_ptrs_w,
            layer.w2_strided_ptrs_s,
            layer.sm70_num_experts,
            layer.sm70_w2_k_dim,
            layer.sm70_w2_n_dim,
            128,
            False,
        )
        torch.ops._moe_C.moe_unpermute(
            buffers["sorted_output"][:, : layer.sm70_hidden_logical_size],
            topk_weights,
            buffers["inv_permuted_idx"],
            buffers["expert_offsets64"],
            top_k,
            output,
        )
        return output
```

- [ ] **Step 7: Run Python tests and verify GREEN**

Run:

```bash
pytest tests/quantization/test_fp8_sm70.py::test_fp8_sm70_serialized_moe_uses_direct_moe_method tests/quantization/test_fp8_sm70.py::test_fp8_sm70_moe_process_prepares_fp8_resident_weights tests/quantization/test_fp8_sm70.py::test_fp8_sm70_moe_apply_calls_fp8_moe_gemm -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
git add tests/quantization/test_fp8_sm70.py vllm/model_executor/layers/quantization/fp8.py
git commit -m "feat: select sm70 direct fp8 moe method"
```

---

### Task 3: Add RED CUDA Kernel Tests For FP8 MoE Prepare And GEMM

**Files:**
- Modify: `tests/kernels/quantization/test_sm70_fp8_runtime_decode.py`

- [ ] **Step 1: Add 3D block-FP8 helpers**

Append helpers after `_to_block_fp8()`:

```python
def _to_block_fp8_3d(weight_fp16: torch.Tensor):
    weights = []
    scales = []
    for expert in range(weight_fp16.shape[0]):
        q, s = _to_block_fp8(weight_fp16[expert])
        weights.append(q)
        scales.append(s)
    return torch.stack(weights), torch.stack(scales)


def _sm70_decode_reference_block_3d(
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    decoded = []
    for expert in range(weight_fp8.shape[0]):
        decoded.append(
            _sm70_decode_reference_block(
                weight_fp8[expert],
                weight_scale[expert],
                128,
                128,
            )
        )
    return torch.stack(decoded)
```

- [ ] **Step 2: Add non-gated FP8 MoE GEMM numeric test**

Append:

```python
@pytest.mark.cuda
def test_sm70_fp8_moe_direct_gemm_matches_reference():
    _require_sm70()
    torch.manual_seed(10)
    num_experts = 2
    k = 256
    n = 256
    x = torch.randn(3, k, device="cuda", dtype=torch.float16) / 4
    w_ref = torch.randn(num_experts, n, k, device="cuda", dtype=torch.float16) / 4
    w_fp8, w_scale = _to_block_fp8_3d(w_ref)
    prepared_w, prepared_s, prepared_meta = ops.sm70_fp8_moe_direct_prepare(
        w_fp8,
        w_scale,
        128,
        128,
        False,
    )
    ptrs_w, ptrs_s = ops.awq_moe_build_strided_ptrs(
        prepared_w,
        prepared_s,
        int(prepared_meta[3].item()),
        int(prepared_meta[4].item()),
        num_experts,
    )
    expert_offsets = torch.tensor([0, 2, 3], dtype=torch.int32, device="cuda")
    out = torch.empty((3, n), dtype=torch.float16, device="cuda")
    ops.sm70_fp8_moe_gemm_out(
        out,
        x,
        expert_offsets,
        ptrs_w,
        ptrs_s,
        num_experts,
        k,
        n,
        128,
        False,
    )

    decoded = _sm70_decode_reference_block_3d(w_fp8, w_scale)
    ref = torch.empty_like(out)
    ref[:2] = x[:2] @ decoded[0].t()
    ref[2:] = x[2:] @ decoded[1].t()
    torch.testing.assert_close(out, ref, atol=6e-1, rtol=8e-2)
```

- [ ] **Step 3: Add gated SiLU test**

Append:

```python
@pytest.mark.cuda
def test_sm70_fp8_moe_direct_gemm_gated_silu_matches_reference():
    _require_sm70()
    torch.manual_seed(11)
    num_experts = 2
    k = 256
    intermediate = 128
    n = intermediate * 2
    x = torch.randn(3, k, device="cuda", dtype=torch.float16) / 4
    w_ref = torch.randn(num_experts, n, k, device="cuda", dtype=torch.float16) / 4
    w_fp8, w_scale = _to_block_fp8_3d(w_ref)
    prepared_w, prepared_s, prepared_meta = ops.sm70_fp8_moe_direct_prepare(
        w_fp8,
        w_scale,
        128,
        128,
        True,
    )
    ptrs_w, ptrs_s = ops.awq_moe_build_strided_ptrs(
        prepared_w,
        prepared_s,
        int(prepared_meta[3].item()),
        int(prepared_meta[4].item()),
        num_experts,
    )
    expert_offsets = torch.tensor([0, 2, 3], dtype=torch.int32, device="cuda")
    out = torch.empty((3, intermediate), dtype=torch.float16, device="cuda")
    ops.sm70_fp8_moe_gemm_out(
        out,
        x,
        expert_offsets,
        ptrs_w,
        ptrs_s,
        num_experts,
        k,
        n,
        128,
        True,
    )

    decoded = _sm70_decode_reference_block_3d(w_fp8, w_scale)
    ref = torch.empty_like(out)
    full0 = x[:2] @ decoded[0].t()
    full1 = x[2:] @ decoded[1].t()
    gate0, up0 = full0.chunk(2, dim=1)
    gate1, up1 = full1.chunk(2, dim=1)
    ref[:2] = torch.nn.functional.silu(gate0) * up0
    ref[2:] = torch.nn.functional.silu(gate1) * up1
    torch.testing.assert_close(out, ref, atol=7e-1, rtol=1e-1)
```

- [ ] **Step 4: Add CUDA graph capture test**

Append:

```python
@pytest.mark.cuda
def test_sm70_fp8_moe_direct_gemm_out_is_cuda_graph_capturable():
    _require_sm70()
    torch.manual_seed(12)
    num_experts = 2
    k = 256
    n = 256
    x = torch.randn(1, k, device="cuda", dtype=torch.float16) / 4
    w_ref = torch.randn(num_experts, n, k, device="cuda", dtype=torch.float16) / 4
    w_fp8, w_scale = _to_block_fp8_3d(w_ref)
    prepared_w, prepared_s, prepared_meta = ops.sm70_fp8_moe_direct_prepare(
        w_fp8,
        w_scale,
        128,
        128,
        False,
    )
    ptrs_w, ptrs_s = ops.awq_moe_build_strided_ptrs(
        prepared_w,
        prepared_s,
        int(prepared_meta[3].item()),
        int(prepared_meta[4].item()),
        num_experts,
    )
    expert_offsets = torch.tensor([0, 1, 1], dtype=torch.int32, device="cuda")
    out = torch.empty((1, n), dtype=torch.float16, device="cuda")

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        ops.sm70_fp8_moe_gemm_out(
            out, x, expert_offsets, ptrs_w, ptrs_s, num_experts, k, n, 128, False
        )
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.sm70_fp8_moe_gemm_out(
            out, x, expert_offsets, ptrs_w, ptrs_s, num_experts, k, n, 128, False
        )
    graph.replay()

    decoded = _sm70_decode_reference_block_3d(w_fp8, w_scale)
    ref = x @ decoded[0].t()
    torch.testing.assert_close(out, ref, atol=6e-1, rtol=8e-2)
```

- [ ] **Step 5: Run tests and verify RED**

Run on one SM70 GPU:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 pytest tests/kernels/quantization/test_sm70_fp8_runtime_decode.py::test_sm70_fp8_moe_direct_gemm_matches_reference tests/kernels/quantization/test_sm70_fp8_runtime_decode.py::test_sm70_fp8_moe_direct_gemm_gated_silu_matches_reference tests/kernels/quantization/test_sm70_fp8_runtime_decode.py::test_sm70_fp8_moe_direct_gemm_out_is_cuda_graph_capturable -q
```

Expected: FAIL because `sm70_fp8_moe_direct_prepare` and `sm70_fp8_moe_gemm_out` do not exist.

---

### Task 4: Implement C++ FP8 MoE Custom Ops And Bindings

**Files:**
- Modify: `csrc/ops.h`
- Modify: `csrc/torch_bindings.cpp`
- Modify: `csrc/quantization/awq/awq_sm70_gemm.cu`
- Modify: `vllm/_custom_ops.py`

- [ ] **Step 1: Add declarations in `csrc/ops.h`**

Add after `sm70_fp8_direct_prepare`:

```cpp
std::vector<torch::Tensor> sm70_fp8_moe_direct_prepare(
    torch::Tensor weight,
    torch::Tensor weight_scale,
    int64_t block_n,
    int64_t block_k,
    bool interleave_gated_silu);
```

Add after `sm70_fp8_direct_gemm_out`:

```cpp
void sm70_fp8_moe_gemm_out(torch::Tensor out,
                           torch::Tensor sorted_input,
                           torch::Tensor expert_offsets,
                           torch::Tensor strided_ptrs_w,
                           torch::Tensor strided_ptrs_s,
                           int64_t num_experts,
                           int64_t k,
                           int64_t n,
                           int64_t group_size,
                           bool gated_silu);
```

- [ ] **Step 2: Register schemas in `csrc/torch_bindings.cpp`**

Add after `sm70_fp8_direct_prepare` registration:

```cpp
  ops.def(
      "sm70_fp8_moe_direct_prepare(Tensor weight, Tensor weight_scale, "
      "int block_n, int block_k, bool interleave_gated_silu) -> Tensor[]");
  ops.impl("sm70_fp8_moe_direct_prepare", torch::kCUDA,
           &sm70_fp8_moe_direct_prepare);
```

Add after `awq_moe_gemm_sm70_out` registration:

```cpp
  ops.def(
      "sm70_fp8_moe_gemm_out(Tensor(a!) out, Tensor sorted_input, "
      "Tensor expert_offsets, Tensor strided_ptrs_w, Tensor strided_ptrs_s, "
      "int num_experts, int k, int n, int group_size, bool gated_silu) -> ()");
  ops.impl("sm70_fp8_moe_gemm_out", torch::kCUDA, &sm70_fp8_moe_gemm_out);
```

- [ ] **Step 3: Add interleave helpers in `awq_sm70_gemm.cu`**

Add inside `namespace vllm::awq_sm70::{namespace { ... }}` near `interleave_gated_silu_cols`:

```cpp
torch::Tensor interleave_gated_silu_rows(torch::Tensor tensor) {
  const int64_t n = tensor.size(0);
  TORCH_CHECK((n % 2) == 0,
              "sm70_fp8_moe_direct_prepare: gated_silu interleave requires even rows.");
  const int64_t half = n / 2;
  auto first = tensor.slice(0, 0, half);
  auto second = tensor.slice(0, half, n);
  return torch::stack({first, second}, 1).reshape(tensor.sizes()).contiguous();
}
```

- [ ] **Step 4: Implement 3D FP8 MoE prepare**

Add after `sm70_fp8_direct_prepare`:

```cpp
std::vector<torch::Tensor> sm70_fp8_moe_direct_prepare(
    torch::Tensor weight,
    torch::Tensor weight_scale,
    int64_t block_n,
    int64_t block_k,
    bool interleave_gated_silu) {
  TORCH_CHECK(weight.is_cuda(),
              "sm70_fp8_moe_direct_prepare: weight must be CUDA.");
  TORCH_CHECK(weight_scale.is_cuda(),
              "sm70_fp8_moe_direct_prepare: weight_scale must be CUDA.");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat8_e4m3fn,
              "sm70_fp8_moe_direct_prepare: weight must be float8_e4m3fn.");
  TORCH_CHECK(weight.dim() == 3,
              "sm70_fp8_moe_direct_prepare: weight must be 3D [E,N,K].");
  TORCH_CHECK(weight_scale.dim() == 3,
              "sm70_fp8_moe_direct_prepare: weight_scale must be 3D [E,Nb,Kb].");
  TORCH_CHECK(block_n == 128 && block_k == 128,
              "sm70_fp8_moe_direct_prepare: only block_n=block_k=128 is supported.");

  weight = weight.contiguous();
  weight_scale = weight_scale.to(torch::kFloat32).contiguous();
  const int64_t experts = weight.size(0);
  const int64_t logical_n = weight.size(1);
  const int64_t logical_k = weight.size(2);
  TORCH_CHECK(logical_n % block_n == 0 && logical_k % block_k == 0,
              "sm70_fp8_moe_direct_prepare: N and K must be multiples of block size.");
  TORCH_CHECK(weight_scale.size(0) == experts &&
                  weight_scale.size(1) == logical_n / block_n &&
                  weight_scale.size(2) == logical_k / block_k,
              "sm70_fp8_moe_direct_prepare: scale shape mismatch.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(weight));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto prepared_weight = torch::empty_like(weight);
  auto prepared_scale = torch::empty(
      {experts, logical_k / block_k, logical_n},
      torch::TensorOptions().dtype(torch::kFloat16).device(weight.device()));
  int64_t k_ld = 0;
  int64_t q_ld = 0;

  for (int64_t e = 0; e < experts; ++e) {
    auto weight_u16 = torch::empty(
        {logical_n, logical_k},
        torch::TensorOptions().dtype(torch::kInt16).device(weight.device()));
    auto scale_group = torch::empty(
        {logical_k / block_k, logical_n},
        torch::TensorOptions().dtype(torch::kFloat16).device(weight.device()));

    extend_fp8_e4m3_to_u16(weight_u16, weight.select(0, e), stream);
    expand_block_fp8_scales_to_group_half(
        scale_group, weight_scale.select(0, e), logical_n, block_n, stream);
    if (interleave_gated_silu) {
      weight_u16 = interleave_gated_silu_rows(weight_u16);
      scale_group = interleave_gated_silu_cols(scale_group).contiguous();
    }
    const int64_t expert_k_ld =
        pack_sm70_fp8_weight_into(
            weight_u16, prepared_weight.select(0, e), stream);
    const int64_t expert_q_ld = pack_sm70_fp8_scales_into(
        scale_group,
        prepared_scale.select(0, e),
        logical_n,
        logical_k,
        block_k,
        stream);
    if (e == 0) {
      k_ld = expert_k_ld;
      q_ld = expert_q_ld;
    } else {
      TORCH_CHECK(k_ld == expert_k_ld && q_ld == expert_q_ld,
                  "sm70_fp8_moe_direct_prepare: inconsistent expert ld.");
    }
  }

  auto prepared_meta = torch::tensor(
      std::vector<int64_t>{logical_n, logical_k, block_k, k_ld, q_ld},
      torch::TensorOptions().dtype(torch::kInt64));
  return {prepared_weight, prepared_scale, prepared_meta};
}
```

- [ ] **Step 5: Implement FP8 MoE grouped GEMM**

Add after `awq_moe_gemm_sm70_out`:

```cpp
void sm70_fp8_moe_gemm_out(
    torch::Tensor out,
    torch::Tensor sorted_input,
    torch::Tensor expert_offsets,
    torch::Tensor strided_ptrs_w,
    torch::Tensor strided_ptrs_s,
    int64_t num_experts,
    int64_t k,
    int64_t n,
    int64_t group_size,
    bool gated_silu) {
  TORCH_CHECK(sorted_input.is_cuda() &&
                  sorted_input.scalar_type() == torch::kFloat16,
              "sm70_fp8_moe_gemm: input must be CUDA float16.");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "sm70_fp8_moe_gemm: output must be CUDA float16.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32,
              "sm70_fp8_moe_gemm: expert_offsets must be CUDA int32.");
  TORCH_CHECK(group_size == 128,
              "sm70_fp8_moe_gemm: only group_size=128 is supported.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(sorted_input));
  const int device = sorted_input.get_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t total_tokens = sorted_input.size(0);
  if (total_tokens == 0) return;

  TORCH_CHECK(out.size(0) == total_tokens,
              "sm70_fp8_moe_gemm: output rows must match input rows.");
  TORCH_CHECK(out.stride(1) == 1,
              "sm70_fp8_moe_gemm: output must be row-major contiguous.");
  if (gated_silu) {
    TORCH_CHECK((n % 2) == 0,
                "sm70_fp8_moe_gemm: gated_silu requires even output dim.");
    TORCH_CHECK(out.size(1) == n / 2,
                "sm70_fp8_moe_gemm: gated_silu output cols must be n/2.");
  } else {
    TORCH_CHECK(out.size(1) == n,
                "sm70_fp8_moe_gemm: output cols must match n.");
  }

  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kFloat8_e4m3, turbomind::kHalf, true, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "sm70_fp8_moe_gemm: no compatible TurboMind converters.");

  turbomind::gemm::MatrixLayout desc_A{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(total_tokens),
      static_cast<int>(k),
      static_cast<int>(k),
  };
  desc_A.num = static_cast<int>(num_experts);
  desc_A.offsets = expert_offsets.data_ptr<int>();
  turbomind::gemm::MatrixLayout desc_U{};

  const auto order_w = conv_w->order;
  const bool is_A_w =
      turbomind::gemm::get_operand_tag(conv_w->pack) ==
      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;
  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }
  turbomind::gemm::MatrixLayout desc_B = w_desc;
  desc_B.type = turbomind::kFloat8_e4m3;
  desc_B.pack = conv_w->pack;
  if (is_A_w) {
    desc_B = turbomind::gemm::transpose(desc_B);
  }
  desc_B.ld = 0;
  desc_B.num = static_cast<int>(num_experts);

  const auto order_s = conv_s->order;
  const bool is_A_s =
      turbomind::gemm::get_operand_tag(conv_s->pack) ==
      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;
  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint16,
      order_s,
      static_cast<int>(n),
      static_cast<int>(k / group_size),
      static_cast<int>(n),
  };
  if (is_B_s) {
    std::swap(s_desc.rows, s_desc.cols);
    s_desc.order = ~s_desc.order;
  }
  turbomind::gemm::MatrixLayout desc_V = s_desc;
  desc_V.pack = conv_s->pack;
  if (is_A_s) {
    desc_V = turbomind::gemm::transpose(desc_V);
  }
  desc_V.ld = 0;
  desc_V.num = static_cast<int>(num_experts);

  turbomind::gemm::MatrixLayout desc_D{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(total_tokens),
      static_cast<int>(n),
      static_cast<int>(out.stride(0)),
  };
  desc_D.num = static_cast<int>(num_experts);
  desc_D.offsets = expert_offsets.data_ptr<int>();

  turbomind::gemm::Operation op{};
  op.dispatch = vllm::awq_sm70::awq_select_moe_dispatch_policy(
      device, static_cast<int>(total_tokens), static_cast<int>(n),
      static_cast<int>(k), static_cast<int>(num_experts),
      static_cast<int>(group_size), stream);
  op.epilogue = gated_silu ? turbomind::gemm::Epilogue::kGatedSilu
                           : turbomind::gemm::Epilogue::kNone;
  op.quant_a = {turbomind::gemm::QuantType::kNone, 0};
  op.quant_b = {turbomind::gemm::QuantType::kK, static_cast<int>(group_size)};
  op.batch_dim = 0;

  auto& workspace_holder = vllm::awq_sm70::get_workspace(device, stream);
  auto& gemm = vllm::awq_sm70::get_gemm(device);
  const int ec = gemm.Run(op,
                          1.f,
                          sorted_input.data_ptr(),
                          desc_A,
                          nullptr,
                          desc_U,
                          strided_ptrs_w.data_ptr(),
                          desc_B,
                          strided_ptrs_s.data_ptr(),
                          desc_V,
                          0.f,
                          out.data_ptr(),
                          desc_D,
                          out.data_ptr(),
                          desc_D,
                          workspace_holder.workspace,
                          stream);
  TORCH_CHECK(ec == 0, "sm70_fp8_moe_gemm: TurboMind batched GEMM failed.");
}
```

- [ ] **Step 6: Add global forwarding wrappers**

Add near the other global wrappers after namespace close:

```cpp
std::vector<torch::Tensor> sm70_fp8_moe_direct_prepare(
    torch::Tensor weight,
    torch::Tensor weight_scale,
    int64_t block_n,
    int64_t block_k,
    bool interleave_gated_silu) {
  return vllm::awq_sm70::sm70_fp8_moe_direct_prepare(
      weight, weight_scale, block_n, block_k, interleave_gated_silu);
}

void sm70_fp8_moe_gemm_out(torch::Tensor out,
                           torch::Tensor sorted_input,
                           torch::Tensor expert_offsets,
                           torch::Tensor strided_ptrs_w,
                           torch::Tensor strided_ptrs_s,
                           int64_t num_experts,
                           int64_t k,
                           int64_t n,
                           int64_t group_size,
                           bool gated_silu) {
  vllm::awq_sm70::sm70_fp8_moe_gemm_out(
      out,
      sorted_input,
      expert_offsets,
      strided_ptrs_w,
      strided_ptrs_s,
      num_experts,
      k,
      n,
      group_size,
      gated_silu);
}
```

- [ ] **Step 7: Add Python wrappers and fake registrations**

Add to `vllm/_custom_ops.py` near direct FP8 wrappers:

```python
def sm70_fp8_moe_direct_prepare(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    block_n: int,
    block_k: int,
    interleave_gated_silu: bool,
) -> list[torch.Tensor]:
    return torch.ops._C.sm70_fp8_moe_direct_prepare(
        weight,
        weight_scale,
        block_n,
        block_k,
        interleave_gated_silu,
    )


if hasattr(torch.ops._C, "sm70_fp8_moe_direct_prepare"):

    @register_fake("_C::sm70_fp8_moe_direct_prepare")
    def _sm70_fp8_moe_direct_prepare_fake(
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        block_n: int,
        block_k: int,
        interleave_gated_silu: bool,
    ) -> list[torch.Tensor]:
        packed_scale = torch.empty(
            (weight.size(0), weight.size(2) // block_k, weight.size(1)),
            dtype=torch.float16,
            device=weight.device,
        )
        meta = torch.empty((5,), dtype=torch.int64)
        return [torch.empty_like(weight), packed_scale, meta]
```

Add near AWQ MoE wrappers:

```python
def sm70_fp8_moe_gemm_out(
    out: torch.Tensor,
    sorted_input: torch.Tensor,
    expert_offsets: torch.Tensor,
    strided_ptrs_w: torch.Tensor,
    strided_ptrs_s: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    group_size: int,
    gated_silu: bool = False,
) -> None:
    torch.ops._C.sm70_fp8_moe_gemm_out(
        out,
        sorted_input,
        expert_offsets,
        strided_ptrs_w,
        strided_ptrs_s,
        num_experts,
        k,
        n,
        group_size,
        gated_silu,
    )


if hasattr(torch.ops._C, "sm70_fp8_moe_gemm_out"):

    @register_fake("_C::sm70_fp8_moe_gemm_out")
    def _sm70_fp8_moe_gemm_out_fake(
        out: torch.Tensor,
        sorted_input: torch.Tensor,
        expert_offsets: torch.Tensor,
        strided_ptrs_w: torch.Tensor,
        strided_ptrs_s: torch.Tensor,
        num_experts: int,
        k: int,
        n: int,
        group_size: int,
        gated_silu: bool = False,
    ) -> None:
        return None
```

- [ ] **Step 8: Build**

Run:

```bash
cmake --build build-force -j 8
```

Expected: exit code 0.

- [ ] **Step 9: Run kernel tests and verify GREEN**

Run:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 pytest tests/kernels/quantization/test_sm70_fp8_runtime_decode.py::test_sm70_fp8_moe_direct_gemm_matches_reference tests/kernels/quantization/test_sm70_fp8_runtime_decode.py::test_sm70_fp8_moe_direct_gemm_gated_silu_matches_reference tests/kernels/quantization/test_sm70_fp8_runtime_decode.py::test_sm70_fp8_moe_direct_gemm_out_is_cuda_graph_capturable -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

Run:

```bash
git add csrc/ops.h csrc/torch_bindings.cpp csrc/quantization/awq/awq_sm70_gemm.cu vllm/_custom_ops.py tests/kernels/quantization/test_sm70_fp8_runtime_decode.py
git commit -m "feat: add sm70 fp8 moe direct kernels"
```

---

### Task 5: Add Warmup Discovery And Tests

**Files:**
- Modify: `vllm/model_executor/warmup/awq_sm70_warmup.py`
- Modify: `tests/model_executor/test_sm70_runtime_warmup.py`

- [ ] **Step 1: Add RED warmup test**

Append to `tests/model_executor/test_sm70_runtime_warmup.py`:

```python
def test_sm70_awq_warmup_handles_direct_fp8_moe(monkeypatch):
    layer = torch.nn.Module()
    layer._sm70_fp8_moe_direct_prepared = True
    layer.sm70_num_experts = 2
    layer.sm70_w13_k_dim = 256
    layer.sm70_w13_n_dim = 256
    layer.sm70_w2_k_dim = 128
    layer.sm70_w2_n_dim = 256
    layer.sm70_intermediate_size = 128
    layer._buf_top_k = 1
    layer.w13_tm_weight = torch.zeros(
        (2, 256, 256), dtype=torch.float8_e4m3fn, device="cuda"
    )
    layer.w13_tm_scales = torch.empty((2, 2, 256), dtype=torch.float16, device="cuda")
    layer.w2_tm_weight = torch.zeros(
        (2, 256, 128), dtype=torch.float8_e4m3fn, device="cuda"
    )
    layer.w2_tm_scales = torch.empty((2, 1, 256), dtype=torch.float16, device="cuda")
    layer.w13_strided_ptrs_w = torch.empty(32, dtype=torch.uint8, device="cuda")
    layer.w13_strided_ptrs_s = torch.empty(32, dtype=torch.uint8, device="cuda")
    layer.w2_strided_ptrs_w = torch.empty(32, dtype=torch.uint8, device="cuda")
    layer.w2_strided_ptrs_s = torch.empty(32, dtype=torch.uint8, device="cuda")

    calls = []

    monkeypatch.setattr(
        "torch.cuda.get_device_capability",
        lambda *_args, **_kwargs: (7, 0),
    )

    def fake_sm70_fp8_moe_gemm_out(
        out, x, offsets, ptrs_w, ptrs_s, experts, k, n, group, gated=False
    ):
        calls.append((tuple(out.shape), tuple(x.shape), experts, k, n, group, gated))

    monkeypatch.setattr(
        "vllm.model_executor.warmup.awq_sm70_warmup.ops.sm70_fp8_moe_gemm_out",
        fake_sm70_fp8_moe_gemm_out,
        raising=False,
    )
    monkeypatch.setattr("torch.cuda.synchronize", lambda *_args, **_kwargs: None)

    sm70_awq_warmup(_DummyWorker(layer))

    assert calls
    assert calls[0] == ((1, 128), (1, 256), 2, 256, 256, 128, True)
    assert calls[1] == ((1, 256), (1, 128), 2, 128, 256, 128, False)
```

- [ ] **Step 2: Run warmup test and verify RED**

Run:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 pytest tests/model_executor/test_sm70_runtime_warmup.py::test_sm70_awq_warmup_handles_direct_fp8_moe -q
```

Expected: FAIL because warmup does not discover `_sm70_fp8_moe_direct_prepared`.

- [ ] **Step 3: Implement warmup iter and call path**

Add to `awq_sm70_warmup.py`:

```python
def _iter_unique_direct_fp8_moe_layers(model: torch.nn.Module) -> Iterable[torch.nn.Module]:
    seen: set[int] = set()
    for layer in model.modules():
        if not getattr(layer, "_sm70_fp8_moe_direct_prepared", False):
            continue
        ident = id(layer)
        if ident in seen:
            continue
        seen.add(ident)
        yield layer
```

Add warmup function:

```python
def _warmup_direct_fp8_moe_layers(
    moe_layers: list[torch.nn.Module],
    token_counts: list[int],
) -> int:
    calls = 0
    for layer in moe_layers:
        device = layer.w13_tm_weight.device
        top_k = int(layer._buf_top_k)
        for num_tokens in token_counts:
            total_slots = num_tokens * top_k
            expert_offsets = _build_balanced_offsets(
                total_slots, int(layer.sm70_num_experts), device
            )
            permuted_input = torch.empty(
                (total_slots, int(layer.sm70_w13_k_dim)),
                dtype=torch.float16,
                device=device,
            )
            intermediate = torch.empty(
                (total_slots, int(layer.sm70_intermediate_size)),
                dtype=torch.float16,
                device=device,
            )
            sorted_output = torch.empty(
                (total_slots, int(layer.sm70_w2_n_dim)),
                dtype=torch.float16,
                device=device,
            )
            ops.sm70_fp8_moe_gemm_out(
                intermediate,
                permuted_input,
                expert_offsets,
                layer.w13_strided_ptrs_w,
                layer.w13_strided_ptrs_s,
                int(layer.sm70_num_experts),
                int(layer.sm70_w13_k_dim),
                int(layer.sm70_w13_n_dim),
                128,
                True,
            )
            ops.sm70_fp8_moe_gemm_out(
                sorted_output,
                intermediate,
                expert_offsets,
                layer.w2_strided_ptrs_w,
                layer.w2_strided_ptrs_s,
                int(layer.sm70_num_experts),
                int(layer.sm70_w2_k_dim),
                int(layer.sm70_w2_n_dim),
                128,
                False,
            )
            calls += 2
    return calls
```

Wire it into `sm70_awq_warmup()` by adding `direct_fp8_moe_layers`, including its count in logs, and calling `_warmup_direct_fp8_moe_layers()`.

- [ ] **Step 4: Run warmup test and verify GREEN**

Run:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 pytest tests/model_executor/test_sm70_runtime_warmup.py::test_sm70_awq_warmup_handles_direct_fp8_moe -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add vllm/model_executor/warmup/awq_sm70_warmup.py tests/model_executor/test_sm70_runtime_warmup.py
git commit -m "feat: warm up sm70 direct fp8 moe"
```

---

### Task 6: Run Focused Regression Suite

**Files:**
- No source edits expected.

- [ ] **Step 1: Run focused Python tests**

Run:

```bash
pytest tests/quantization/test_fp8_sm70.py tests/model_executor/test_sm70_runtime_warmup.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run focused CUDA kernel tests**

Run:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 pytest tests/kernels/quantization/test_sm70_fp8_runtime_decode.py -q
```

Expected: all tests pass or non-SM70 tests skip only when CUDA device is not V100.

- [ ] **Step 3: Build once after all edits**

Run:

```bash
cmake --build build-force -j 8
```

Expected: exit code 0.

- [ ] **Step 4: Commit if any test-only fixes were needed**

Run this only if Task 6 required additional source edits:

```bash
git add csrc/ops.h csrc/torch_bindings.cpp csrc/quantization/awq/awq_sm70_gemm.cu vllm/_custom_ops.py vllm/model_executor/layers/quantization/fp8.py vllm/model_executor/warmup/awq_sm70_warmup.py tests/kernels/quantization/test_sm70_fp8_runtime_decode.py tests/quantization/test_fp8_sm70.py tests/model_executor/test_sm70_runtime_warmup.py
git commit -m "fix: stabilize sm70 fp8 moe tests"
```

---

### Task 7: Qwen3.6 FP8 MoE Service Smoke And Benchmark

**Files:**
- Create temporary scripts under `/tmp` only.
- No repo source edits expected.

- [ ] **Step 1: Free selected SM70 GPUs**

Inspect running vLLM servers:

```bash
pgrep -af "vllm serve"
```

If a previous test server is still using GPUs 2-5, terminate only that vLLM server process after confirming it belongs to this task:

```bash
kill <pid>
```

- [ ] **Step 2: Start Qwen3.6 FP8 MoE service with CUDA graph**

Use TP=2 first because model weights are about 36 GB on disk and a single V100 may not have enough usable memory:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5 \
PYTHONPATH=/mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split \
VLLM_SM70_FP8_DIRECT_GEMM=1 \
vllm serve /mnt/data6/models/Qwen3.6-35B-A3B-FP8 \
  --host 127.0.0.1 \
  --port 24164 \
  --served-model-name qwen36-35b-a3b-fp8-sm70-tp2 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --dtype float16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32832 \
  --compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256],"max_cudagraph_capture_size":256,"pass_config":{"fuse_norm_quant":false,"fuse_act_quant":false},"inductor_compile_config":{"combo_kernels":false,"benchmark_combo_kernel":false}}'
```

Expected log evidence:

```text
CUDAGraphMode.FULL_AND_PIECEWISE
Warming up SM70 AWQ/FP8 kernels (... direct FP8 MoE shapes ...)
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)
Capturing CUDA graphs (decode, FULL)
Starting vLLM server on http://127.0.0.1:24164
```

- [ ] **Step 3: Confirm `/v1/models`**

Run:

```bash
curl -sf http://127.0.0.1:24164/v1/models
```

Expected: JSON contains `"id":"qwen36-35b-a3b-fp8-sm70-tp2"`.

- [ ] **Step 4: Create benchmark script**

Create `/tmp/qwen36_fp8_moe_bench.py` with:

```python
import json
import re
import time
from typing import Any

import requests
from transformers import AutoTokenizer


BASE_URL = "http://127.0.0.1:24164/v1"
MODEL = "qwen36-35b-a3b-fp8-sm70-tp2"
MODEL_DIR = "/mnt/data6/models/Qwen3.6-35B-A3B-FP8"
END_VALUE = 64


def make_prompt(tokenizer: Any, target_tokens: int) -> str:
    prefix = (
        "You are a precise copy machine. Ignore the filler context until the task.\n"
        "Filler context:\n"
    )
    suffix = (
        f"\n\nTask: Output numbers 1 through {END_VALUE} separated by spaces. "
        f"After {END_VALUE} output DONE. Do not output DONE before {END_VALUE}.\n"
        "Answer:\n"
    )
    filler = " filler" * max(0, target_tokens - len(tokenizer.encode(prefix + suffix, add_special_tokens=False)))
    prompt = prefix + filler + suffix
    while len(tokenizer.encode(prompt, add_special_tokens=False)) < target_tokens:
        filler += " filler"
        prompt = prefix + filler + suffix
    while filler and len(tokenizer.encode(prompt, add_special_tokens=False)) > target_tokens:
        filler = filler[: filler.rfind(" filler")]
        prompt = prefix + filler + suffix
    return prompt


def run_case(tokenizer: Any, target_tokens: int) -> dict[str, Any]:
    prompt = make_prompt(tokenizer, target_tokens)
    prompt_tokens_local = len(tokenizer.encode(prompt, add_special_tokens=False))
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "stop": ["DONE"],
    }
    start = time.perf_counter()
    first_token_time = None
    finish_reason = None
    usage = None
    pieces = []
    with requests.post(f"{BASE_URL}/completions", json=payload, stream=True, timeout=300) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue
            data = raw_line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage") is not None:
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            text = choice.get("text") or ""
            if text:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                pieces.append(text)
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    end = time.perf_counter()
    output = "".join(pieces)
    if first_token_time is None:
        first_token_time = end
    prompt_tokens = int((usage or {}).get("prompt_tokens", prompt_tokens_local))
    completion_tokens = int((usage or {}).get("completion_tokens", len(tokenizer.encode(output, add_special_tokens=False))))
    ints = [int(x) for x in re.findall(r"\b\d+\b", output)]
    semantic_ok = ints == list(range(1, END_VALUE + 1))
    return {
        "target_prompt_tokens": target_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prefill_tokens_s": prompt_tokens / max(first_token_time - start, 1e-9),
        "decode_tokens_s": completion_tokens / max(end - first_token_time, 1e-9),
        "TTFT_s": first_token_time - start,
        "finish_reason": finish_reason,
        "semantic_quality": f"PASS: copied 1..{END_VALUE} exactly" if semantic_ok else f"FAIL: first integers were {ints[:16]}",
        "output_preview": output[:200],
    }


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    requests.post(f"{BASE_URL}/completions", json={"model": MODEL, "prompt": "Say OK.", "max_tokens": 8, "temperature": 0}, timeout=120).raise_for_status()
    print(json.dumps([run_case(tokenizer, 1024), run_case(tokenizer, 32000)], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run benchmark**

Run:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5 \
PYTHONPATH=/mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split \
python /tmp/qwen36_fp8_moe_bench.py
```

Expected: JSON for 1k and 32k containing `prefill_tokens_s`, `decode_tokens_s`, `TTFT_s`, `finish_reason`, and `semantic_quality`. `semantic_quality` must be `PASS` for both runs before claiming semantic correctness.

---

### Task 8: Final Verification And Report

**Files:**
- No source edits expected.

- [ ] **Step 1: Run complete focused verification**

Run:

```bash
cmake --build build-force -j 8
pytest tests/quantization/test_fp8_sm70.py tests/model_executor/test_sm70_runtime_warmup.py -q
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 pytest tests/kernels/quantization/test_sm70_fp8_runtime_decode.py -q
```

Expected: build exits 0 and tests pass.

- [ ] **Step 2: Confirm git status**

Run:

```bash
git status --short
```

Expected: only intentional SM70 FP8 MoE changes plus pre-existing unrelated files remain.

- [ ] **Step 3: Final response content**

Report:

```text
Implemented SM70 FP8 direct MoE for language_model routed experts.
Verified build command and focused pytest commands with pass counts.
Service endpoint and model id.
1k/32k benchmark table with prefill tokens/s, decode tokens/s, TTFT, finish_reason, semantic_quality.
Known limits: visual/MTP/KV cache not optimized in this round; first version requires 128x128 block FP8 and 128-aligned TP shards.
```
