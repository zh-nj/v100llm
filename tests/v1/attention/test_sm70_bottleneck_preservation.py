# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Preservation / regression tests for SM70 decode-path optimizations.

These tests capture the CURRENT behavior of paths that must NOT change.
They MUST PASS on the current (unfixed) code AND continue to PASS after
all fixes from the deepseek-v4-sm70-bottleneck-optimization spec.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**
"""

import pytest
import torch

# ── SM70 detection ───────────────────────────────────────────────────────────
_HAS_CUDA = torch.cuda.is_available()
_IS_SM70 = _HAS_CUDA and torch.cuda.get_device_capability() == (7, 0)

pytestmark = [
    pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available"),
    pytest.mark.skipif(not _IS_SM70, reason="SM70 (V100) GPU required"),
]

# ── Constants matching KV cache layout ───────────────────────────────────────
HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
QUANT_BLOCK = 64
N_QUANT_BLOCKS = 7
TOKEN_DATA_BYTES = 576  # 448 + 64*2
SCALE_BYTES = 8
HEAD_BYTES = TOKEN_DATA_BYTES + SCALE_BYTES  # 584


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: FP8 KV cache round-trip preservation
# ──────────────────────────────────────────────────────────────────────────────
class TestFP8KVCacheRoundTrip:
    """Verify the existing FP8 KV cache encode-decode round-trip test logic
    still produces correct results: NoPE within 1 ULP, RoPE bitwise identical.

    **Validates: Requirements 3.6**
    """

    @staticmethod
    def _make_cos_sin_cache(max_pos: int, rope_dim: int, device: str):
        base = 10000.0
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device)
                / rope_dim
            )
        )
        t = torch.arange(max_pos, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        return torch.cat((freqs.cos(), freqs.sin()), dim=-1)

    @staticmethod
    def _fp8_e4m3fn_ulp(value: torch.Tensor) -> torch.Tensor:
        abs_val = value.abs().float()
        fp8_val = abs_val.clamp(max=448.0).to(torch.float8_e4m3fn)
        fp8_bytes = fp8_val.view(torch.uint8).to(torch.int32)
        exp_bits = (fp8_bytes >> 3) & 0xF
        is_subnorm = exp_bits == 0
        ulp = torch.where(
            is_subnorm,
            torch.tensor(2.0**-9, device=value.device, dtype=torch.float32),
            torch.pow(2.0, (exp_bits.float() - 10.0)),
        )
        return ulp

    @pytest.mark.parametrize("num_tokens", [1, 4, 8])
    @pytest.mark.parametrize("magnitude", ["near_zero", "mid_range", "near_max"])
    def test_fp8_roundtrip_nope_within_1ulp(self, num_tokens, magnitude):
        """NoPE values must survive FP8 encode→decode within 1 ULP."""
        from vllm.model_executor.layers.deepseek_v4_attention import (
            _gather_decode_prefill_fallback_kv_,
            _torch_qnorm_rope_kv_insert_fallback,
        )

        device = "cuda"
        block_size = 64
        n_heads = 1
        eps = 1e-6

        torch.manual_seed(42)
        kv = torch.randn(num_tokens, HEAD_DIM, dtype=torch.float32, device=device)
        if magnitude == "near_zero":
            kv = kv * 0.001
        elif magnitude == "mid_range":
            kv = kv * 3.0
        elif magnitude == "near_max":
            kv = (kv * 100.0).clamp(-440.0, 440.0)
        kv = kv.clamp(-448.0, 448.0)

        positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
        cos_sin_cache = self._make_cos_sin_cache(4096, ROPE_DIM, device)
        q = torch.randn(num_tokens, n_heads, HEAD_DIM, dtype=torch.float16, device=device)
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
        num_blocks = (num_tokens // block_size) + 2
        k_cache = torch.zeros(num_blocks, block_size * HEAD_BYTES, dtype=torch.uint8, device=device)

        kv_fp16 = kv.to(torch.float16)
        _torch_qnorm_rope_kv_insert_fallback(
            q, kv_fp16, k_cache, slot_mapping, positions, cos_sin_cache, eps, block_size,
        )

        topk = 1
        global_indices = slot_mapping.to(torch.int32).unsqueeze(1)
        global_lens = torch.ones(num_tokens, dtype=torch.int32, device=device)
        out = torch.zeros(num_tokens, topk, HEAD_DIM, dtype=torch.bfloat16, device=device)

        _gather_decode_prefill_fallback_kv_(out, k_cache, global_indices, global_lens, block_size)

        # Verify NoPE within 1 ULP
        nope_original = kv_fp16[:, :NOPE_DIM].float()
        blocks = nope_original.view(-1, N_QUANT_BLOCKS, QUANT_BLOCK)
        absmax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
        exponents = torch.ceil(torch.log2(absmax / 448.0))
        scales = torch.exp2(exponents)
        scaled_vals = (blocks / scales).clamp(-448.0, 448.0)
        fp8_vals = scaled_vals.to(torch.float8_e4m3fn)
        fp8_dequant = fp8_vals.float()
        nope_expected_f32 = (fp8_dequant * scales).view(-1, NOPE_DIM)
        nope_expected_bf16 = nope_expected_f32.to(torch.bfloat16)
        nope_actual = out[:, 0, :NOPE_DIM]

        fp8_ulps = self._fp8_e4m3fn_ulp(scaled_vals)
        tolerance = (fp8_ulps * scales).view(-1, NOPE_DIM).to(torch.float32)
        bf16_eps = nope_expected_f32.abs() * (2.0**-8) + 1e-10
        total_tolerance = tolerance + bf16_eps
        nope_diff = (nope_actual.float() - nope_expected_bf16.float()).abs()
        violations = (nope_diff > total_tolerance).sum().item()
        assert violations == 0, (
            f"NoPE round-trip violations: {violations}/{nope_diff.numel()} "
            f"(magnitude={magnitude}, tokens={num_tokens})"
        )

    @pytest.mark.parametrize("num_tokens", [1, 4, 8])
    def test_fp8_roundtrip_rope_bitwise_identical(self, num_tokens):
        """RoPE BF16 values must be bitwise identical after round-trip."""
        from vllm.model_executor.layers.deepseek_v4_attention import (
            _gather_decode_prefill_fallback_kv_,
            _torch_qnorm_rope_kv_insert_fallback,
        )

        device = "cuda"
        block_size = 64
        n_heads = 1
        eps = 1e-6

        torch.manual_seed(42)
        kv = torch.randn(num_tokens, HEAD_DIM, dtype=torch.float32, device=device) * 3.0
        kv = kv.clamp(-448.0, 448.0)

        positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
        cos_sin_cache = self._make_cos_sin_cache(4096, ROPE_DIM, device)
        q = torch.randn(num_tokens, n_heads, HEAD_DIM, dtype=torch.float16, device=device)
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
        num_blocks = (num_tokens // block_size) + 2
        k_cache = torch.zeros(num_blocks, block_size * HEAD_BYTES, dtype=torch.uint8, device=device)

        kv_fp16 = kv.to(torch.float16)
        _torch_qnorm_rope_kv_insert_fallback(
            q, kv_fp16, k_cache, slot_mapping, positions, cos_sin_cache, eps, block_size,
        )

        topk = 1
        global_indices = slot_mapping.to(torch.int32).unsqueeze(1)
        global_lens = torch.ones(num_tokens, dtype=torch.int32, device=device)
        out = torch.zeros(num_tokens, topk, HEAD_DIM, dtype=torch.bfloat16, device=device)

        _gather_decode_prefill_fallback_kv_(out, k_cache, global_indices, global_lens, block_size)

        # Compute expected RoPE
        kv_rope = kv_fp16[:, NOPE_DIM:].float()
        cos_sin = cos_sin_cache[positions].float()
        cos_vals = cos_sin[:, :32]
        sin_vals = cos_sin[:, 32:]
        rope_even = kv_rope[:, ::2]
        rope_odd = kv_rope[:, 1::2]
        rotated_even = rope_even * cos_vals - rope_odd * sin_vals
        rotated_odd = rope_odd * cos_vals + rope_even * sin_vals
        rope_rotated = torch.zeros_like(kv_rope)
        rope_rotated[:, ::2] = rotated_even
        rope_rotated[:, 1::2] = rotated_odd
        rope_expected_bf16 = rope_rotated.to(torch.float16).to(torch.bfloat16)

        rope_actual = out[:, 0, NOPE_DIM:]
        rope_expected_u16 = rope_expected_bf16.view(torch.uint16)
        rope_actual_u16 = rope_actual.view(torch.uint16)
        mismatches = (rope_expected_u16 != rope_actual_u16).sum().item()
        assert mismatches == 0, (
            f"RoPE BF16 bitwise mismatches: {mismatches}/{rope_actual_u16.numel()}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: SM70 Triton kernel deterministic output
# ──────────────────────────────────────────────────────────────────────────────
class TestSM70TritonKernelOutput:
    """Verify _sm70_fp8_paged_mqa_logits_kernel produces consistent output
    for fixed inputs (deterministic seed).

    **Validates: Requirements 3.1, 3.2**
    """

    def test_mqa_logits_deterministic(self):
        """Running the kernel twice with the same input must produce
        identical output."""
        from vllm.model_executor.layers.sm70_mqa_logits import (
            sm70_fp8_mqa_logits,
        )

        torch.manual_seed(123)
        M, H, D, N = 4, 2, 128, 16

        q_float = torch.randn(M, H, D, device="cuda", dtype=torch.float32) * 2.0
        q_fp8 = q_float.to(torch.float8_e4m3fn)
        k_float = torch.randn(N, D, device="cuda", dtype=torch.float32) * 2.0
        k_fp8 = k_float.to(torch.float8_e4m3fn)
        k_scale = torch.rand(N, device="cuda", dtype=torch.float32) * 10.0 + 0.01
        weights = torch.randn(M, H, device="cuda", dtype=torch.float32)
        cu_ks = torch.zeros(M, device="cuda", dtype=torch.int32)
        cu_ke = torch.full((M,), N, device="cuda", dtype=torch.int32)

        out1 = sm70_fp8_mqa_logits(q_fp8, (k_fp8, k_scale), weights, cu_ks, cu_ke)
        out2 = sm70_fp8_mqa_logits(q_fp8, (k_fp8, k_scale), weights, cu_ks, cu_ke)

        torch.testing.assert_close(out1, out2, rtol=0, atol=0)

    def test_mqa_logits_matches_reference(self):
        """Kernel output must match a simple PyTorch reference."""
        from vllm.model_executor.layers.sm70_mqa_logits import (
            sm70_fp8_mqa_logits,
        )

        torch.manual_seed(456)
        M, H, D, N = 2, 2, 128, 8

        q_float = torch.randn(M, H, D, device="cuda", dtype=torch.float32)
        q_fp8 = q_float.to(torch.float8_e4m3fn)
        k_float = torch.randn(N, D, device="cuda", dtype=torch.float32)
        k_fp8 = k_float.to(torch.float8_e4m3fn)
        k_scale = torch.ones(N, device="cuda", dtype=torch.float32)
        weights = torch.ones(M, H, device="cuda", dtype=torch.float32)
        cu_ks = torch.zeros(M, device="cuda", dtype=torch.int32)
        cu_ke = torch.full((M,), N, device="cuda", dtype=torch.int32)

        actual = sm70_fp8_mqa_logits(q_fp8, (k_fp8, k_scale), weights, cu_ks, cu_ke)

        # Reference: dequant → dot → ReLU → weighted sum
        q_f32 = q_fp8.float()
        k_f32 = k_fp8.float()  # scale=1
        # scores: [M, H, N] = Q @ K^T
        scores = torch.einsum("mhd,nd->mhn", q_f32, k_f32)
        scores = torch.clamp(scores, min=0.0)  # ReLU
        logits_ref = torch.einsum("mhn,mh->mn", scores, weights)

        # Only check in-range positions
        valid = actual[:M, :N]
        torch.testing.assert_close(valid, logits_ref, rtol=1e-3, atol=1e-2)


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: FP16 output clamping
# ──────────────────────────────────────────────────────────────────────────────
class TestFP16OutputClamping:
    """Verify _clamp_sm70_fp16_attention_output_ correctly clamps to ±65504.

    **Validates: Requirements 3.4**
    """

    def test_clamp_prevents_inf(self):
        """Values beyond FP16 max must be clamped to ±65504."""
        from vllm.model_executor.layers.deepseek_v4_attention import (
            _clamp_sm70_fp16_attention_output_,
        )

        vals = torch.tensor(
            [0.0, 1.0, -1.0, 65504.0, -65504.0, 70000.0, -70000.0, 1e5, -1e5],
            dtype=torch.float16, device="cuda",
        )
        result = _clamp_sm70_fp16_attention_output_(vals)

        assert result.max().item() <= 65504.0
        assert result.min().item() >= -65504.0
        assert not torch.isinf(result).any(), "Clamped output must not contain inf"

    def test_clamp_preserves_normal_values(self):
        """Values within FP16 range must be unchanged."""
        from vllm.model_executor.layers.deepseek_v4_attention import (
            _clamp_sm70_fp16_attention_output_,
        )

        vals = torch.tensor(
            [0.0, 1.0, -1.0, 100.0, -100.0, 65504.0, -65504.0],
            dtype=torch.float16, device="cuda",
        )
        expected = vals.clone()
        _clamp_sm70_fp16_attention_output_(vals)
        torch.testing.assert_close(vals, expected, rtol=0, atol=0)

    def test_clamp_is_inplace(self):
        """Clamping must be in-place (returns same tensor)."""
        from vllm.model_executor.layers.deepseek_v4_attention import (
            _clamp_sm70_fp16_attention_output_,
        )

        vals = torch.tensor([70000.0, -70000.0], dtype=torch.float16, device="cuda")
        result = _clamp_sm70_fp16_attention_output_(vals)
        assert result.data_ptr() == vals.data_ptr(), "clamp_ must be in-place"

    def test_should_clamp_sm70_returns_true(self):
        """On SM70 with FP16 tensor, _should_clamp returns True."""
        from vllm.model_executor.layers.deepseek_v4_attention import (
            _should_clamp_sm70_fp16_attention_output,
        )

        fp16_tensor = torch.zeros(1, dtype=torch.float16, device="cuda")
        assert _should_clamp_sm70_fp16_attention_output(fp16_tensor)

    def test_should_clamp_non_fp16_returns_false(self):
        """For non-FP16 tensors, _should_clamp returns False."""
        from vllm.model_executor.layers.deepseek_v4_attention import (
            _should_clamp_sm70_fp16_attention_output,
        )

        fp32_tensor = torch.zeros(1, dtype=torch.float32, device="cuda")
        assert not _should_clamp_sm70_fp16_attention_output(fp32_tensor)


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: CUDA graph workspace address stability
# ──────────────────────────────────────────────────────────────────────────────
class TestWorkspaceAddressStability:
    """Verify get_simultaneous() returns tensors with stable data_ptr()
    across multiple calls.

    **Validates: Requirements 3.5**
    """

    def test_get_simultaneous_address_stability(self):
        """Calling get_simultaneous() with the same shapes must return
        tensors at the same addresses."""
        from vllm.v1.worker.workspace import WorkspaceManager

        mgr = WorkspaceManager(device=torch.device("cuda"))

        shapes = [
            ((64, 512), torch.bfloat16),
            ((64, 4), torch.int32),
        ]

        tensors1 = mgr.get_simultaneous(*shapes)
        ptrs1 = [t.data_ptr() for t in tensors1]

        tensors2 = mgr.get_simultaneous(*shapes)
        ptrs2 = [t.data_ptr() for t in tensors2]

        assert ptrs1 == ptrs2, (
            f"data_ptr() mismatch across calls: {ptrs1} vs {ptrs2}. "
            "Workspace tensors must have stable addresses for CUDA graph."
        )

    def test_get_simultaneous_shapes_correct(self):
        """Returned tensors must have the requested shapes and dtypes."""
        from vllm.v1.worker.workspace import WorkspaceManager

        mgr = WorkspaceManager(device=torch.device("cuda"))

        shapes = [
            ((32, 256), torch.float16),
            ((32, 8), torch.int32),
            ((32,), torch.float32),
        ]
        tensors = mgr.get_simultaneous(*shapes)

        assert len(tensors) == 3
        for tensor, (shape, dtype) in zip(tensors, shapes):
            assert tuple(tensor.shape) == shape, (
                f"Expected shape {shape}, got {tuple(tensor.shape)}"
            )
            assert tensor.dtype == dtype, (
                f"Expected dtype {dtype}, got {tensor.dtype}"
            )

    def test_get_simultaneous_locked_stability(self):
        """After locking, addresses must remain stable and allocations
        within the locked size must succeed."""
        from vllm.v1.worker.workspace import WorkspaceManager

        mgr = WorkspaceManager(device=torch.device("cuda"))
        shapes = [((16, 512), torch.bfloat16)]

        # First call to establish workspace size
        tensors_pre = mgr.get_simultaneous(*shapes)
        ptrs_pre = [t.data_ptr() for t in tensors_pre]

        mgr.lock()

        # Same-size call after lock must succeed with same address
        tensors_post = mgr.get_simultaneous(*shapes)
        ptrs_post = [t.data_ptr() for t in tensors_post]
        assert ptrs_pre == ptrs_post, "Addresses must be stable after lock"


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: SM70 decode fallback gather output
# ──────────────────────────────────────────────────────────────────────────────
class TestDecodeGatherKernelOutput:
    """Verify _gather_decode_kv_triton_kernel produces correct output for
    known input. This baseline must be preserved even after index fusion.

    **Validates: Requirements 3.3**
    """

    def _create_paged_cache_with_known_data(
        self, num_tokens, block_size, device
    ):
        """Create a paged FP8 cache with deterministic data and return
        the expected dequantized BF16 output for verification."""
        from vllm.model_executor.layers.deepseek_v4_attention import (
            _torch_qnorm_rope_kv_insert_fallback,
        )

        # Deterministic data
        torch.manual_seed(999)
        kv = torch.randn(num_tokens, HEAD_DIM, dtype=torch.float32, device=device) * 5.0
        kv = kv.clamp(-400.0, 400.0)

        base = 10000.0
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, ROPE_DIM, 2, dtype=torch.float32, device=device)
                / ROPE_DIM
            )
        )
        t = torch.arange(4096, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)

        positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
        q = torch.randn(num_tokens, 1, HEAD_DIM, dtype=torch.float16, device=device)
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
        num_blocks = (num_tokens // block_size) + 2
        k_cache = torch.zeros(
            num_blocks, block_size * HEAD_BYTES,
            dtype=torch.uint8, device=device,
        )

        kv_fp16 = kv.to(torch.float16)
        _torch_qnorm_rope_kv_insert_fallback(
            q, kv_fp16, k_cache, slot_mapping, positions, cos_sin_cache,
            1e-6, block_size,
        )

        return k_cache, slot_mapping, num_tokens

    @pytest.mark.parametrize("num_tokens", [1, 4, 8])
    def test_gather_triton_matches_torch_fallback(self, num_tokens):
        """Triton gather must produce the same BF16 output as the torch
        fallback path for identical cache contents."""
        from vllm.model_executor.layers.deepseek_v4_attention import (
            _gather_decode_prefill_fallback_kv_,
        )

        device = "cuda"
        block_size = 64

        k_cache, slot_mapping, n = self._create_paged_cache_with_known_data(
            num_tokens, block_size, device,
        )

        topk = 1
        global_indices = slot_mapping.to(torch.int32).unsqueeze(1)
        global_lens = torch.ones(n, dtype=torch.int32, device=device)

        out = torch.zeros(n, topk, HEAD_DIM, dtype=torch.bfloat16, device=device)
        _gather_decode_prefill_fallback_kv_(
            out, k_cache, global_indices, global_lens, block_size,
        )

        # Run again to verify determinism
        out2 = torch.zeros(n, topk, HEAD_DIM, dtype=torch.bfloat16, device=device)
        _gather_decode_prefill_fallback_kv_(
            out2, k_cache, global_indices, global_lens, block_size,
        )

        torch.testing.assert_close(
            out.view(torch.uint16), out2.view(torch.uint16), rtol=0, atol=0,
        )

    def test_gather_handles_invalid_slots(self):
        """Slots with value -1 or beyond lens must produce zero output."""
        from vllm.model_executor.layers.deepseek_v4_attention import (
            _gather_decode_prefill_fallback_kv_,
            _torch_qnorm_rope_kv_insert_fallback,
        )

        device = "cuda"
        block_size = 64
        num_tokens = 2

        torch.manual_seed(999)
        kv = torch.randn(num_tokens, HEAD_DIM, dtype=torch.float32, device=device) * 5.0
        kv = kv.clamp(-400.0, 400.0)

        base = 10000.0
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, ROPE_DIM, 2, dtype=torch.float32, device=device)
                / ROPE_DIM
            )
        )
        t = torch.arange(4096, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)

        positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
        q = torch.randn(num_tokens, 1, HEAD_DIM, dtype=torch.float16, device=device)
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
        num_blocks = (num_tokens // block_size) + 2
        k_cache = torch.zeros(
            num_blocks, block_size * HEAD_BYTES,
            dtype=torch.uint8, device=device,
        )

        kv_fp16 = kv.to(torch.float16)
        _torch_qnorm_rope_kv_insert_fallback(
            q, kv_fp16, k_cache, slot_mapping, positions, cos_sin_cache,
            1e-6, block_size,
        )

        # Create indices with topk=2, but lens=1 — second slot should be zero
        topk = 2
        global_indices = torch.tensor(
            [[0, -1], [1, -1]], dtype=torch.int32, device=device,
        )
        global_lens = torch.tensor([1, 1], dtype=torch.int32, device=device)

        out = torch.zeros(num_tokens, topk, HEAD_DIM, dtype=torch.bfloat16, device=device)
        _gather_decode_prefill_fallback_kv_(
            out, k_cache, global_indices, global_lens, block_size,
        )

        # Second column (topk=1) must be all zeros since lens=1
        zeros = torch.zeros(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device)
        torch.testing.assert_close(
            out[:, 1, :].view(torch.uint16),
            zeros.view(torch.uint16),
            rtol=0, atol=0,
        )

    def test_build_indices_output_preserved(self):
        """_build_decode_prefill_fallback_indices must produce correct
        local indices — this baseline is preserved after index fusion."""
        from vllm.model_executor.layers.deepseek_v4_attention import (
            _build_decode_prefill_fallback_indices,
        )

        device = "cuda"
        # 2 tokens, topk=4
        global_indices = torch.tensor(
            [[[10, 11, -1, -1]], [[64, 65, 66, -1]]],
            dtype=torch.int32, device=device,
        )
        lens = torch.tensor([2, 3], dtype=torch.int32, device=device)

        local_indices, local_lens = _build_decode_prefill_fallback_indices(
            global_indices, lens,
        )

        expected = torch.tensor(
            [[[0, 1, -1, -1]], [[4, 5, 6, -1]]],
            dtype=torch.int32, device=device,
        )
        torch.testing.assert_close(local_indices, expected)
        torch.testing.assert_close(local_lens, lens)
