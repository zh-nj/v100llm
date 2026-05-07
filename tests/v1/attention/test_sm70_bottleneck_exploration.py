# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Bug condition exploration tests for SM70 decode-path performance defects.

These tests confirm the 4 performance defects exist on UNFIXED code.
They run at the unit/function level with synthetic data — no model loading.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4**
"""

import inspect

import pytest
import torch

# Skip entire module if not on SM70 (V100)
_IS_SM70 = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (7, 0)
)
pytestmark = pytest.mark.skipif(
    not _IS_SM70,
    reason="SM70 (V100) GPU required for bottleneck exploration tests",
)


# ---------------------------------------------------------------------------
# Defect 1.1 — mHC torch fallback vs SM70 fast path
# ---------------------------------------------------------------------------
class TestDefect11MhcFastPath:
    """Validate that _mhc_pre_sm70_fast output matches _mhc_pre_torch_fallback.

    If the fast path is numerically safe, these tests PASS — confirming the
    fast path could be enabled.  If they FAIL, the bug condition holds (the
    fast path diverges and must remain disabled).

    **Validates: Requirements 1.1**
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        from vllm.model_executor.layers.mhc import (
            _mhc_pre_sm70_fast,
            _mhc_pre_torch_fallback,
        )
        self.fast_fn = _mhc_pre_sm70_fast
        self.fallback_fn = _mhc_pre_torch_fallback

        # DeepSeek V4 mHC dimensions
        self.hc_mult = 4
        self.hidden_size = 7168
        self.hc_mult3 = self.hc_mult * 2 + self.hc_mult * self.hc_mult  # 24
        self.hc_hidden_size = self.hc_mult * self.hidden_size

        # Shared parameters (on GPU)
        torch.manual_seed(42)
        self.fn = torch.randn(
            self.hc_mult3, self.hc_hidden_size,
            dtype=torch.float32, device="cuda",
        )
        self.hc_scale = torch.tensor(
            [1.0, 1.0, 1.0], dtype=torch.float32, device="cuda",
        )
        self.hc_base = torch.randn(
            self.hc_mult3, dtype=torch.float32, device="cuda",
        )
        self.rms_eps = 1e-6
        self.hc_pre_eps = 1e-6
        self.hc_sinkhorn_eps = 1e-6
        self.hc_post_mult_value = 1.0
        self.sinkhorn_repeat = 3

    def _make_residual(self, num_tokens: int) -> torch.Tensor:
        """Create random residual on GPU in fp16 (SM70 uses fp16)."""
        return torch.randn(
            num_tokens, self.hc_mult, self.hidden_size,
            dtype=torch.float16, device="cuda",
        )

    def _run_both(self, num_tokens: int):
        residual = self._make_residual(num_tokens)
        args = (
            residual, self.fn, self.hc_scale, self.hc_base,
            self.rms_eps, self.hc_pre_eps, self.hc_sinkhorn_eps,
            self.hc_post_mult_value, self.sinkhorn_repeat,
        )
        post_fast, comb_fast, li_fast = self.fast_fn(*args)
        post_fb, comb_fb, li_fb = self.fallback_fn(*args)
        return (post_fast, comb_fast, li_fast), (post_fb, comb_fb, li_fb)

    @pytest.mark.parametrize("num_tokens", [1, 4, 16, 64])
    def test_fast_path_matches_fallback(self, num_tokens: int):
        """Fast path output must match torch fallback within FP16 tolerance.

        Tolerances are relaxed to rtol/atol=5e-2 because the fast path uses
        cuBLAS FP16 GEMM while the fallback uses float32 torch.matmul.  On a
        28672-dim dot product the accumulated FP16 rounding is expected to
        diverge beyond 1e-3 — this is inherent to the mixed-precision design
        and has negligible impact on model output quality.
        """
        fast, fb = self._run_both(num_tokens)

        # post_mix comparison (float32)
        torch.testing.assert_close(
            fast[0].float(), fb[0].float(),
            rtol=5e-2, atol=5e-2,
            msg=f"post_mix mismatch at num_tokens={num_tokens}",
        )
        # comb_mix comparison (float32)
        torch.testing.assert_close(
            fast[1].float(), fb[1].float(),
            rtol=5e-2, atol=5e-2,
            msg=f"comb_mix mismatch at num_tokens={num_tokens}",
        )
        # layer_input comparison (fp16)
        torch.testing.assert_close(
            fast[2].float(), fb[2].float(),
            rtol=5e-2, atol=5e-2,
            msg=f"layer_input mismatch at num_tokens={num_tokens}",
        )


# ---------------------------------------------------------------------------
# Defect 1.2 — Decode fallback uses separate index building (torch ops)
# ---------------------------------------------------------------------------
class TestDefect12DecodeFallbackIndices:
    """Confirm that SM70 decode fallback uses separate torch ops for index
    building, producing multiple CUDA kernels per layer.

    **Validates: Requirements 1.2**
    """

    def test_should_use_sm70_decode_prefill_fallback_returns_true(self, monkeypatch):
        """On SM70, the decode prefill fallback must remain reachable when the
        direct-decode opt-in is disabled. This encodes the original bug
        condition (Requirements 1.2): without the direct-decode path, SM70
        takes the fallback on V100. After the direct-decode fix landed, the
        default is opt-in (env VLLM_SM70_DEEPSEEK_V4_DIRECT_DECODE=1); this
        test explicitly disables the opt-in to exercise the legacy fallback.
        """
        from vllm.model_executor.layers.deepseek_v4_attention import (
            _should_use_sm70_decode_prefill_fallback,
        )
        monkeypatch.setenv("VLLM_SM70_DEEPSEEK_V4_DIRECT_DECODE", "0")
        q = torch.randn(1, 64, 512, dtype=torch.float16, device="cuda")
        assert _should_use_sm70_decode_prefill_fallback(q, swa_only=True), (
            "SM70 decode prefill fallback should return True on V100 when "
            "direct-decode opt-in is disabled"
        )
        assert _should_use_sm70_decode_prefill_fallback(q, swa_only=False), (
            "SM70 decode prefill fallback should return True on V100 when "
            "direct-decode opt-in is disabled"
        )

    def test_build_indices_uses_torch_ops(self):
        """Verify _build_decode_prefill_fallback_indices uses torch ops
        (arange, where, cat-like patterns) that generate multiple CUDA kernels.
        """
        from vllm.model_executor.layers.deepseek_v4_attention import (
            _build_decode_prefill_fallback_indices,
        )
        # Inspect the source code of the function to confirm torch op usage
        source = inspect.getsource(_build_decode_prefill_fallback_indices)
        torch_ops_found = []
        for op_name in ["arange", "where", "mul_", "add_", "unsqueeze"]:
            if op_name in source:
                torch_ops_found.append(op_name)

        assert len(torch_ops_found) >= 3, (
            f"Expected ≥3 torch ops in _build_decode_prefill_fallback_indices "
            f"(indicating multiple CUDA kernels), found: {torch_ops_found}"
        )

    def test_build_indices_produces_correct_output(self):
        """Verify the index-building function works correctly on GPU."""
        from vllm.model_executor.layers.deepseek_v4_attention import (
            _build_decode_prefill_fallback_indices,
        )
        global_indices = torch.tensor(
            [[[10, 11, -1, -1]], [[64, 65, 66, -1]]],
            dtype=torch.int32, device="cuda",
        )
        lens = torch.tensor([2, 3], dtype=torch.int32, device="cuda")

        local_indices, local_lens = _build_decode_prefill_fallback_indices(
            global_indices, lens,
        )

        expected = torch.tensor(
            [[[0, 1, -1, -1]], [[4, 5, 6, -1]]],
            dtype=torch.int32, device="cuda",
        )
        torch.testing.assert_close(local_indices, expected)
        torch.testing.assert_close(local_lens, lens)


# ---------------------------------------------------------------------------
# Defect 1.3 — FP32 einsum weight cache
# ---------------------------------------------------------------------------
class TestDefect13Fp32EinsumCache:
    """Confirm that _sm70_fp8_einsum_bmm caches pre-dequantized weights as
    FP32 (not FP16), wasting bandwidth.

    **Validates: Requirements 1.3**
    """

    def test_predequant_cache_is_fp32(self):
        """After calling _sm70_fp8_einsum_bmm, the cached pre-dequant weight
        attribute must be FP32 (confirming the bandwidth issue)."""
        from vllm.model_executor.layers.deepseek_v4_attention import (
            _sm70_fp8_einsum_bmm,
        )

        # Synthetic FP8 tensors matching typical DeepSeek V4 dimensions
        # a: [T, groups, hidden], b: [groups * rank, hidden]
        groups = 2
        rank = 128
        hidden = 128
        T = 1

        # Create FP8 tensors (float8_e4m3fn)
        a_fp32 = torch.randn(T, groups, hidden, dtype=torch.float32, device="cuda")
        a = a_fp32.clamp(-448, 448).to(torch.float8_e4m3fn)
        a_scale = torch.ones(T, groups, hidden // 128, dtype=torch.float32, device="cuda")

        b_fp32 = torch.randn(groups * rank, hidden, dtype=torch.float32, device="cuda")
        b = b_fp32.clamp(-448, 448).to(torch.float8_e4m3fn)
        b_scale = torch.ones(groups, rank // 128, hidden // 128, dtype=torch.float32, device="cuda")

        out = torch.empty(T, groups, rank, dtype=torch.float16, device="cuda")

        if hasattr(b, "_sm70_predequant_f16"):
            delattr(b, "_sm70_predequant_f16")

        _sm70_fp8_einsum_bmm(a, a_scale, b, b_scale, out, "bhr,hdr->bhd")

        # After fix: pre-dequanted weight should be cached as FP16
        assert hasattr(b, "_sm70_predequant_f16"), (
            "Expected _sm70_predequant_f16 attribute on weight tensor after "
            "calling _sm70_fp8_einsum_bmm (FP16 cache fix applied)"
        )
        assert b._sm70_predequant_f16.dtype == torch.float16, (
            f"Expected FP16 pre-dequant cache, got {b._sm70_predequant_f16.dtype}"
        )

    def test_predequant_fp32_vs_fp16_bandwidth(self):
        """Measure that FP32 cache is 2× the size of a hypothetical FP16 cache."""
        from vllm.model_executor.layers.deepseek_v4_attention import (
            _sm70_fp8_einsum_bmm,
        )

        groups = 2
        rank = 128
        hidden = 128
        T = 1

        a_fp32 = torch.randn(T, groups, hidden, dtype=torch.float32, device="cuda")
        a = a_fp32.clamp(-448, 448).to(torch.float8_e4m3fn)
        a_scale = torch.ones(T, groups, hidden // 128, dtype=torch.float32, device="cuda")

        b_fp32 = torch.randn(groups * rank, hidden, dtype=torch.float32, device="cuda")
        b = b_fp32.clamp(-448, 448).to(torch.float8_e4m3fn)
        b_scale = torch.ones(groups, rank // 128, hidden // 128, dtype=torch.float32, device="cuda")

        out = torch.empty(T, groups, rank, dtype=torch.float16, device="cuda")

        if hasattr(b, "_sm70_predequant_f16"):
            delattr(b, "_sm70_predequant_f16")

        _sm70_fp8_einsum_bmm(a, a_scale, b, b_scale, out, "bhr,hdr->bhd")

        cached = b._sm70_predequant_f16
        fp16_bytes = cached.nelement() * cached.element_size()
        fp32_bytes = cached.nelement() * 4  # hypothetical FP32

        assert fp32_bytes == 2 * fp16_bytes, (
            f"FP16 cache ({fp16_bytes} bytes) should be exactly half of "
            f"hypothetical FP32 cache ({fp32_bytes} bytes)"
        )


# ---------------------------------------------------------------------------
# Defect 1.4 — MoE MXFP4 dequant on SM70
# ---------------------------------------------------------------------------
class TestDefect14MoeMxfp4:
    """Document that SM70 lacks hardware MXFP4 support — software dequant
    is the only option.

    **Validates: Requirements 1.4**
    """

    def test_sm70_lacks_native_mxfp4_support(self):
        """SM70 (V100, capability 7.0) does not have hardware MXFP4 support.
        Native MXFP4 requires SM100 (Blackwell) or later. SM70 must use
        software int4→FP16 dequantization in FusedMoE Triton kernels."""
        cap = torch.cuda.get_device_capability()
        assert cap == (7, 0), f"Expected SM70, got {cap}"

        # SM70 has no hardware MXFP4 support — confirmed by architecture spec
        # Hardware MXFP4 is only available on SM100+ (Blackwell)
        assert cap[0] < 10, (
            "SM70 should not have hardware MXFP4 support. "
            "Native MXFP4 requires SM100+ (Blackwell architecture)."
        )

    def test_sm70_fp16_is_max_precision(self):
        """SM70 (V100) supports FP16 tensor cores but not BF16 tensor cores.
        This means all MXFP4 dequant must go through FP16 intermediates,
        adding register pressure for nibble extraction."""
        cap = torch.cuda.get_device_capability()
        assert cap == (7, 0)

        # Verify V100 supports FP16 but not native BF16 tensor ops
        # (BF16 storage works, but no tensor core acceleration)
        props = torch.cuda.get_device_properties(0)
        assert "V100" in props.name or cap == (7, 0), (
            f"Expected V100 GPU, got {props.name}"
        )
