# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom-op boundary dtype shield for SM70 DeepSeek V4 Flash.

Under `FULL_DECODE_ONLY` + `splitting_ops` that include the DeepSeek V4
custom ops (`deepseek_v4_attention`, `sparse_attn_indexer`, `mhc_pre`,
`mhc_post`), inductor's graph partitioner may export the fp32
pre-cast intermediate of the RMSNorm `x = x * weight` expression to
the downstream sub-graph instead of the fp16 post-cast result. The
downstream custom op then calls an fp16 Linear (e.g.
`weights_proj`) and raises ``expected mat1 and mat2 to have the same
dtype, but got: float != c10::Half``.

This module defines a tiny helper that each affected custom op calls
at its entry to enforce the dtype contract. When the tensor already
has the expected dtype, the helper is a strict no-op — the R4 hot
path is unaffected.

See .kiro/specs/deepseek-v4-flash-compile-path-regression/ for the
full design and measurements.
"""
from __future__ import annotations

import os

import torch


def _shield_enabled() -> bool:
    """Env-gated toggle for the dtype shield.

    Default ON. Set ``VLLM_SM70_COMPILE_BOUNDARY_SHIELD=0`` to disable
    for bug-preservation testing (reverts to the pre-fix behavior so
    the original mismatch can be re-observed).
    """
    return os.environ.get("VLLM_SM70_COMPILE_BOUNDARY_SHIELD", "1") == "1"


def is_boundary_shield_enabled() -> bool:
    return _shield_enabled()


def ensure_boundary_dtype(x: torch.Tensor, expected: torch.dtype) -> torch.Tensor:
    """Return ``x`` cast to ``expected`` dtype if and only if it differs.

    Strict no-op when ``x.dtype == expected``, so this is safe to call
    unconditionally at custom-op entry points. Never allocates when the
    dtype already matches. Returns the **same tensor object** on the no-op
    path; the shield is idempotent.

    Args:
        x: the input tensor arriving at a custom-op boundary.
        expected: the dtype the custom op's downstream kernels require.

    Returns:
        ``x`` unchanged when ``x.dtype == expected`` or the shield is
        disabled; otherwise ``x.to(expected)``.
    """
    if not _shield_enabled():
        return x
    if x.dtype == expected:
        return x
    return x.to(expected)
