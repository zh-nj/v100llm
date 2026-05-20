# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Retired sparse prefill v2 compatibility shim.

The direct-cache sparse prefill v2 prototype was slower than the production
TileLang sparse prefill path and is no longer wired into runtime routing.
Keep this module only so old imports fail with a clear error instead of a
missing-module crash.
"""

from __future__ import annotations


def _retired(*_args, **_kwargs):
    raise RuntimeError(
        "sm70 sparse prefill v2 is retired; use VLLM_SM70_USE_TILELANG_SPARSE_PREFILL=1"
    )


flash_mla_sparse_prefill_v2 = _retired
