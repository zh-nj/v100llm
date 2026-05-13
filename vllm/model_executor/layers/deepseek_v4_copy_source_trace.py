# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional NVTX source tags for DeepSeek V4 copy/round-trip tracing."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import torch


def _copy_source_trace_enabled() -> bool:
    return os.getenv("VLLM_DEEPSEEK_V4_COPY_SOURCE_TRACE", "0") == "1"


def _torch_compiler_is_compiling() -> bool:
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:
        return False


@contextmanager
def copy_source_trace(label: str) -> Iterator[None]:
    """Emit an NVTX range for source attribution of generic copy kernels.

    Unlike the CUDA-event phase profiler, this helper intentionally does not
    skip CUDA graph capture. Its purpose is to let nsys associate captured graph
    nodes such as generic `copy.float` kernels with a narrower source range.
    """
    pushed = False
    if (
        _copy_source_trace_enabled()
        and torch.cuda.is_available()
        and not _torch_compiler_is_compiling()
    ):
        try:
            torch.cuda.nvtx.range_push(f"copy_source.{label}")
            pushed = True
        except Exception:
            pushed = False
    try:
        yield
    finally:
        if pushed:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass
