# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest


def test_sparse_prefill_v2_module_is_retired():
    from vllm.v1.attention.ops import tilelang_sparse_prefill_v2 as v2

    with pytest.raises(RuntimeError, match="sparse prefill v2 is retired"):
        v2.flash_mla_sparse_prefill_v2()
