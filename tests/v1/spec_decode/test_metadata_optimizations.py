# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from vllm.v1.spec_decode.metadata import build_spec_decode_metadata_indices
from vllm.v1.spec_decode.utils import should_skip_intermediate_prefill_draft


def test_build_spec_decode_metadata_indices_matches_documented_example():
    num_draft_tokens = np.array([3, 0, 2, 0, 1], dtype=np.int32)
    cu_num_scheduled_tokens = np.array([4, 104, 107, 207, 209])

    metadata = build_spec_decode_metadata_indices(
        num_draft_tokens,
        cu_num_scheduled_tokens,
        np.arange(32, dtype=np.int64),
        np.empty(32, dtype=np.int64),
    )

    assert metadata.cu_num_draft_tokens.tolist() == [3, 3, 5, 5, 6]
    assert metadata.cu_num_sampled_tokens.tolist() == [4, 5, 8, 9, 11]
    assert metadata.logits_indices.tolist() == [
        0,
        1,
        2,
        3,
        103,
        104,
        105,
        106,
        206,
        207,
        208,
    ]
    assert metadata.target_logits_indices.tolist() == [0, 1, 2, 5, 6, 9]
    assert metadata.bonus_logits_indices.tolist() == [3, 4, 7, 8, 10]


def test_should_skip_intermediate_prefill_draft_only_for_non_final_chunks():
    assert should_skip_intermediate_prefill_draft(
        num_computed_tokens=np.array([0, 4096], dtype=np.int32),
        num_prompt_tokens=np.array([32000, 32000], dtype=np.int32),
        num_scheduled_tokens=np.array([4096, 4096], dtype=np.int32),
    )

    assert not should_skip_intermediate_prefill_draft(
        num_computed_tokens=np.array([28672], dtype=np.int32),
        num_prompt_tokens=np.array([32768], dtype=np.int32),
        num_scheduled_tokens=np.array([4096], dtype=np.int32),
    )

    assert not should_skip_intermediate_prefill_draft(
        num_computed_tokens=np.array([32768], dtype=np.int32),
        num_prompt_tokens=np.array([32768], dtype=np.int32),
        num_scheduled_tokens=np.array([2], dtype=np.int32),
    )
