# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SpecDecodeMetadataIndices:
    cu_num_draft_tokens: np.ndarray
    cu_num_sampled_tokens: np.ndarray
    logits_indices: np.ndarray
    target_logits_indices: np.ndarray
    bonus_logits_indices: np.ndarray


def _get_cumsum_and_arange(
    num_tokens: np.ndarray,
    arange_np: np.ndarray,
    arange_out: np.ndarray,
    cumsum_dtype: np.dtype | None = None,
) -> np.ndarray:
    cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
    total_num_tokens = int(cu_num_tokens[-1])
    cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
    np.subtract(
        arange_np[:total_num_tokens],
        cumsums_offsets,
        out=arange_out[:total_num_tokens],
    )
    return cu_num_tokens


def build_spec_decode_metadata_indices(
    num_draft_tokens: np.ndarray,
    cu_num_scheduled_tokens: np.ndarray,
    arange_np: np.ndarray,
    arange_scratch: np.ndarray,
) -> SpecDecodeMetadataIndices:
    # Inputs:
    # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
    # num_draft_tokens:         [  3,   0,   2,   0,   1]
    # Outputs:
    # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
    # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
    #                            206, 207, 208]
    # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
    # bonus_logits_indices:     [  3,   4,   7,   8,  10]
    num_sampled_tokens = num_draft_tokens + 1

    cu_num_sampled_tokens = _get_cumsum_and_arange(
        num_sampled_tokens,
        arange_np,
        arange_scratch,
        cumsum_dtype=np.int32,
    )
    total_num_sampled_tokens = int(cu_num_sampled_tokens[-1])
    logits_indices = np.repeat(
        cu_num_scheduled_tokens - num_sampled_tokens,
        num_sampled_tokens,
    )
    logits_indices += arange_scratch[:total_num_sampled_tokens]

    bonus_logits_indices = cu_num_sampled_tokens - 1

    cu_num_draft_tokens = _get_cumsum_and_arange(
        num_draft_tokens,
        arange_np,
        arange_scratch,
        cumsum_dtype=np.int32,
    )
    total_num_draft_tokens = int(cu_num_draft_tokens[-1])
    target_logits_indices = np.repeat(
        cu_num_sampled_tokens - num_sampled_tokens,
        num_draft_tokens,
    )
    target_logits_indices += arange_scratch[:total_num_draft_tokens]

    return SpecDecodeMetadataIndices(
        cu_num_draft_tokens=cu_num_draft_tokens,
        cu_num_sampled_tokens=cu_num_sampled_tokens,
        logits_indices=logits_indices,
        target_logits_indices=target_logits_indices,
        bonus_logits_indices=bonus_logits_indices,
    )


@dataclass
class SpecDecodeMetadata:
    # [num_tokens]
    draft_token_ids: torch.Tensor
    # [batch_size]
    num_draft_tokens: list[int]
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor
    # [batch_size]
    cu_num_sampled_tokens: torch.Tensor
    # [num_tokens]
    target_logits_indices: torch.Tensor
    # [batch_size]
    bonus_logits_indices: torch.Tensor
    # [num_tokens + batch_size]
    logits_indices: torch.Tensor

    def __post_init__(self):
        self.max_spec_len = max(self.num_draft_tokens)

    @classmethod
    def make_dummy(
        cls,
        draft_token_ids: list[list[int]],
        device: torch.device,
    ) -> "SpecDecodeMetadata":
        batch_size = len(draft_token_ids)
        num_draft_tokens = [len(ids) for ids in draft_token_ids]
        num_sampled_tokens = [len(ids) + 1 for ids in draft_token_ids]
        flattened_draft_token_ids = sum(draft_token_ids, [])
        num_tokens = len(flattened_draft_token_ids)

        draft_token_ids_tensor = torch.tensor(
            flattened_draft_token_ids, dtype=torch.int32, device=device
        )
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        cu_num_draft_tokens_tensor = torch.from_numpy(cu_num_draft_tokens).to(device)
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
        cu_num_sampled_tokens_tensor = torch.from_numpy(cu_num_sampled_tokens).to(
            device
        )

        target_logits_indices = torch.zeros(
            num_tokens, dtype=torch.int32, device=device
        )
        bonus_logits_indices = torch.zeros(batch_size, dtype=torch.int32, device=device)
        logits_indices = torch.zeros(
            num_tokens + batch_size, dtype=torch.int32, device=device
        )
        return cls(
            draft_token_ids=draft_token_ids_tensor,
            num_draft_tokens=num_draft_tokens,
            cu_num_draft_tokens=cu_num_draft_tokens_tensor,
            cu_num_sampled_tokens=cu_num_sampled_tokens_tensor,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )
