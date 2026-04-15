#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that disabling NVML still resolves the CUDA platform."""

import os

for key in ["CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"]:
    os.environ.pop(key, None)

os.environ["VLLM_DISABLE_NVML"] = "1"

import torch  # noqa: E402

assert not torch.cuda.is_initialized(), "CUDA initialized before import"

from vllm.platforms import current_platform  # noqa: E402

assert current_platform.device_type == "cuda", (
    f"Expected CUDA platform, got {current_platform.device_type!r}"
)
assert not torch.cuda.is_initialized(), (
    "CUDA was initialized while resolving current_platform with VLLM_DISABLE_NVML=1"
)
print("OK")
