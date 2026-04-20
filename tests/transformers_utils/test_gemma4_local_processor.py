# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest

from vllm.transformers_utils.processor import get_processor


GEMMA4_LOCAL = Path("/mnt/data6/models/gemma-4-31B-it-AWQ-4bit")
pytestmark = pytest.mark.skipif(
    not GEMMA4_LOCAL.exists(),
    reason="local Gemma4 model is not available",
)


def test_local_gemma4_processor_loads():
    processor = get_processor(
        str(GEMMA4_LOCAL),
        trust_remote_code=True,
    )
    assert getattr(processor, "tokenizer", None) is not None
    assert processor.tokenizer.__class__.__name__.startswith("Gemma")
