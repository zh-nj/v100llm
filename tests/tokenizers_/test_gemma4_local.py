# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest
from transformers import AutoTokenizer

from vllm.tokenizers.hf import CachedHfTokenizer


GEMMA4_LOCAL = Path("/mnt/data6/models/gemma-4-31B-it-AWQ-4bit")
pytestmark = pytest.mark.skipif(
    not GEMMA4_LOCAL.exists(),
    reason="local Gemma4 model is not available",
)


def test_local_gemma4_auto_tokenizer_loads():
    tokenizer = AutoTokenizer.from_pretrained(
        str(GEMMA4_LOCAL),
        trust_remote_code=True,
    )
    assert tokenizer.__class__.__name__.startswith("Gemma")
    assert "<|video|>" in tokenizer.all_special_tokens


def test_local_gemma4_cached_tokenizer_loads():
    tokenizer = CachedHfTokenizer.from_pretrained(
        str(GEMMA4_LOCAL),
        trust_remote_code=True,
    )
    token_ids = tokenizer.encode("hello from gemma4")
    assert isinstance(token_ids, list)
    assert len(token_ids) > 0
