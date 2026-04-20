# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def test_common_requirements_allow_transformers_55x():
    text = _read("requirements/common.txt")
    assert (
        "transformers >= 4.56.0, != 5.0.*, != 5.1.*, != 5.2.*, "
        "!= 5.3.*, != 5.4.*, != 5.5.0"
    ) in text
    assert "compressed-tensors == 0.15.0.1" in text


def test_test_requirements_pin_transformers_554():
    assert "transformers==5.5.4" in _read("requirements/test.in")
    test_txt = _read("requirements/test.txt")
    assert "transformers==5.5.4" in test_txt
    assert "tokenizers==0.22.2" in test_txt
