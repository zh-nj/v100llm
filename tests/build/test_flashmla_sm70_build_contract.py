# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def test_setup_allows_flashmla_sm70_on_cuda_128_when_opted_in() -> None:
    text = _read("setup.py")

    assert "FLASH_MLA_ENABLE_SM70" in text
    assert 'Version("12.8")' in text
    assert "_flashmla_sm70_enabled()" in text
    assert 'nvcc_cuda_version >= Version("12.9")' in text


def test_flashmla_cmake_declares_sm70_opt_in_arch_and_sources() -> None:
    text = _read("cmake/external_projects/flashmla.cmake")

    assert "FLASH_MLA_ENABLE_SM70" in text
    assert 'list(APPEND SUPPORT_ARCHS "7.0")' in text
    assert "csrc/sm70/decode/sparse_fp8/instantiations/v32_fp8.cu" in text
    assert "csrc/sm70/decode/sparse_fp8/instantiations/model1_fp8.cu" in text
    assert "csrc/sm70/prefill/sparse/instantiations/bf16.cu" in text


def test_flashmla_cmake_skips_dense_extension_for_sm70_only_build() -> None:
    text = _read("cmake/external_projects/flashmla.cmake")

    assert "FLASHMLA_BUILD_DENSE_EXTENSION" in text
    assert "add_custom_target(_flashmla_extension_C)" in text


def test_flashmla_cmake_mirrors_sm70_feature_and_tuning_defines() -> None:
    text = _read("cmake/external_projects/flashmla.cmake")

    assert "KERUTILS_ALLOW_SM70_STUB_COMPILE" in text
    assert "FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE" in text
    assert "FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE" in text
