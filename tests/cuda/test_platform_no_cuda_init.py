# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test that platform imports do not prematurely initialize CUDA.

This is critical for Ray-based multi-GPU setups where workers need to
set CUDA_VISIBLE_DEVICES after importing vLLM but before CUDA is initialized.
If CUDA is initialized during import, device_count() gets locked and ignores
subsequent env var changes.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import vllm.platforms as platforms_module
import vllm.utils.import_utils as import_utils

SCRIPTS_DIR = Path(__file__).parent / "scripts"


def run_script(script_name: str) -> subprocess.CompletedProcess:
    """Run a test script in a subprocess with clean CUDA state."""
    script_path = SCRIPTS_DIR / script_name
    return subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
    )


def test_platform_import_does_not_init_cuda():
    """Test that importing vllm.platforms does not initialize CUDA."""
    result = run_script("check_platform_no_cuda_init.py")
    if result.returncode != 0:
        pytest.fail(f"Platform import initialized CUDA:\n{result.stderr}")


def test_device_count_respects_env_after_platform_import():
    """Test that device_count respects CUDA_VISIBLE_DEVICES after import."""
    result = run_script("check_device_count_respects_env.py")
    if result.returncode != 0:
        pytest.fail(
            f"device_count does not respect env var after import:\n{result.stderr}"
        )


def test_disable_nvml_plugin_uses_non_nvml_fallback(monkeypatch):
    monkeypatch.setenv("VLLM_DISABLE_NVML", "1")
    monkeypatch.setattr(platforms_module, "vllm_version_matches_substr", lambda _: False)
    monkeypatch.setattr(
        platforms_module,
        "_non_nvml_cuda_platform_available",
        lambda: True,
    )

    assert platforms_module.cuda_platform_plugin() == "vllm.platforms.cuda.CudaPlatform"


def test_cuda_plugin_falls_back_to_device_files_on_nvml_error(monkeypatch):
    class MockNVMLError_Unknown(Exception):
        pass

    def raise_nvml_error():
        raise MockNVMLError_Unknown("Unknown Error")

    monkeypatch.delenv("VLLM_DISABLE_NVML", raising=False)
    monkeypatch.setattr(platforms_module, "vllm_version_matches_substr", lambda _: False)
    monkeypatch.setattr(import_utils, "import_pynvml", raise_nvml_error)
    monkeypatch.setattr(platforms_module, "_has_nvidia_device_files", lambda: True)

    assert platforms_module.cuda_platform_plugin() == "vllm.platforms.cuda.CudaPlatform"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA host")
def test_disable_nvml_still_detects_cuda_platform():
    result = run_script("check_disable_nvml_cuda_platform.py")
    if result.returncode != 0:
        pytest.fail(f"disable-nvml CUDA detection failed:\n{result.stderr}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA host")
def test_cuda_platform_uses_custom_op_collectives():
    from vllm.platforms import current_platform

    assert current_platform.use_custom_op_collectives()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
