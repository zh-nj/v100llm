import importlib.util
from pathlib import Path

import pytest
import torch

from vllm.vllm_flash_attn import flash_attn_interface as fai


def test_fa2_varlen_uses_legacy_sm70_abi_without_sinks(monkeypatch):
    q = torch.randn(2, 2, 8, dtype=torch.float16)
    k = torch.randn(4, 1, 8, dtype=torch.float16)
    v = torch.randn(4, 1, 8, dtype=torch.float16)
    cu_seqlens_q = torch.tensor([0, 2], dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, 4], dtype=torch.int32)
    seen = {}

    def fake_varlen_fwd(*args):
        seen["argc"] = len(args)
        assert len(args) == 22
        return torch.empty_like(args[0]), torch.empty(
            args[0].shape[1], args[0].shape[0], dtype=torch.float32
        )

    monkeypatch.setattr(fai, "fa2_varlen_supports_s_aux", lambda: False)
    monkeypatch.setattr(fai.torch.ops._vllm_fa2_C, "varlen_fwd", fake_varlen_fwd)

    fai.flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=2,
        max_seqlen_k=4,
        fa_version=2,
    )

    assert seen["argc"] == 22


def test_flash_attn_prefers_worktree_vendored_extension_when_available():
    vendored_path = Path(fai.__file__).with_name("_vllm_fa2_C.abi3.so")
    if not vendored_path.exists():
        pytest.skip("worktree vendored FA2 extension is not built")

    spec = importlib.util.find_spec("vllm_flash_attn")
    if spec is None:
        assert fai.FA2_EXTENSION_SOURCE == "vendored"
        return

    assert fai.FA2_EXTENSION_SOURCE == "vendored"
