import pytest
import torch

import flash_mla
import kernelkit as kk
import lib
from lib import RawTestParamForDecode
import ref


def _require_sm70_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the SM70 sparse decode runtime test")
    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 sparse decode runtime test requires a Volta/V100 GPU")


def _make_two_decode_rows_share_valid_prefix(scope) -> None:
    assert scope.topk_length is not None
    valid_len = int(min(scope.topk_length[0].item(), scope.topk_length[1].item()))
    scope.cache_seqlens[1].copy_(scope.cache_seqlens[0])
    scope.block_table[1].copy_(scope.block_table[0])
    scope.abs_indices[1].copy_(scope.abs_indices[0])
    scope.indices_in_kvcache[1].copy_(scope.indices_in_kvcache[0])
    scope.topk_length[0] = valid_len
    scope.topk_length[1] = valid_len


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_sm70_sparse_decode_batch2_verify_matching_topk_rows_correctness():
    _require_sm70_cuda()
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    kk.set_random_seed(20260516)

    p = RawTestParamForDecode(
        b=2,
        h_q=64,
        s_q=1,
        h_kv=1,
        s_kv=2048,
        is_varlen=False,
        topk=128,
        have_topk_length=True,
        enable_attn_sink=True,
        extra_s_k=2048,
        extra_topk=512,
        block_size=256,
        extra_block_size=64,
        have_extra_topk_length=True,
        d_qk=512,
        check_correctness=True,
        num_runs=0,
        seed=20260516,
    ).to_test_param()
    t = lib.generate_testcase_for_decode(p)
    _make_two_decode_rows_share_valid_prefix(t.kv_scope)
    assert t.extra_kv_scope is not None
    _make_two_decode_rows_share_valid_prefix(t.extra_kv_scope)

    tile_scheduler_metadata, _ = flash_mla.get_mla_metadata()
    out_ans, lse_ans = lib.run_flash_mla_decode(p, t, tile_scheduler_metadata, None)
    out_ref, lse_ref = ref.ref_sparse_attn_decode(p, t)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        out_ans.float(),
        out_ref.float(),
        atol=1e-3,
        rtol=2.01 / 128,
    )
    torch.testing.assert_close(
        lse_ans.float(),
        lse_ref.float(),
        atol=1e-6,
        rtol=8.01 / 65536,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_sm70_sparse_decode_batch2_verify_mismatched_topk_rows_falls_back():
    _require_sm70_cuda()
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    kk.set_random_seed(20260517)

    p = RawTestParamForDecode(
        b=2,
        h_q=64,
        s_q=1,
        h_kv=1,
        s_kv=2048,
        is_varlen=False,
        topk=128,
        have_topk_length=True,
        enable_attn_sink=True,
        extra_s_k=2048,
        extra_topk=512,
        block_size=256,
        extra_block_size=64,
        have_extra_topk_length=True,
        d_qk=512,
        check_correctness=True,
        num_runs=0,
        seed=20260517,
    ).to_test_param()
    t = lib.generate_testcase_for_decode(p)
    assert t.extra_kv_scope is not None

    tile_scheduler_metadata, _ = flash_mla.get_mla_metadata()
    out_ans, lse_ans = lib.run_flash_mla_decode(p, t, tile_scheduler_metadata, None)
    out_ref, lse_ref = ref.ref_sparse_attn_decode(p, t)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        out_ans.float(),
        out_ref.float(),
        atol=1e-3,
        rtol=2.01 / 128,
    )
    torch.testing.assert_close(
        lse_ans.float(),
        lse_ref.float(),
        atol=1e-6,
        rtol=8.01 / 65536,
    )
