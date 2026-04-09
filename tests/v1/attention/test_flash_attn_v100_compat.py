import torch

from vllm.vllm_flash_attn import flash_attn_interface as fai


def test_flash_attn_func_wraps_varlen_inputs(monkeypatch):
    q = torch.randn(2, 3, 4, 8)
    k = torch.randn(2, 3, 2, 8)
    v = torch.randn(2, 3, 2, 8)
    seen = {}

    def fake_varlen(*, q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k,
                    cu_seqlens_k=None, **kwargs):
        seen["q_shape"] = tuple(q.shape)
        seen["k_shape"] = tuple(k.shape)
        seen["v_shape"] = tuple(v.shape)
        seen["max_seqlen_q"] = max_seqlen_q
        seen["max_seqlen_k"] = max_seqlen_k
        seen["cu_seqlens_q"] = cu_seqlens_q.clone()
        seen["cu_seqlens_k"] = cu_seqlens_k.clone()
        seen["kwargs"] = kwargs
        return q + 1

    monkeypatch.setattr(fai, "_get_fa2_varlen_func", lambda: fake_varlen)

    out = fai.flash_attn_func(q, k, v, causal=True, softmax_scale=0.25)

    assert seen["q_shape"] == (6, 4, 8)
    assert seen["k_shape"] == (6, 2, 8)
    assert seen["v_shape"] == (6, 2, 8)
    assert seen["max_seqlen_q"] == 3
    assert seen["max_seqlen_k"] == 3
    assert torch.equal(seen["cu_seqlens_q"], torch.tensor([0, 3, 6], dtype=torch.int32))
    assert torch.equal(seen["cu_seqlens_k"], torch.tensor([0, 3, 6], dtype=torch.int32))
    assert seen["kwargs"]["causal"] is True
    assert seen["kwargs"]["softmax_scale"] == 0.25
    assert tuple(out.shape) == tuple(q.shape)
    assert torch.equal(out, q + 1)


def test_flash_attn_decode_paged_calls_fwd_kvcache(monkeypatch):
    query = torch.randn(3, 4, 8)
    key_cache = torch.randn(7, 16, 2, 8)
    value_cache = torch.randn(7, 16, 2, 8)
    block_table = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32)
    seq_lens = torch.tensor([15, 11, 9], dtype=torch.int32)
    out = torch.empty_like(query)
    seen = {}

    def fake_kvcache(q, kcache, vcache, k, v, seqlens_k, rotary_cos, rotary_sin,
                     cache_batch_idx, leftpad_k, block_table, alibi_slopes, out,
                     softmax_scale, is_causal, window_size_left, window_size_right,
                     softcap, is_rotary_interleaved, num_splits):
        seen["q_shape"] = tuple(q.shape)
        seen["kcache_shape"] = tuple(kcache.shape)
        seen["vcache_shape"] = tuple(vcache.shape)
        seen["seqlens_k"] = seqlens_k.clone()
        seen["block_table"] = block_table.clone()
        seen["out_shape"] = tuple(out.shape)
        seen["softmax_scale"] = softmax_scale
        seen["is_causal"] = is_causal
        seen["window_size_left"] = window_size_left
        seen["window_size_right"] = window_size_right
        seen["softcap"] = softcap
        seen["is_rotary_interleaved"] = is_rotary_interleaved
        seen["num_splits"] = num_splits
        filled = torch.full_like(out, 42)
        return [filled, torch.zeros(1, dtype=out.dtype)]

    monkeypatch.setattr(fai, "_get_fa2_kvcache_op", lambda: fake_kvcache)

    result = fai.flash_attn_decode_paged(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        softmax_scale=0.5,
        out=out,
    )

    assert seen["q_shape"] == (3, 1, 4, 8)
    assert seen["kcache_shape"] == (7, 16, 2, 8)
    assert seen["vcache_shape"] == (7, 16, 2, 8)
    assert torch.equal(seen["seqlens_k"], seq_lens)
    assert torch.equal(seen["block_table"], block_table)
    assert seen["out_shape"] == (3, 1, 4, 8)
    assert seen["softmax_scale"] == 0.5
    assert seen["is_causal"] is True
    assert seen["window_size_left"] == -1
    assert seen["window_size_right"] == -1
    assert seen["softcap"] == 0.0
    assert seen["is_rotary_interleaved"] is True
    assert seen["num_splits"] == 0
    assert tuple(result.shape) == tuple(query.shape)
    assert torch.equal(result, torch.full_like(query, 42))
