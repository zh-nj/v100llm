# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.warmup.awq_sm70_warmup import sm70_awq_warmup


class _DummyWorker:
    def __init__(self, layer):
        self.device = torch.device("cuda:0")
        self.scheduler_config = type("Scheduler", (), {"max_num_batched_tokens": 8})()
        self.vllm_config = type(
            "Cfg",
            (),
            {
                "compilation_config": type(
                    "CC", (), {"cudagraph_capture_sizes": [1, 2, 4]}
                )()
            },
        )()
        self._model = torch.nn.Module()
        self._model.layer = layer

    def get_model(self):
        return self._model


def test_sm70_awq_warmup_handles_runtime_decode_dense(monkeypatch):
    layer = torch.nn.Module()
    layer._sm70_fp8_runtime_prepared = True
    layer._sm70_fp8_output_size = 320
    layer.weight = torch.zeros(320, 256, dtype=torch.float8_e4m3fn, device="cuda")
    layer.weight_scale_inv = torch.ones(3, 2, dtype=torch.float32, device="cuda")
    layer._sm70_fp8_prepared_weight = layer.weight
    layer._sm70_fp8_prepared_scale = layer.weight_scale_inv
    layer._sm70_fp8_prepared_meta = torch.tensor(
        [320, 256, 320, 256, 128, 2, -1, 128, 128],
        dtype=torch.int64,
        device="cuda",
    )
    layer._sm70_fp8_workspace_meta = torch.tensor(
        [128, 256, 320, 256, 4],
        dtype=torch.int64,
        device="cuda",
    )

    calls = []

    monkeypatch.setattr(
        "torch.cuda.get_device_capability",
        lambda *_args, **_kwargs: (7, 0),
    )
    monkeypatch.setattr(
        "vllm.model_executor.warmup.awq_sm70_warmup.ops.sm70_fp8_runtime_gemm_out",
        lambda out, x, pw, ps, pm, decoded, packed, meta: calls.append(
            (tuple(out.shape), tuple(decoded.shape), tuple(packed.shape))
        ),
        raising=False,
    )
    monkeypatch.setattr("torch.cuda.synchronize", lambda *_args, **_kwargs: None)

    sm70_awq_warmup(_DummyWorker(layer))

    assert calls
    assert calls[0][0][1] == 320
    assert calls[0][1] == (128, 256)
    assert calls[0][2] == (320, 256)


def test_sm70_awq_warmup_handles_direct_fp8_dense(monkeypatch):
    layer = torch.nn.Module()
    layer._sm70_fp8_runtime_prepared = True
    layer._sm70_fp8_direct_prepared = True
    layer._sm70_fp8_output_size = 320
    layer.weight = torch.zeros(320, 256, dtype=torch.float8_e4m3fn, device="cuda")
    layer._sm70_fp8_prepared_weight = layer.weight
    layer._sm70_fp8_prepared_scale = torch.empty(
        (2, 320), dtype=torch.float16, device="cuda"
    )
    layer._sm70_fp8_prepared_meta = torch.tensor(
        [320, 256, 128, 8192, 320],
        dtype=torch.int64,
        device="cuda",
    )

    calls = []

    monkeypatch.setattr(
        "torch.cuda.get_device_capability",
        lambda *_args, **_kwargs: (7, 0),
    )
    monkeypatch.setattr(
        "vllm.model_executor.warmup.awq_sm70_warmup.ops.sm70_fp8_direct_gemm_out",
        lambda out, x, pw, ps, pm: calls.append(
            (tuple(out.shape), tuple(x.shape), tuple(ps.shape))
        ),
        raising=False,
    )
    monkeypatch.setattr("torch.cuda.synchronize", lambda *_args, **_kwargs: None)

    sm70_awq_warmup(_DummyWorker(layer))

    assert calls
    assert calls[0] == ((1, 320), (1, 256), (2, 320))


def test_sm70_awq_warmup_handles_direct_fp8_moe(monkeypatch):
    layer = torch.nn.Module()
    layer._sm70_fp8_moe_direct_prepared = True
    layer.sm70_num_experts = 2
    layer.sm70_w13_k_dim = 256
    layer.sm70_w13_n_dim = 256
    layer.sm70_w2_k_dim = 128
    layer.sm70_w2_n_dim = 256
    layer.sm70_intermediate_size = 128
    layer._buf_top_k = 1
    layer.w13_tm_weight = torch.zeros(
        (2, 256, 256), dtype=torch.float8_e4m3fn, device="cuda"
    )
    layer.w13_tm_scales = torch.empty((2, 2, 256), dtype=torch.float16, device="cuda")
    layer.w2_tm_weight = torch.zeros(
        (2, 256, 128), dtype=torch.float8_e4m3fn, device="cuda"
    )
    layer.w2_tm_scales = torch.empty((2, 1, 256), dtype=torch.float16, device="cuda")
    layer.w13_strided_ptrs_w = torch.empty(32, dtype=torch.uint8, device="cuda")
    layer.w13_strided_ptrs_s = torch.empty(32, dtype=torch.uint8, device="cuda")
    layer.w2_strided_ptrs_w = torch.empty(32, dtype=torch.uint8, device="cuda")
    layer.w2_strided_ptrs_s = torch.empty(32, dtype=torch.uint8, device="cuda")

    calls = []

    monkeypatch.setattr(
        "torch.cuda.get_device_capability",
        lambda *_args, **_kwargs: (7, 0),
    )

    def fake_sm70_fp8_moe_gemm_out(
        out, x, offsets, ptrs_w, ptrs_s, experts, k, n, group, gated=False
    ):
        calls.append((tuple(out.shape), tuple(x.shape), experts, k, n, group, gated))

    monkeypatch.setattr(
        "vllm.model_executor.warmup.awq_sm70_warmup.ops.sm70_fp8_moe_gemm_out",
        fake_sm70_fp8_moe_gemm_out,
        raising=False,
    )
    monkeypatch.setattr("torch.cuda.synchronize", lambda *_args, **_kwargs: None)

    sm70_awq_warmup(_DummyWorker(layer))

    assert calls
    assert calls[0] == ((1, 128), (1, 256), 2, 256, 256, 128, True)
    assert calls[1] == ((1, 256), (1, 128), 2, 128, 256, 128, False)
