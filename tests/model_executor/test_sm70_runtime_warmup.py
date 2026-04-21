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
