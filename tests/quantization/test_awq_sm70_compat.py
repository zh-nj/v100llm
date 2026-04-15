import importlib

import torch


def test_awq_sm70_exports_default_max_tokens_compat_alias() -> None:
    module = importlib.import_module(
        "vllm.model_executor.layers.quantization.awq_sm70_moe"
    )

    assert hasattr(module, "_DEFAULT_MAX_TOKENS")
    assert module._DEFAULT_MAX_TOKENS == module._DEFAULT_PERSISTENT_MAX_TOKENS


def test_ct_sm70_post_load_delegates_to_awq_method(monkeypatch) -> None:
    ct_module = importlib.import_module(
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe"
    )
    awq_module = importlib.import_module(
        "vllm.model_executor.layers.quantization.awq_sm70_moe"
    )
    ops_module = importlib.import_module("vllm._custom_ops")

    called: dict[str, bool] = {}

    def fake_awq_process_weights_after_loading(self, layer) -> None:
        called["delegated"] = True

    def fake_awq_sm70_prepare(qweight, scales, qzeros, group_size, **kwargs):
        meta = torch.tensor([1, 1], dtype=torch.int32)
        return qweight.contiguous(), scales.contiguous(), meta

    def fake_awq_moe_build_strided_ptrs(weight, scales, k_ld, q_ld, num_experts):
        ptrs = torch.zeros(num_experts, 2, dtype=torch.uint8, device=weight.device)
        return ptrs, ptrs.clone()

    monkeypatch.setattr(
        awq_module.AWQSM70MoEMethod,
        "process_weights_after_loading",
        fake_awq_process_weights_after_loading,
    )
    monkeypatch.setattr(ops_module, "awq_sm70_prepare", fake_awq_sm70_prepare)
    monkeypatch.setattr(
        ops_module,
        "awq_moe_build_strided_ptrs",
        fake_awq_moe_build_strided_ptrs,
    )

    method = ct_module.CompressedTensorsSM70WNA16MoEMethod(
        weight_quant=type("Q", (), {"num_bits": 4, "group_size": 32})(),
        input_quant=None,
        moe=type("M", (), {"experts_per_token": 8})(),
    )
    layer = torch.nn.Module()
    layer.register_parameter(
        "w13_weight_packed",
        torch.nn.Parameter(torch.zeros(1, 4, 64, dtype=torch.int32), requires_grad=False),
    )
    layer.register_parameter(
        "w13_weight_scale",
        torch.nn.Parameter(torch.ones(1, 1, 64, dtype=torch.float16), requires_grad=False),
    )
    layer.register_parameter(
        "w2_weight_packed",
        torch.nn.Parameter(torch.zeros(1, 4, 32, dtype=torch.int32), requires_grad=False),
    )
    layer.register_parameter(
        "w2_weight_scale",
        torch.nn.Parameter(torch.ones(1, 1, 32, dtype=torch.float16), requires_grad=False),
    )

    method.process_weights_after_loading(layer)

    assert called == {"delegated": True}
