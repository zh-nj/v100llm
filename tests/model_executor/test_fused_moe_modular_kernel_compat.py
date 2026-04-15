# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def test_modular_kernel_exports_legacy_aliases() -> None:
    from vllm.model_executor.layers.fused_moe import modular_kernel as mk

    assert mk.FusedMoEPermuteExpertsUnpermute is mk.FusedMoEExpertsModular
    assert mk.FusedMoEModularKernel is mk.FusedMoEKernel
