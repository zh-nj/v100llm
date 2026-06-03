# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 (V100) 跨卡 MoE dispatch/combine 接口占位（Phase 2，非阻塞增量）。

本模块属于 ``deepgemm-megamoe-sm70-port`` 规划中的 **Phase 2** 交付物，对应
需求 R4「dispatch/combine 通信与计算重叠」。**Phase 1 不实现任何通信或重叠逻辑**，
此处仅落接口占位，目的是：

1. 固定 Phase 2 的公共接口形状（``dispatch`` / ``combine`` / overlap 编排），
   让单卡融合路径（Phase 1）与未来跨卡路径有清晰的对接边界。
2. 保证在 Phase 1 误调用跨卡路径时，**立即抛出带明确诊断信息的异常，
   而不是静默挂起**（R4.4）。

Phase 2 的实现约束（设计 §Components and Interfaces 6）：

* 先交付**功能正确的非重叠路径**（dispatch → 本地计算 → combine），
  再叠加 CUDA stream / event（计算流 + 通信流 + event）做通信/计算重叠（R4.1/R4.2）。
* V100 平台**不依赖** PyTorch ``symm_mem``（对称内存）或 cluster 原语——这些在
  SM70 上不可用；跨卡通信将基于 NCCL all-to-all 或手写 P2P 实现。
* 任一通信路径出错或超时时，**给出明确诊断信息而非静默挂起**（R4.4）。

注意：本模块刻意**不导入** ``symm_mem`` / cluster 相关依赖，以确保 Phase 1
环境（V100）下导入本模块不会引入不可用的现代原语依赖。
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Phase 2 占位统一诊断前缀：误用时给出可定位、非静默的错误信息（R4.4）。
_PHASE2_NOT_IMPLEMENTED = (
    "SM70MoEDispatchCombine.{method}() 是 Phase 2（跨卡 EP dispatch/combine 通信"
    "与计算重叠，需求 R4）的交付物，当前 deepgemm-megamoe-sm70-port 的 Phase 1 "
    "仅交付单卡算子融合（linear1 → SwiGLU → linear2），尚未实现跨卡通信路径。\n"
    "请确认：(1) 当前未在 SM70 上启用专家并行（EP）的融合通信路径；"
    "(2) 单卡融合路径不应调用 dispatch/combine。\n"
    "此处显式抛出 NotImplementedError 以避免静默挂起（R4.4）；"
    "Phase 2 将基于 NCCL all-to-all / 手写 P2P（不依赖 symm_mem / cluster）"
    "先交付功能正确的非重叠路径，再叠加 stream/event 重叠。"
)


class SM70MoEDispatchCombine:
    """SM70 跨卡 MoE dispatch/combine 接口占位（Phase 2，仅接口）。

    本类定义 Phase 2 跨卡专家并行（EP）通信路径的公共接口，但在 Phase 1 内
    **不提供任何实现**。所有方法均抛出带明确诊断信息的 :class:`NotImplementedError`
    （满足 R4.4「明确诊断信息而非静默挂起」原则）。

    设计意图（设计 §Components and Interfaces 6）::

        class SM70MoEDispatchCombine:           # Phase 2，仅接口
            def dispatch(self, x, topk_ids, ep_group): ...   # NCCL all-to-all
            def combine(self, y, ep_group): ...
            # overlap: 计算流 + 通信流 + event；不依赖 symm_mem/cluster

    Phase 2 实现时：先交付功能正确的非重叠路径，再做 stream/event 重叠；
    通信基于 NCCL / 手写 P2P，**不依赖** ``symm_mem`` 或 cluster 原语。
    """

    def dispatch(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        ep_group: object,
    ) -> torch.Tensor:
        """[Phase 2 占位] 将 token 发送到其目标专家所在的 rank（NCCL all-to-all）。

        Args:
            x: 待分发的输入激活（fp16）。
            topk_ids: 每个 token 选中的专家索引，决定其目标 rank。
            ep_group: 专家并行通信组。

        Raises:
            NotImplementedError: Phase 1 始终抛出，附带明确诊断信息（R4.4）。
        """
        raise NotImplementedError(_PHASE2_NOT_IMPLEMENTED.format(method="dispatch"))

    def combine(
        self,
        y: torch.Tensor,
        ep_group: object,
    ) -> torch.Tensor:
        """[Phase 2 占位] 从各专家所在 rank 回收本地专家计算结果（NCCL all-to-all）。

        Args:
            y: 本地专家输出，待回收/规约。
            ep_group: 专家并行通信组。

        Raises:
            NotImplementedError: Phase 1 始终抛出，附带明确诊断信息（R4.4）。
        """
        raise NotImplementedError(_PHASE2_NOT_IMPLEMENTED.format(method="combine"))

    def dispatch_compute_combine_overlap(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        ep_group: object,
    ) -> torch.Tensor:
        """[Phase 2 占位] dispatch → 本地计算 → combine 的通信/计算重叠编排。

        Phase 2 将以 CUDA 计算流 + 通信流 + event 实现重叠（R4.2），
        在已有功能正确的非重叠路径前提下叠加；**不依赖** ``symm_mem`` / cluster。

        Args:
            x: 输入激活（fp16）。
            topk_ids: 每个 token 选中的专家索引。
            ep_group: 专家并行通信组。

        Raises:
            NotImplementedError: Phase 1 始终抛出，附带明确诊断信息（R4.4）。
        """
        raise NotImplementedError(
            _PHASE2_NOT_IMPLEMENTED.format(method="dispatch_compute_combine_overlap")
        )
