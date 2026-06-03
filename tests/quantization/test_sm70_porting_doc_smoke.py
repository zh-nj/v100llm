# SPDX-License-Identifier: Apache-2.0
"""SMOKE test: R1 可行性/移植文档章节完整性校验.

Feature: deepgemm-megamoe-sm70-port

R1 ("可行性分析与移植路径文档") 是一项**文档性交付**, 其载体即本 spec 的
``design.md`` —— 具体为 "可行性分析与移植路径 (R1)" 一节及 Overview 中的
"收益的诚实框定 (R1.6)" 与 MXFP4 软件反量化前提 (R1.7)。本测试以 SMOKE 粒度
校验该文档章节与关键内容标记齐全, 覆盖验收准则 R1.1 ~ R1.7。

本 spec 已重定范围为 **dsv4f MXFP4** 优化: 集成入口为
``Mxfp4SM70MoEMethod`` (而非旧 ``AWQSM70MoEMethod``), 回退基线为其既有
TurboMind grouped GEMM 逐算子路径; 专家权重为 MXFP4 (E2M1 + per-32 block
scale), V100 上以软件反量化到 fp16 为前提 (FP8 仅占位)。

由于 R1 为文档交付 (见 design.md §Requirements Mapping: "SMOKE（章节完整性）"),
本测试只读取 ``design.md`` 文本并断言章节/关键词存在, 不依赖 torch 或任何运行时。

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Spec 目录 (相对工作区根) 内的设计文档即 R1 交付物。
_SPEC_REL = Path(".kiro/specs/deepgemm-megamoe-sm70-port/design.md")


def _locate_design_doc() -> Path:
    """Robustly locate ``design.md`` by walking up from this test file.

    测试文件位于 ``tests/quantization/`` 下, 设计文档位于仓库根的
    ``.kiro/specs/deepgemm-megamoe-sm70-port/design.md``。我们从本文件出发逐级
    向上查找, 以避免对当前工作目录 (cwd / worktree) 的脆弱依赖。
    """
    here = Path(__file__).resolve()
    # 从测试文件所在目录向上, 直到文件系统根。
    for parent in [here.parent, *here.parents]:
        candidate = parent / _SPEC_REL
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"无法定位 R1 设计文档 {_SPEC_REL} (从 {here} 向上查找均未命中)"
    )


@pytest.fixture(scope="module")
def design_doc_text() -> str:
    path = _locate_design_doc()
    text = path.read_text(encoding="utf-8")
    assert text.strip(), f"设计文档为空: {path}"
    return text


# --- 顶层章节: R1 即设计文档中的 "可行性分析与移植路径 (R1)" 一节 ----------


def test_r1_section_present(design_doc_text: str) -> None:
    """R1 交付物章节 (含 6 个子条款标题) 必须存在。"""
    assert "## 可行性分析与移植路径 (R1)" in design_doc_text
    for subheading in (
        "### R1.1 硬件原语逐项结论",
        "### R1.2 五子阶段 × SM70 实现策略",
        "### R1.3 单卡融合 vs 跨卡重叠",
        "### R1.4 集成点与回退",
        "### R1.5 不可行/负收益记录原则",
    ):
        assert subheading in design_doc_text, f"缺少子章节: {subheading}"


# --- R1.1 逐项硬件原语结论 -------------------------------------------------

# design.md §R1.1 列出的、必须逐项给出 SM70 结论的硬件原语。
_R1_1_HARDWARE_PRIMITIVES = (
    "tcgen05",
    "TMEM",
    "WGMMA",
    "TMA",
    "cluster",
    "cp.async",
    "FP8",
    "FP4",
    "bfloat16",
    "symm_mem",
)


@pytest.mark.parametrize("primitive", _R1_1_HARDWARE_PRIMITIVES)
def test_r1_1_hardware_primitive_listed(design_doc_text: str, primitive: str) -> None:
    """R1.1: 逐项列出 DeepGEMM 依赖的硬件原语并给出 SM70 结论。"""
    assert primitive in design_doc_text, f"R1.1 缺少硬件原语条目: {primitive}"


def test_r1_1_conclusion_and_alternatives(design_doc_text: str) -> None:
    """R1.1: 每项原语需标注 SM70 状态 (可用/不可用) 及替代方案。"""
    section = _extract_section(design_doc_text, "### R1.1 硬件原语逐项结论")
    # 结论列 (不可用) 与替代方案列均应出现在该表格段内。
    assert "不可用" in section or "无硬件" in section, "R1.1 缺少 SM70 状态结论"
    assert "替代方案" in section, "R1.1 缺少替代方案列"
    assert "mma.sync" in section, "R1.1 应给出第一代 mma.sync 替代结论"


# --- R1.2 五子阶段 × SM70 策略 ---------------------------------------------

_R1_2_SUBSTAGES = ("dispatch", "linear1", "SwiGLU", "linear2", "combine")


@pytest.mark.parametrize("substage", _R1_2_SUBSTAGES)
def test_r1_2_substage_strategy(design_doc_text: str, substage: str) -> None:
    """R1.2: 五个子阶段各自给出 SM70 实现策略。"""
    section = _extract_section(design_doc_text, "### R1.2 五子阶段 × SM70 实现策略")
    assert substage in section, f"R1.2 缺少子阶段策略: {substage}"


def test_r1_2_has_sm70_strategy_column(design_doc_text: str) -> None:
    """R1.2: 表格须包含 "SM70 实现策略" 列及 Phase 分期。"""
    section = _extract_section(design_doc_text, "### R1.2 五子阶段 × SM70 实现策略")
    assert "SM70 实现策略" in section, "R1.2 缺少 SM70 实现策略列"
    assert "Phase" in section, "R1.2 缺少分阶段 (Phase) 标注"


# --- R1.3 单卡 vs 跨卡二分 -------------------------------------------------


def test_r1_3_single_vs_cross_card(design_doc_text: str) -> None:
    """R1.3: 明确区分单卡算子融合与跨卡通信重叠, 并给出分期建议。"""
    section = _extract_section(design_doc_text, "### R1.3 单卡融合 vs 跨卡重叠")
    assert "单卡" in section, "R1.3 缺少单卡融合表述"
    assert "跨卡" in section, "R1.3 缺少跨卡重叠表述"
    assert "Phase 1" in section, "R1.3 缺少 Phase 1 (单卡) 分期"
    assert "Phase 2" in section, "R1.3 缺少 Phase 2 (跨卡) 分期"


# --- R1.4 集成点与回退 -----------------------------------------------------


def test_r1_4_integration_and_fallback(design_doc_text: str) -> None:
    """R1.4: 给出与既有 dsv4f MXFP4 MoE 路径的集成点与回退策略。

    重定范围后集成入口为 ``Mxfp4SM70MoEMethod`` (dsv4f 在 V100 实际命中的 MXFP4
    MoE 方法), 回退基线为其既有 TurboMind grouped GEMM 逐算子路径 (替换旧
    ``AWQSM70MoEMethod`` / ``_apply_batched`` 假设)。
    """
    section = _extract_section(design_doc_text, "### R1.4 集成点与回退")
    assert "Mxfp4SM70MoEMethod" in section, "R1.4 缺少集成入口 Mxfp4SM70MoEMethod"
    assert ".apply" in section, "R1.4 缺少集成入口方法 apply()"
    assert "回退" in section, "R1.4 缺少回退 (fallback) 表述"
    # 回退到既有路径的具体落点: dsv4f 既有 TurboMind grouped GEMM 逐算子路径。
    assert (
        "TurboMind" in section and "grouped GEMM" in section
    ), "R1.4 缺少既有回退路径 (TurboMind grouped GEMM 逐算子)"
    assert "逐算子" in section, "R1.4 缺少 '逐算子' 回退路径表述"


# --- R1.5 不可行/负收益记录原则 --------------------------------------------


def test_r1_5_infeasible_negative_benefit_principle(design_doc_text: str) -> None:
    """R1.5: 不可行或负收益子阶段须显式记录结论与理由, 而非强行实现。"""
    section = _extract_section(design_doc_text, "### R1.5 不可行/负收益记录原则")
    assert "负收益" in section, "R1.5 缺少负收益记录原则"
    assert "记录" in section, "R1.5 缺少 '显式记录结论' 表述"
    assert "而非强行实现" in section, "R1.5 缺少 '而非强行实现' 原则"


# --- R1.6 prefill/decode 收益框定 ------------------------------------------


def test_r1_6_benefit_framing_section_present(design_doc_text: str) -> None:
    """R1.6: 收益的诚实框定章节存在 (位于 Overview)。"""
    assert "收益的诚实框定" in design_doc_text, "缺少 R1.6 收益框定章节"
    assert "R1.6" in design_doc_text, "R1.6 章节未标注需求编号"


def test_r1_6_prefill_decode_benefit_sources(design_doc_text: str) -> None:
    """R1.6: 分别量化 prefill / decode 的收益来源。"""
    section = _extract_section(design_doc_text, "### 收益的诚实框定（R1.6 / R2.6 / R2.7）")
    # prefill 收益: 消除中间激活 HBM 往返。
    assert "Prefill" in section or "prefill" in section, "R1.6 缺少 prefill 收益框定"
    assert "HBM" in section, "R1.6 prefill 应说明消除中间激活 HBM 往返"
    # decode 收益: 减少 kernel launch。
    assert "Decode" in section or "decode" in section, "R1.6 缺少 decode 收益框定"
    assert "kernel launch" in section, "R1.6 decode 应说明减少 kernel launch 次数"


def test_r1_6_honest_throughput_caveat(design_doc_text: str) -> None:
    """R1.6: 诚实标注 "融合省的是开销, 不改变 fp16 GEMM 吞吐上限"。"""
    section = _extract_section(design_doc_text, "### 收益的诚实框定（R1.6 / R2.6 / R2.7）")
    assert "省的是开销" in section, "R1.6 缺少 '融合省的是开销' 的诚实前提"
    assert "吞吐上限" in section, "R1.6 缺少 '不改变 fp16 GEMM 吞吐上限' 的前提"


# --- R1.7 dsv4f MXFP4 + 软件反量化前提 (FP8 占位) ---------------------------


def test_r1_7_mxfp4_software_dequant_premise(design_doc_text: str) -> None:
    """R1.7: dsv4f 专家为 MXFP4 (E2M1 + per-32 block scale), V100 软件反量化到 fp16。

    重定范围后的核心前提: 专家权重格式为 MXFP4, 元素为 E2M1, 每 32 元素一个 block
    scale; V100 无 FP4 硬件, 故在内核内**软件反量化**为 fp16 再做 fp16 GEMM。
    断言整篇 design.md 含上述 MXFP4 / 软件反量化关键陈述。
    """
    assert "MXFP4" in design_doc_text, "R1.7 缺少 MXFP4 专家权重格式陈述"
    assert "E2M1" in design_doc_text, "R1.7 缺少 E2M1 (FP4 尾数) 陈述"
    assert (
        "per-32 block scale" in design_doc_text
    ), "R1.7 缺少 per-32 block scale 陈述"
    assert "软件反量化" in design_doc_text, "R1.7 缺少 '软件反量化' 前提"
    # V100 上以软件反量化到 fp16 为前提。
    assert (
        "软件反量化为 fp16" in design_doc_text
        or "软件反量化**为\nfp16" in design_doc_text
        or "软件反量化**为 fp16" in design_doc_text
    ), "R1.7 缺少 'MXFP4 软件反量化为 fp16' 的明确前提"


def test_r1_7_fp8_placeholder_non_target(design_doc_text: str) -> None:
    """R1.7: FP8 专家权重为非目标, 仅占位。"""
    assert "FP8" in design_doc_text, "R1.7 缺少 FP8 占位说明"
    # FP8 专家权重本期为非目标 / 仅占位。
    assert (
        "占位" in design_doc_text
    ), "R1.7 缺少 FP8 专家权重 '占位' (非目标) 的表述"


# --- helpers ---------------------------------------------------------------


def _extract_section(text: str, heading: str) -> str:
    """Return the slice of ``text`` from ``heading`` up to the next heading.

    "下一个标题" 指同级或更高级别的 Markdown 标题 (``#`` ~ 与 heading 同级)。
    为简化, 我们截取到下一个以相同或更少 ``#`` 数量开头的标题行。
    """
    assert heading in text, f"文档中缺少章节标题: {heading}"
    level = len(heading) - len(heading.lstrip("#"))
    start = text.index(heading)
    rest = text[start + len(heading) :]
    lines = rest.splitlines()
    collected: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#"):
            hashes = len(stripped) - len(stripped.lstrip("#"))
            # 命中同级或更高级别标题 -> 章节结束。
            if 1 <= hashes <= level:
                break
        collected.append(line)
    return "\n".join(collected)
