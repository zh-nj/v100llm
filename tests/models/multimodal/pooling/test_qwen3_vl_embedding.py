# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from ....conftest import VllmRunner

MODEL_NAME = os.getenv(
    "TEST_QWEN3_VL_EMBED_MODEL",
    "Qwen/Qwen3-VL-Embedding-2B",
)
GPU_MEMORY_UTILIZATION = float(
    os.getenv(
        "TEST_QWEN3_VL_EMBED_GPU_MEMORY_UTILIZATION",
        "0.35",
    )
)
MAX_MODEL_LEN = int(
    os.getenv(
        "TEST_QWEN3_VL_EMBED_MAX_MODEL_LEN",
        "640",
    )
)
VIDEO_NUM_FRAMES = int(
    os.getenv(
        "TEST_QWEN3_VL_EMBED_VIDEO_NUM_FRAMES",
        "8",
    )
)
EMBED_DIM = 4096
TEXT = "A solid red square."
VIDEO_TEXT = "A short blue animation."
MISMATCH_TEXT = "A tax form on an office desk."
SYSTEM_TEXT = "Represent the user's input."
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"


def _make_image() -> Image.Image:
    return Image.new("RGB", (224, 224), color=(220, 30, 30))


def _make_video() -> tuple[np.ndarray, dict[str, Any]]:
    frames = np.zeros((VIDEO_NUM_FRAMES, 96, 96, 3), dtype=np.uint8)
    for idx in range(frames.shape[0]):
        frames[idx, :, :, 2] = min(255, 80 + idx * 20)
        row_start = idx * 8
        row_end = min(frames.shape[1], row_start + 16)
        frames[idx, row_start:row_end, :, 1] = 160
    return (
        frames,
        {
            "total_num_frames": int(frames.shape[0]),
            "fps": 2.0,
            "duration": float(frames.shape[0]) / 2.0,
            "video_backend": "opencv",
            "frames_indices": list(range(int(frames.shape[0]))),
            "do_sample_frames": True,
        },
    )


def _make_prompt(user_content: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_TEXT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _cosine(lhs: list[float], rhs: list[float]) -> float:
    lhs_arr = np.asarray(lhs, dtype=np.float32)
    rhs_arr = np.asarray(rhs, dtype=np.float32)
    return float(
        np.dot(lhs_arr, rhs_arr)
        / (np.linalg.norm(lhs_arr) * np.linalg.norm(rhs_arr))
    )


def _embed(
    vllm_model: VllmRunner,
    kind: str,
    image: Image.Image,
    video: np.ndarray,
) -> list[float]:
    if kind == "text":
        return vllm_model.embed([_make_prompt(TEXT)])[0]
    if kind == "image":
        return vllm_model.embed(
            [_make_prompt(IMAGE_PLACEHOLDER)],
            images=[image],
        )[0]
    if kind == "video":
        return vllm_model.embed(
            [_make_prompt(VIDEO_PLACEHOLDER)],
            videos=[video],
        )[0]
    if kind == "image_text":
        return vllm_model.embed(
            [_make_prompt(f"{IMAGE_PLACEHOLDER}{TEXT}")],
            images=[image],
        )[0]
    if kind == "video_text":
        return vllm_model.embed(
            [_make_prompt(f"{VIDEO_PLACEHOLDER}{VIDEO_TEXT}")],
            videos=[video],
        )[0]
    raise ValueError(f"Unknown kind: {kind}")


@pytest.fixture(scope="module")
def image() -> Image.Image:
    return _make_image()


@pytest.fixture(scope="module")
def video() -> tuple[np.ndarray, dict[str, Any]]:
    return _make_video()


@pytest.fixture(scope="module")
def qwen3_vl_embed_runner(vllm_runner):
    with vllm_runner(
        MODEL_NAME,
        runner="pooling",
        dtype="half",
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        limit_mm_per_prompt={"image": 2, "video": 1},
        media_io_kwargs={"video": {"num_frames": VIDEO_NUM_FRAMES}},
    ) as vllm_model:
        yield vllm_model


@pytest.mark.parametrize(
    "kind",
    ["text", "image", "video", "image_text", "video_text"],
)
def test_qwen3_vl_embedding_modalities(
    qwen3_vl_embed_runner: VllmRunner,
    image: Image.Image,
    video: np.ndarray,
    kind: str,
) -> None:
    embedding = _embed(qwen3_vl_embed_runner, kind, image, video)

    assert len(embedding) == EMBED_DIM
    embedding_arr = np.asarray(embedding, dtype=np.float32)
    assert np.isfinite(embedding_arr).all()
    assert np.linalg.norm(embedding_arr) > 0.0


def test_qwen3_vl_embedding_semantic_sanity(
    qwen3_vl_embed_runner: VllmRunner,
    image: Image.Image,
    video: np.ndarray,
) -> None:
    text_only = _embed(qwen3_vl_embed_runner, "text", image, video)
    image_match = _embed(qwen3_vl_embed_runner, "image_text", image, video)
    image_mismatch = qwen3_vl_embed_runner.embed(
        [_make_prompt(f"{IMAGE_PLACEHOLDER}{MISMATCH_TEXT}")],
        images=[image],
    )[0]
    video_query = qwen3_vl_embed_runner.embed([_make_prompt(VIDEO_TEXT)])[0]
    video_match = _embed(qwen3_vl_embed_runner, "video_text", image, video)
    video_mismatch = qwen3_vl_embed_runner.embed(
        [_make_prompt(f"{VIDEO_PLACEHOLDER}Cars driving on a highway.")],
        videos=[video],
    )[0]

    assert _cosine(image_match, text_only) > _cosine(image_mismatch, text_only)
    assert _cosine(video_match, video_query) > _cosine(video_mismatch, video_query)
    assert not torch.allclose(
        torch.tensor(text_only),
        torch.tensor(image_match),
    )
