# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from typing import Any

import numpy as np
import pytest
import requests
from PIL import Image

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.pooling.embed.protocol import EmbeddingResponse
from vllm.multimodal.utils import (
    encode_image_url,
    encode_video_url,
)

MODEL_NAME = os.getenv(
    "TEST_QWEN3_VL_EMBED_MODEL",
    "Qwen/Qwen3-VL-Embedding-2B",
)
GPU_MEMORY_UTILIZATION = os.getenv(
    "TEST_QWEN3_VL_EMBED_GPU_MEMORY_UTILIZATION",
    "0.35",
)
MAX_MODEL_LEN = os.getenv(
    "TEST_QWEN3_VL_EMBED_MAX_MODEL_LEN",
    "1536",
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


def _make_messages(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Represent the user's input."}],
        },
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": ""}]},
    ]


def _post_embedding(
    server: RemoteOpenAIServer,
    content: list[dict[str, Any]],
    **extra_body: Any,
) -> EmbeddingResponse:
    response = requests.post(
        server.url_for("v1/embeddings"),
        json={
            "model": MODEL_NAME,
            "messages": _make_messages(content),
            "encoding_format": "float",
            "continue_final_message": True,
            "add_special_tokens": True,
            **extra_body,
        },
        timeout=300,
    )
    response.raise_for_status()
    return EmbeddingResponse.model_validate(response.json())


def _cosine(lhs: list[float], rhs: list[float]) -> float:
    lhs_arr = np.asarray(lhs, dtype=np.float32)
    rhs_arr = np.asarray(rhs, dtype=np.float32)
    return float(
        np.dot(lhs_arr, rhs_arr)
        / (np.linalg.norm(lhs_arr) * np.linalg.norm(rhs_arr))
    )


def _build_content(
    kind: str,
    image_data_uri: str,
    video_data_uri: str,
) -> list[dict[str, Any]]:
    mapping = {
        "text": [{"type": "text", "text": TEXT}],
        "image": [{"type": "image_url", "image_url": {"url": image_data_uri}}],
        "video": [{"type": "video_url", "video_url": {"url": video_data_uri}}],
        "image_text": [
            {"type": "image_url", "image_url": {"url": image_data_uri}},
            {"type": "text", "text": TEXT},
        ],
        "video_text": [
            {"type": "video_url", "video_url": {"url": video_data_uri}},
            {"type": "text", "text": VIDEO_TEXT},
        ],
    }
    return mapping[kind]


def _make_image_data_uri() -> str:
    image = Image.new("RGB", (224, 224), color=(220, 30, 30))
    return encode_image_url(image)


def _make_video_data_uri() -> str:
    frames = np.zeros((8, 96, 96, 3), dtype=np.uint8)
    for idx in range(frames.shape[0]):
        frames[idx, :, :, 2] = min(255, 80 + idx * 20)
        row_start = idx * 8
        row_end = min(frames.shape[1], row_start + 16)
        frames[idx, row_start:row_end, :, 1] = 160
    return encode_video_url(frames)


@pytest.fixture(scope="module")
def server():
    args = [
        "--runner",
        "pooling",
        "--dtype",
        "half",
        "--gpu-memory-utilization",
        GPU_MEMORY_UTILIZATION,
        "--max-model-len",
        MAX_MODEL_LEN,
        "--max-num-seqs",
        "1",
        "--enforce-eager",
        "--limit-mm-per-prompt",
        json.dumps({"image": 2, "video": 1}),
        "--media-io-kwargs",
        json.dumps({"video": {"num_frames": VIDEO_NUM_FRAMES}}),
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def image_data_uri() -> str:
    return _make_image_data_uri()


@pytest.fixture(scope="module")
def video_data_uri() -> str:
    return _make_video_data_uri()


@pytest.mark.parametrize(
    "kind",
    ["text", "image", "video", "image_text", "video_text"],
)
def test_qwen3_vl_chat_embeddings_modalities(
    server: RemoteOpenAIServer,
    image_data_uri: str,
    video_data_uri: str,
    kind: str,
) -> None:
    output = _post_embedding(
        server,
        _build_content(kind, image_data_uri, video_data_uri),
    )

    assert len(output.data) == 1
    assert len(output.data[0].embedding) == EMBED_DIM
    assert output.usage.prompt_tokens > 0
    assert output.usage.completion_tokens == 0

    embedding = np.asarray(output.data[0].embedding, dtype=np.float32)
    assert np.isfinite(embedding).all()
    assert np.linalg.norm(embedding) > 0.0


def test_qwen3_vl_chat_embeddings_video_media_io_override(
    server: RemoteOpenAIServer,
    video_data_uri: str,
) -> None:
    default_output = _post_embedding(
        server,
        _build_content("video_text", "", video_data_uri),
    )
    override_output = _post_embedding(
        server,
        _build_content("video_text", "", video_data_uri),
        media_io_kwargs={"video": {"num_frames": 4}},
    )

    assert override_output.usage.prompt_tokens < default_output.usage.prompt_tokens


def test_qwen3_vl_chat_embeddings_semantic_sanity(
    server: RemoteOpenAIServer,
    image_data_uri: str,
    video_data_uri: str,
) -> None:
    text_only = _post_embedding(server, [{"type": "text", "text": TEXT}])
    image_match = _post_embedding(
        server,
        [
            {"type": "image_url", "image_url": {"url": image_data_uri}},
            {"type": "text", "text": TEXT},
        ],
    )
    image_mismatch = _post_embedding(
        server,
        [
            {"type": "image_url", "image_url": {"url": image_data_uri}},
            {"type": "text", "text": MISMATCH_TEXT},
        ],
    )
    video_query = _post_embedding(server, [{"type": "text", "text": VIDEO_TEXT}])
    video_match = _post_embedding(
        server,
        [
            {"type": "video_url", "video_url": {"url": video_data_uri}},
            {"type": "text", "text": VIDEO_TEXT},
        ],
    )
    video_mismatch = _post_embedding(
        server,
        [
            {"type": "video_url", "video_url": {"url": video_data_uri}},
            {"type": "text", "text": "Cars driving on a highway."},
        ],
    )

    assert _cosine(
        image_match.data[0].embedding,
        text_only.data[0].embedding,
    ) > _cosine(image_mismatch.data[0].embedding, text_only.data[0].embedding)
    assert _cosine(
        video_match.data[0].embedding,
        video_query.data[0].embedding,
    ) > _cosine(video_mismatch.data[0].embedding, video_query.data[0].embedding)
