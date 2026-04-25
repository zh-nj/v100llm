# SM70 Qwen3-VL-Embedding FP8 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Qwen/Qwen3-VL-Embedding-2B` the upstream regression target and validate `/mnt/data6/models/Qwen3-VL-Embedding-8B-FP8` on single-card SM70 for `text / image / video / image+text / video+text` embedding via both `LLM.embed()` and `vllm serve /v1/embeddings`.

**Architecture:** Reuse the existing `Qwen3VLForConditionalGeneration -> pooling/embed` conversion path instead of introducing a new embedding model class. Keep the implementation narrow: add regression coverage for model resolution, add dedicated online/offline multimodal embedding tests that fix the required three-turn prompt format, update public examples to include conservative single-card defaults, then validate the local FP8 checkpoint on a single V100/PG503-216 in eager and `FULL_AND_PIECEWISE` CUDA graph modes.

**Tech Stack:** Python 3.13, Conda `gptq`, pytest, `RemoteOpenAIServer`, `vllm_runner`, OpenAI-compatible `/v1/embeddings`, Qwen3-VL multimodal processors, local model `/mnt/data6/models/Qwen3-VL-Embedding-8B-FP8`.

---

## File Structure

- Modify `tests/test_config.py`
  - Lock `Qwen/Qwen3-VL-Embedding-2B` to `runner="pooling"` + `convert="embed"` for both default `convert="auto"` and explicit `convert="embed"`.
- Create `tests/entrypoints/pooling/embed/test_online_qwen3_vl.py`
  - Dedicated `/v1/embeddings` chat-style multimodal coverage for `text`, `image`, `video`, `image+text`, `video+text`.
  - Reuse a fixed Qwen3-VL three-turn message builder and conservative single-card server args.
- Create `tests/models/multimodal/pooling/test_qwen3_vl_embedding.py`
  - Offline `vllm_runner` coverage for the same five modalities plus cosine-based semantic sanity checks.
- Modify `examples/pooling/embed/vision_embedding_online.py`
  - Extend `run_qwen3_vl()` to cover video and `video+text`, and document the single-card serving flags.
- Modify `examples/pooling/embed/vision_embedding_offline.py`
  - Extend `run_qwen3_vl()` to cover offline video embedding, `video+text`, and the same conservative defaults.

---

### Task 1: Add Dedicated Online `/v1/embeddings` Coverage For Qwen3-VL

**Files:**
- Create: `tests/entrypoints/pooling/embed/test_online_qwen3_vl.py`
- Test: `tests/entrypoints/pooling/embed/test_online_qwen3_vl.py`

- [ ] **Step 1: Write the dedicated online regression file**

Create `tests/entrypoints/pooling/embed/test_online_qwen3_vl.py` with:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import Any

import numpy as np
import pytest
import requests

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.pooling.embed.protocol import EmbeddingResponse
from vllm.multimodal.utils import (
    encode_image_url,
    encode_video_url,
    fetch_image,
    fetch_video,
)

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
EMBED_DIM = 4096
IMAGE_URL = (
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/"
    "multimodal_asset/cat_snow.jpg"
)
VIDEO_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4"
TEXT = "A cat standing in the snow."
VIDEO_TEXT = "A hand is drawing on paper."
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


@pytest.fixture(scope="module")
def server():
    args = [
        "--runner",
        "pooling",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "1",
        "--enforce-eager",
        "--limit-mm-per-prompt",
        json.dumps({"image": 2, "video": 1}),
        "--media-io-kwargs",
        json.dumps({"video": {"num_frames": 8}}),
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def image_data_uri() -> str:
    return encode_image_url(fetch_image(IMAGE_URL))


@pytest.fixture(scope="module")
def video_data_uri() -> str:
    return encode_video_url(fetch_video(VIDEO_URL)[0])


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
    image_data_uri: str,
    video_data_uri: str,
) -> None:
    del image_data_uri

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
```

- [ ] **Step 2: Run the new online test file**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
pytest tests/entrypoints/pooling/embed/test_online_qwen3_vl.py -v'
```

Expected:

- the file is collected successfully
- each request shape reaches `/v1/embeddings`
- failures, if any, point to real multimodal embedding gaps rather than missing test infrastructure

- [ ] **Step 3: Fix only the online-path breakages exposed by the new file**

Keep the fix narrow. If the failures are about request rendering or tokenization, update the new test file helpers rather than changing the service layer. If the failures are about Qwen3-VL prompt conversion, patch the message shape in the helper to exactly match the three-turn contract:

```python
return [
    {
        "role": "system",
        "content": [{"type": "text", "text": "Represent the user's input."}],
    },
    {"role": "user", "content": content},
    {"role": "assistant", "content": [{"type": "text", "text": ""}]},
]
```

Do not add a second input protocol; keep the test strictly on `/v1/embeddings + messages`.

- [ ] **Step 4: Re-run the online test file and confirm GREEN**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
pytest tests/entrypoints/pooling/embed/test_online_qwen3_vl.py -v'
```

Expected: PASS.

- [ ] **Step 5: Commit the online coverage**

Run:

```bash
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split add \
  tests/entrypoints/pooling/embed/test_online_qwen3_vl.py
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split commit -m \
  "test: 增加 Qwen3-VL embedding 在线多模态覆盖"
```

---

### Task 2: Add Offline `vllm_runner` Coverage For The Same Five Modalities

**Files:**
- Create: `tests/models/multimodal/pooling/test_qwen3_vl_embedding.py`
- Test: `tests/models/multimodal/pooling/test_qwen3_vl_embedding.py`

- [ ] **Step 1: Write the offline pooling regression file**

Create `tests/models/multimodal/pooling/test_qwen3_vl_embedding.py` with:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.multimodal.utils import fetch_image, fetch_video

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
EMBED_DIM = 4096
IMAGE_URL = (
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/"
    "multimodal_asset/cat_snow.jpg"
)
VIDEO_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4"
TEXT = "A cat standing in the snow."
VIDEO_TEXT = "A hand is drawing on paper."
MISMATCH_TEXT = "A tax form on an office desk."


def _text_prompt(text: str) -> str:
    return (
        "<|im_start|>system\n"
        "Represent the user's input.<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _image_prompt(text: str = "") -> str:
    placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    return (
        "<|im_start|>system\n"
        "Represent the user's input.<|im_end|>\n"
        f"<|im_start|>user\n{placeholder}{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _video_prompt(text: str = "") -> str:
    placeholder = "<|vision_start|><|video_pad|><|vision_end|>"
    return (
        "<|im_start|>system\n"
        "Represent the user's input.<|im_end|>\n"
        f"<|im_start|>user\n{placeholder}{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _cosine(lhs: list[float], rhs: list[float]) -> float:
    lhs_arr = np.asarray(lhs, dtype=np.float32)
    rhs_arr = np.asarray(rhs, dtype=np.float32)
    return float(
        np.dot(lhs_arr, rhs_arr)
        / (np.linalg.norm(lhs_arr) * np.linalg.norm(rhs_arr))
    )


@pytest.mark.core_model
def test_qwen3_vl_embedding_modalities(vllm_runner) -> None:
    image = fetch_image(IMAGE_URL)
    video = fetch_video(VIDEO_URL)[0]

    requests = [
        {"prompt": _text_prompt(TEXT)},
        {"prompt": _image_prompt(), "multi_modal_data": {"image": image}},
        {"prompt": _video_prompt(), "multi_modal_data": {"video": video}},
        {"prompt": _image_prompt(TEXT), "multi_modal_data": {"image": image}},
        {"prompt": _video_prompt(VIDEO_TEXT), "multi_modal_data": {"video": video}},
    ]

    with vllm_runner(
        MODEL_NAME,
        runner="pooling",
        dtype="half",
        max_model_len=8192,
        max_num_seqs=1,
        gpu_memory_utilization=0.70,
        limit_mm_per_prompt={"image": 2, "video": 1},
        media_io_kwargs={"video": {"num_frames": 8}},
        enforce_eager=True,
    ) as vllm_model:
        outputs = vllm_model.llm.embed(requests, use_tqdm=False)

    assert len(outputs) == 5
    for output in outputs:
        embedding = np.asarray(output.outputs.embedding, dtype=np.float32)
        assert embedding.shape == (EMBED_DIM,)
        assert np.isfinite(embedding).all()
        assert np.linalg.norm(embedding) > 0.0


@pytest.mark.core_model
def test_qwen3_vl_embedding_semantic_sanity(vllm_runner) -> None:
    image = fetch_image(IMAGE_URL)
    video = fetch_video(VIDEO_URL)[0]

    requests = [
        {"prompt": _text_prompt(TEXT)},
        {"prompt": _image_prompt(TEXT), "multi_modal_data": {"image": image}},
        {"prompt": _image_prompt(MISMATCH_TEXT), "multi_modal_data": {"image": image}},
        {"prompt": _text_prompt(VIDEO_TEXT)},
        {"prompt": _video_prompt(VIDEO_TEXT), "multi_modal_data": {"video": video}},
        {
            "prompt": _video_prompt("Cars driving on a highway."),
            "multi_modal_data": {"video": video},
        },
    ]

    with vllm_runner(
        MODEL_NAME,
        runner="pooling",
        dtype="half",
        max_model_len=8192,
        max_num_seqs=1,
        gpu_memory_utilization=0.70,
        limit_mm_per_prompt={"image": 2, "video": 1},
        media_io_kwargs={"video": {"num_frames": 8}},
        enforce_eager=True,
    ) as vllm_model:
        outputs = vllm_model.llm.embed(requests, use_tqdm=False)

    text_only, image_match, image_mismatch, video_query, video_match, video_mismatch = (
        output.outputs.embedding for output in outputs
    )

    assert _cosine(image_match, text_only) > _cosine(image_mismatch, text_only)
    assert _cosine(video_match, video_query) > _cosine(video_mismatch, video_query)
```

- [ ] **Step 2: Run the new offline test file**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
pytest tests/models/multimodal/pooling/test_qwen3_vl_embedding.py -v'
```

Expected:

- both tests run under `runner="pooling"`
- any failures point to real offline multimodal embedding behavior gaps

- [ ] **Step 3: Keep offline fixes narrow and prompt-format specific**

If the failures are only due to prompt formatting, keep the correction inside the local prompt builders:

```python
placeholder = "<|vision_start|><|video_pad|><|vision_end|>"
return (
    "<|im_start|>system\n"
    "Represent the user's input.<|im_end|>\n"
    f"<|im_start|>user\n{placeholder}{text}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
```

Do not add model-specific conditionals in the runtime until the new offline file proves the current generic path is insufficient.

- [ ] **Step 4: Re-run the offline test file and confirm GREEN**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
pytest tests/models/multimodal/pooling/test_qwen3_vl_embedding.py -v'
```

Expected: PASS.

- [ ] **Step 5: Commit the offline coverage**

Run:

```bash
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split add \
  tests/models/multimodal/pooling/test_qwen3_vl_embedding.py
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split commit -m \
  "test: 增加 Qwen3-VL embedding 离线多模态回归"
```

---

### Task 3: Lock Model Resolution To The Existing Pooling/Embed Adapter Path

**Files:**
- Modify: `tests/test_config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Add config regression cases for Qwen3-VL embedding**

Append these tests near `test_pooling_runner` in `tests/test_config.py`:

```python
@pytest.mark.parametrize(
    ("convert", "expected_convert_type"),
    [
        ("auto", "embed"),
        ("embed", "embed"),
    ],
)
def test_qwen3_vl_embedding_pooling_runner(convert, expected_convert_type):
    config = ModelConfig(
        "Qwen/Qwen3-VL-Embedding-2B",
        runner="pooling",
        convert=convert,
    )

    assert config.runner_type == "pooling"
    assert config.convert_type == expected_convert_type
    assert config.hf_config.architectures == ["Qwen3VLForConditionalGeneration"]
```

- [ ] **Step 2: Run the focused config regression**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
pytest tests/test_config.py::test_qwen3_vl_embedding_pooling_runner -v'
```

Expected: PASS, proving that the current implementation already resolves to the embed adapter without a new model class.

- [ ] **Step 3: Keep the runtime code unchanged if the new regression is green**

Do not add a `Qwen3VLForEmbedding` class when the new config regression already passes. The intended outcome of this task is the explicit test-only guard:

```python
assert config.convert_type == "embed"
assert config.hf_config.architectures == ["Qwen3VLForConditionalGeneration"]
```

- [ ] **Step 4: Re-run the focused config regression together with the existing pooling defaults**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
pytest tests/test_config.py::test_pooling_runner \
tests/test_config.py::test_qwen3_vl_embedding_pooling_runner -v'
```

Expected: PASS.

- [ ] **Step 5: Commit the config guard**

Run:

```bash
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split add \
  tests/test_config.py
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split commit -m \
  "test: 锁定 Qwen3-VL embedding 的 pooling 转换路径"
```

---

### Task 4: Refresh The Public Qwen3-VL Embedding Examples

**Files:**
- Modify: `examples/pooling/embed/vision_embedding_online.py`
- Modify: `examples/pooling/embed/vision_embedding_offline.py`

- [ ] **Step 1: Extend the online example to cover video and single-card flags**

Update `examples/pooling/embed/vision_embedding_online.py`:

```python
video_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4"
video_text = "A hand is drawing on paper."


def _qwen3_vl_messages(
    content: list[dict[str, object]],
) -> list[ChatCompletionMessageParam]:
    default_instruction = "Represent the user's input."
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": default_instruction}],
        },
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": ""}]},
    ]


def run_qwen3_vl(client: OpenAI, model: str):
    """
    Start the server using:

    vllm serve Qwen/Qwen3-VL-Embedding-2B \
        --runner pooling \
        --max-model-len 8192 \
        --max-num-seqs 1 \
        --limit-mm-per-prompt '{"image":2,"video":1}' \
        --media-io-kwargs '{"video":{"num_frames":8}}'
    """
    print("Text embedding output:")
    response = create_chat_embeddings(
        client,
        messages=_qwen3_vl_messages([{"type": "text", "text": text}]),
        model=model,
        encoding_format="float",
        continue_final_message=True,
        add_special_tokens=True,
    )
    print_embeddings(response.data[0].embedding)

    print("Image embedding output:")
    response = create_chat_embeddings(
        client,
        messages=_qwen3_vl_messages(
            [{"type": "image_url", "image_url": {"url": image_url}}]
        ),
        model=model,
        encoding_format="float",
        continue_final_message=True,
        add_special_tokens=True,
    )
    print_embeddings(response.data[0].embedding)

    print("Image+Text embedding output:")
    response = create_chat_embeddings(
        client,
        messages=_qwen3_vl_messages(
            [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": text},
            ]
        ),
        model=model,
        encoding_format="float",
        continue_final_message=True,
        add_special_tokens=True,
    )
    print_embeddings(response.data[0].embedding)

    print("Video embedding output:")
    response = create_chat_embeddings(
        client,
        messages=_qwen3_vl_messages(
            [{"type": "video_url", "video_url": {"url": video_url}}]
        ),
        model=model,
        encoding_format="float",
        continue_final_message=True,
        add_special_tokens=True,
    )
    print_embeddings(response.data[0].embedding)

    print("Video+Text embedding output:")
    response = create_chat_embeddings(
        client,
        messages=_qwen3_vl_messages(
            [
                {"type": "video_url", "video_url": {"url": video_url}},
                {"type": "text", "text": video_text},
            ]
        ),
        model=model,
        encoding_format="float",
        continue_final_message=True,
        add_special_tokens=True,
    )
    print_embeddings(response.data[0].embedding)
```

- [ ] **Step 2: Extend the offline example with matching prompt builders**

Update `examples/pooling/embed/vision_embedding_offline.py`:

```python
from vllm.multimodal.utils import fetch_image, fetch_video

video_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4"
video_text = "A hand is drawing on paper."
multi_modal_video_data = {"video": fetch_video(video_url)[0]}


def _qwen3_vl_prompt(placeholder: str | None = None, text: str = "") -> str:
    user_payload = text if placeholder is None else f"{placeholder}{text}"
    return (
        "<|im_start|>system\n"
        "Represent the user's input.<|im_end|>\n"
        f"<|im_start|>user\n{user_payload}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def run_qwen3_vl(seed: int):
    image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    video_placeholder = "<|vision_start|><|video_pad|><|vision_end|>"

    llm = LLM(
        model="Qwen/Qwen3-VL-Embedding-2B",
        runner="pooling",
        max_model_len=8192,
        max_num_seqs=1,
        limit_mm_per_prompt={"image": 2, "video": 1},
        media_io_kwargs={"video": {"num_frames": 8}},
        mm_processor_kwargs={"do_resize": False} if smart_resize is not None else None,
        seed=seed,
    )

    print("Text embedding output:")
    outputs = llm.embed(_qwen3_vl_prompt(text=text), use_tqdm=False)
    print_embeddings(outputs[0].outputs.embedding)

    print("Image embedding output:")
    outputs = llm.embed(
        {
            "prompt": _qwen3_vl_prompt(image_placeholder),
            "multi_modal_data": multi_modal_data,
        },
        use_tqdm=False,
    )
    print_embeddings(outputs[0].outputs.embedding)

    print("Image+Text embedding output:")
    outputs = llm.embed(
        {
            "prompt": _qwen3_vl_prompt(image_placeholder, text),
            "multi_modal_data": multi_modal_data,
        },
        use_tqdm=False,
    )
    print_embeddings(outputs[0].outputs.embedding)

    print("Video embedding output:")
    outputs = llm.embed(
        {
            "prompt": _qwen3_vl_prompt(video_placeholder),
            "multi_modal_data": multi_modal_video_data,
        },
        use_tqdm=False,
    )
    print_embeddings(outputs[0].outputs.embedding)

    print("Video+Text embedding output:")
    outputs = llm.embed(
        {
            "prompt": _qwen3_vl_prompt(video_placeholder, video_text),
            "multi_modal_data": multi_modal_video_data,
        },
        use_tqdm=False,
    )
    print_embeddings(outputs[0].outputs.embedding)
```

- [ ] **Step 3: Verify both examples still parse**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
python -m compileall \
  examples/pooling/embed/vision_embedding_online.py \
  examples/pooling/embed/vision_embedding_offline.py'
```

Expected: both files compile successfully.

- [ ] **Step 4: Re-run the new focused tests after updating the examples**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
pytest tests/entrypoints/pooling/embed/test_online_qwen3_vl.py \
tests/models/multimodal/pooling/test_qwen3_vl_embedding.py \
tests/test_config.py::test_qwen3_vl_embedding_pooling_runner -q'
```

Expected: PASS.

- [ ] **Step 5: Commit the example refresh**

Run:

```bash
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split add \
  examples/pooling/embed/vision_embedding_online.py \
  examples/pooling/embed/vision_embedding_offline.py
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split commit -m \
  "docs: 补齐 Qwen3-VL embedding 的图像与视频示例"
```

---

### Task 5: Run The Focused Regression Suite

**Files:**
- No source edits expected.

- [ ] **Step 1: Run the config and online regression set**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
pytest \
  tests/test_config.py::test_qwen3_vl_embedding_pooling_runner \
  tests/entrypoints/pooling/embed/test_online_qwen3_vl.py -v'
```

Expected: PASS.

- [ ] **Step 2: Run the offline pooling regression set**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
pytest tests/models/multimodal/pooling/test_qwen3_vl_embedding.py -v'
```

Expected: PASS.

- [ ] **Step 3: Run the existing generic embedding API regression to guard against route breakage**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
pytest tests/entrypoints/pooling/embed/test_online.py -q'
```

Expected: PASS.

- [ ] **Step 4: Commit only if the regression pass required follow-up fixes**

Run this only if Task 5 exposed additional source edits:

```bash
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split add \
  tests/test_config.py \
  tests/entrypoints/pooling/embed/test_online_qwen3_vl.py \
  tests/models/multimodal/pooling/test_qwen3_vl_embedding.py \
  examples/pooling/embed/vision_embedding_online.py \
  examples/pooling/embed/vision_embedding_offline.py
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split commit -m \
  "fix: 收敛 Qwen3-VL embedding 多模态回归"
```

---

### Task 6: Validate The Local FP8 Checkpoint On Single-Card SM70

**Files:**
- Create temporary scripts under `/tmp` only.
- No repo source edits expected.

- [ ] **Step 1: Pick a lightly loaded V100 / PG503-216**

Run:

```bash
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader,nounits
```

Expected: choose one card from `2,3,4,5` with enough free memory for a single-card 8B FP8 pooling service.

- [ ] **Step 2: Start the local model in eager mode for bring-up**

Run:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split \
/home/z/anaconda3/envs/gptq/bin/vllm serve \
  /mnt/data6/models/Qwen3-VL-Embedding-8B-FP8 \
  --host 127.0.0.1 \
  --port 24166 \
  --served-model-name qwen3-vl-embed-8b-fp8-sm70 \
  --runner pooling \
  --convert embed \
  --dtype float16 \
  --gpu-memory-utilization 0.78 \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":2,"video":1}' \
  --media-io-kwargs '{"video":{"num_frames":8}}' \
  --enforce-eager
```

Expected log evidence:

```text
Resolved architecture: Qwen3VLForConditionalGeneration
runner=pooling
convert=embed
Starting vLLM server on http://127.0.0.1:24166
```

- [ ] **Step 3: Confirm the service is visible**

Run:

```bash
curl -sf http://127.0.0.1:24166/v1/models
```

Expected: JSON contains `"id":"qwen3-vl-embed-8b-fp8-sm70"`.

- [ ] **Step 4: Create and run a reusable multimodal smoke script**

Create `/tmp/qwen3_vl_embedding_smoke.py` with:

```python
import numpy as np
import requests

from vllm.multimodal.utils import (
    encode_image_url,
    encode_video_url,
    fetch_image,
    fetch_video,
)

BASE_URL = "http://127.0.0.1:24166/v1"
MODEL = "qwen3-vl-embed-8b-fp8-sm70"
IMAGE_URL = (
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/"
    "multimodal_asset/cat_snow.jpg"
)
VIDEO_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4"
TEXT = "A cat standing in the snow."
VIDEO_TEXT = "A hand is drawing on paper."
MISMATCH_TEXT = "A tax form on an office desk."

IMAGE_DATA_URI = encode_image_url(fetch_image(IMAGE_URL))
VIDEO_DATA_URI = encode_video_url(fetch_video(VIDEO_URL)[0])


def make_messages(content):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Represent the user's input."}],
        },
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": ""}]},
    ]


def embed(content, **extra_body):
    response = requests.post(
        BASE_URL + "/embeddings",
        json={
            "model": MODEL,
            "messages": make_messages(content),
            "encoding_format": "float",
            "continue_final_message": True,
            "add_special_tokens": True,
            **extra_body,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def cosine(lhs, rhs):
    lhs_arr = np.asarray(lhs, dtype=np.float32)
    rhs_arr = np.asarray(rhs, dtype=np.float32)
    return float(
        np.dot(lhs_arr, rhs_arr)
        / (np.linalg.norm(lhs_arr) * np.linalg.norm(rhs_arr))
    )


cases = {
    "text": [{"type": "text", "text": TEXT}],
    "image": [{"type": "image_url", "image_url": {"url": IMAGE_DATA_URI}}],
    "video": [{"type": "video_url", "video_url": {"url": VIDEO_DATA_URI}}],
    "image+text": [
        {"type": "image_url", "image_url": {"url": IMAGE_DATA_URI}},
        {"type": "text", "text": TEXT},
    ],
    "video+text": [
        {"type": "video_url", "video_url": {"url": VIDEO_DATA_URI}},
        {"type": "text", "text": VIDEO_TEXT},
    ],
}

outputs = {}
for name, content in cases.items():
    payload = embed(content)
    emb = payload["data"][0]["embedding"]
    outputs[name] = emb
    arr = np.asarray(emb, dtype=np.float32)
    print(
        name,
        "dim=",
        len(emb),
        "prompt_tokens=",
        payload["usage"]["prompt_tokens"],
        "norm=",
        float(np.linalg.norm(arr)),
    )

text_query = embed([{"type": "text", "text": TEXT}])["data"][0]["embedding"]
image_match = embed(
    [
        {"type": "image_url", "image_url": {"url": IMAGE_DATA_URI}},
        {"type": "text", "text": TEXT},
    ]
)["data"][0]["embedding"]
image_mismatch = embed(
    [
        {"type": "image_url", "image_url": {"url": IMAGE_DATA_URI}},
        {"type": "text", "text": MISMATCH_TEXT},
    ]
)["data"][0]["embedding"]
video_query = embed([{"type": "text", "text": VIDEO_TEXT}])["data"][0]["embedding"]
video_match = embed(
    [
        {"type": "video_url", "video_url": {"url": VIDEO_DATA_URI}},
        {"type": "text", "text": VIDEO_TEXT},
    ]
)["data"][0]["embedding"]
video_mismatch = embed(
    [
        {"type": "video_url", "video_url": {"url": VIDEO_DATA_URI}},
        {"type": "text", "text": "Cars driving on a highway."},
    ]
)["data"][0]["embedding"]

print(
    "image semantic:",
    cosine(image_match, text_query),
    ">",
    cosine(image_mismatch, text_query),
)
print(
    "video semantic:",
    cosine(video_match, video_query),
    ">",
    cosine(video_mismatch, video_query),
)
```

Run:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split \
/home/z/anaconda3/envs/gptq/bin/python /tmp/qwen3_vl_embedding_smoke.py
```

Expected:

- all five modalities return `dim= 4096`
- each embedding norm is non-zero
- both printed semantic inequalities hold

- [ ] **Step 5: Re-run the same service in formal CUDA graph mode**

Stop the eager server, then start:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split \
/home/z/anaconda3/envs/gptq/bin/vllm serve \
  /mnt/data6/models/Qwen3-VL-Embedding-8B-FP8 \
  --host 127.0.0.1 \
  --port 24166 \
  --served-model-name qwen3-vl-embed-8b-fp8-sm70 \
  --runner pooling \
  --convert embed \
  --dtype float16 \
  --gpu-memory-utilization 0.78 \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":2,"video":1}' \
  --media-io-kwargs '{"video":{"num_frames":8}}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1]}'
```

Then re-run:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split \
/home/z/anaconda3/envs/gptq/bin/python /tmp/qwen3_vl_embedding_smoke.py
```

Expected:

- the service stays up with `FULL_AND_PIECEWISE`
- the smoke script still returns correct 4096-dim embeddings
- semantic inequalities remain true in graph mode

---

## Final Verification Checklist

- `pytest tests/entrypoints/pooling/embed/test_online_qwen3_vl.py -v`
- `pytest tests/models/multimodal/pooling/test_qwen3_vl_embedding.py -v`
- `pytest tests/test_config.py::test_qwen3_vl_embedding_pooling_runner -v`
- `python -m compileall examples/pooling/embed/vision_embedding_online.py examples/pooling/embed/vision_embedding_offline.py`
- single-card eager smoke on `/mnt/data6/models/Qwen3-VL-Embedding-8B-FP8`
- single-card `FULL_AND_PIECEWISE` smoke on `/mnt/data6/models/Qwen3-VL-Embedding-8B-FP8`

## Commit Strategy

1. `test: 增加 Qwen3-VL embedding 在线多模态覆盖`
2. `test: 增加 Qwen3-VL embedding 离线多模态回归`
3. `test: 锁定 Qwen3-VL embedding 的 pooling 转换路径`
4. `docs: 补齐 Qwen3-VL embedding 的图像与视频示例`
5. `fix: 收敛 Qwen3-VL embedding 多模态回归` only if Task 5 exposed follow-up source edits
