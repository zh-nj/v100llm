#!/usr/bin/env python3
import argparse
import json
import sys
import time
import uuid
import urllib.request


BASE = (
    "DeepSeek V4 sparse prefill measurement paragraph. "
    "请保持上下文唯一，下面是用于预填充性能测试的重复内容。"
)


def normalize_endpoint(raw: str) -> str:
    endpoint = raw.rstrip("/")
    if endpoint.endswith("/v1"):
        return endpoint + "/chat/completions"
    return endpoint + "/v1/chat/completions"


def load_tokenizer(model: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception:
        return None


def token_len(tokenizer, text: str) -> int:
    if tokenizer is None:
        return max(1, len(text) // 4)
    return len(tokenizer.encode(text, add_special_tokens=False))


def make_prompt(tokenizer, target: int) -> str:
    marker = f"UNIQUE_PREFILL_{target}_{uuid.uuid4().hex}\n"
    unit = BASE + f" target={target}.\n"
    unit_tokens = max(token_len(tokenizer, unit), 1)
    repeats = max(1, target // unit_tokens)
    lo = max(1, repeats // 2)
    hi = max(lo + 1, repeats * 2 + 8)
    best = marker + unit * repeats
    best_delta = abs(token_len(tokenizer, best) - target)
    for _ in range(24):
        mid = (lo + hi) // 2
        candidate = marker + unit * mid
        n_tokens = token_len(tokenizer, candidate)
        delta = abs(n_tokens - target)
        if delta < best_delta:
            best = candidate
            best_delta = delta
        if n_tokens < target:
            lo = mid + 1
        else:
            hi = mid
    return best


def parse_sse_line(raw_line: bytes) -> dict | None:
    line = raw_line.decode("utf-8", errors="replace").strip()
    if not line.startswith("data:"):
        return None
    payload = line.removeprefix("data:").strip()
    if payload == "[DONE]":
        return {"done": True}
    return json.loads(payload)


def choice_contains_generated_delta(choice: dict) -> bool:
    token_ids = choice.get("token_ids")
    if token_ids:
        return True
    delta = choice.get("delta") or {}
    content = delta.get("content") or ""
    if content:
        return True
    return "content" in delta and "role" not in delta


def request_nonstream(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: int,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = json.loads(response.read().decode("utf-8"))
    total_s = time.perf_counter() - start
    usage = body.get("usage") or {}
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") or choice.get("text") or ""
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_s": total_s,
        "prefill_plus_one_tps": (
            prompt_tokens / total_s if prompt_tokens is not None and total_s > 0 else None
        ),
        "decode_tokens_per_s": None,
        "finish_reason": choice.get("finish_reason"),
        "output_preview": text[:120],
    }


def request_stream(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: int,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first_token_time = None
    prompt_tokens = None
    completion_tokens = None
    finish_reason = None
    pieces: list[str] = []
    delta_chunks = 0
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        for raw_line in response:
            chunk = parse_sse_line(raw_line)
            if chunk is None:
                continue
            if chunk.get("done"):
                break
            usage = chunk.get("usage")
            if usage and usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])
            if usage and usage.get("prompt_tokens") is not None:
                prompt_tokens = int(usage["prompt_tokens"])
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            content = (choice.get("delta") or {}).get("content") or ""
            if choice_contains_generated_delta(choice):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                delta_chunks += 1
            if content:
                pieces.append(content)
    end = time.perf_counter()
    ttft_s = None if first_token_time is None else first_token_time - start
    decode_window_s = None if first_token_time is None else max(end - first_token_time, 1e-9)
    generated = completion_tokens if completion_tokens is not None else delta_chunks
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_s": end - start,
        "ttft_s": ttft_s,
        "prefill_tps_ttft": (
            prompt_tokens / ttft_s
            if prompt_tokens is not None and ttft_s is not None and ttft_s > 0
            else None
        ),
        "decode_window_s": decode_window_s,
        "decode_tokens_per_s": (
            generated / decode_window_s
            if decode_window_s is not None and decode_window_s > 0
            else None
        ),
        "finish_reason": finish_reason,
        "output_preview": "".join(pieces)[:120],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint")
    parser.add_argument("model")
    parser.add_argument("lengths_csv")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="HTTP request timeout in seconds for each prompt length.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Measure TTFT and decode tok/s with streamed output.",
    )
    args = parser.parse_args()

    endpoint = normalize_endpoint(args.endpoint)
    model = args.model
    lengths = [int(part) for part in args.lengths_csv.split(",") if part]
    tokenizer = load_tokenizer(model)
    results = []
    for target in lengths:
        prompt = make_prompt(tokenizer, target)
        local_prompt_tokens = token_len(tokenizer, prompt)
        if args.stream:
            row = request_stream(endpoint, model, prompt, args.max_tokens,
                                 args.timeout)
        else:
            row = request_nonstream(endpoint, model, prompt, args.max_tokens,
                                    args.timeout)
        row["target"] = target
        row["tokenized_before_request"] = local_prompt_tokens
        row["stream"] = args.stream
        row["max_tokens"] = args.max_tokens
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
        results.append(row)
    print(json.dumps({"results": results}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
