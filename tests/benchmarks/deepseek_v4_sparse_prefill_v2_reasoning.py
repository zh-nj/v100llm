#!/usr/bin/env python3
import json
import sys
import time
import urllib.request


ENDPOINT = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8199/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "/mnt/data6/models/DeepSeek-V4-Flash"


def post(payload: dict) -> tuple[dict, float]:
    request = urllib.request.Request(
        ENDPOINT + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body, time.perf_counter() - start


def main() -> int:
    plain, plain_s = post(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "2+3等于几？只回答结果。"}],
            "temperature": 0.0,
            "max_tokens": 16,
        }
    )
    plain_choice = (plain.get("choices") or [{}])[0]
    plain_text = ((plain_choice.get("message") or {}).get("content") or "").strip()

    tool, tool_s = post(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "请调用工具查询北京今天的天气。"}],
            "temperature": 0.0,
            "max_tokens": 64,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                            },
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        }
    )
    tool_choice = (tool.get("choices") or [{}])[0]
    tool_message = tool_choice.get("message") or {}
    tool_calls = tool_message.get("tool_calls") or []
    result = {
        "plain_answer": plain_text,
        "plain_wall_s": plain_s,
        "plain_usage": plain.get("usage"),
        "tool_wall_s": tool_s,
        "tool_finish_reason": tool_choice.get("finish_reason"),
        "tool_call_count": len(tool_calls),
        "tool_calls": tool_calls,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    ok_plain = "5" in plain_text
    ok_tool = bool(tool_calls)
    return 0 if ok_plain and ok_tool else 1


if __name__ == "__main__":
    raise SystemExit(main())
