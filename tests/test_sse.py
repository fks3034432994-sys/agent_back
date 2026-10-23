"""SSE 协议格式：前端手写解析器就靠这几条不变式。"""

import json

from src.api.sse import sse_event, sse_ping


def _parse(block: str) -> tuple[str, dict]:
    """把一条 SSE 事件解回 (event, data)，顺带校验帧格式。"""
    assert block.endswith("\n\n"), "事件必须以空行结尾"
    lines = block[:-2].split("\n")
    assert len(lines) == 2, f"一条事件应只有 event/data 两行，实际 {lines}"
    assert lines[0].startswith("event: ")
    assert lines[1].startswith("data: ")
    return lines[0][7:], json.loads(lines[1][6:])


def test_event_format_is_parsable():
    name, data = _parse(sse_event("token", {"content": "你好"}))
    assert name == "token"
    assert data == {"content": "你好"}


def test_chinese_is_not_escaped():
    # ensure_ascii=False：中文以原字符下发，前端不必二次解码
    assert "你好" in sse_event("token", {"content": "你好"})
    assert "\\u" not in sse_event("token", {"content": "你好"})


def test_token_with_blank_lines_does_not_split_the_event():
    """模型输出的代码块/段落里带空行时，不能把一条事件切成两条。"""
    block = sse_event("token", {"content": "第一段\n\n第二段\n"})
    assert block.count("\n\n") == 1  # 只有结尾那个分隔空行
    assert _parse(block) == ("token", {"content": "第一段\n\n第二段\n"})


def test_ping_is_a_comment_line():
    # 心跳是 SSE 注释行（以 ":" 开头），客户端解析器会直接忽略
    assert sse_ping() == ": ping\n\n"
