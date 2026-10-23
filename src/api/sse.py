import json
from typing import Any


def sse_event(event: str, data: Any) -> str:
    """构造一条 SSE 事件：event 行 + data 行（JSON），以空行结尾。"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def sse_ping() -> str:
    """心跳注释行，防止代理/网关因长时间无数据断开连接。"""
    return ": ping\n\n"
