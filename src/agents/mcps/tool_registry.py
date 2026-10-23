"""本轮请求的"正在执行的工具调用"记录（主进程）。

用途只有一个：客户端断开/点击停止时，要知道**本次请求**此刻有哪些工具调用在执行，
才能反查它属于哪个 MCP 服务并请求取消（见 main_graph.cancel_running_tasks）。

为什么用 ContextVar 而不是全局 dict：全局字典会把其它会话正在执行的工具也算进来，
取消时会误伤别人的任务。ContextVar 的值随 asyncio 任务派生自动传递，天然做到
"一个请求一份"；容器是可变对象，子任务里 mutate 对父任务可见（context 是浅拷贝，
拷贝的是同一个容器的引用）。

时序要求：取消会让节点的 finally 立刻注销登记，所以调用方必须**先快照、后取消**
（见 api/chat.py 中 _stream_events 的收尾逻辑）。

与旧版 kill_tools 的区别：旧版记的是"工具名 -> MCP 进程 PID"然后杀进程；HTTP 常驻
服务不能整体杀掉，改为按 tool_call_id 发起协作式取消，本模块只负责记录"有哪些在跑"。
"""

from __future__ import annotations

import contextvars

# 当前请求的工具调用记录：tool_call_id -> tool_name
_running: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "running_tool_calls", default=None
)


def start_recording() -> dict[str, str]:
    """为一个请求建立工具调用记录，返回承载容器。

    调用方持有这个容器引用，随时都能读到当前快照（不用再查全局状态）。
    """
    container: dict[str, str] = {}
    _running.set(container)
    return container


def register_running_tool(call_id: str, tool_name: str) -> None:
    """工具开始执行前登记。"""
    container = _running.get()
    if container is not None:
        container[call_id] = tool_name


def unregister_running_tool(call_id: str) -> None:
    """工具执行结束（含异常、含被取消）后注销。"""
    container = _running.get()
    if container is not None:
        container.pop(call_id, None)
