"""MCP 服务进程内的"可取消任务"注册表（服务端）。

HTTP transport 下 MCP 服务是常驻进程，客户端断开时不能照搬旧 stdio 方案的
"按 PID 杀进程"——那会连带杀死**其它会话**正在执行的任务。改为协作式取消：

1. 主进程把 tool_call_id 作为 task_id 注入长耗时工具的参数（见 main_graph）；
2. 工具开始执行时用本模块登记一个 Event；
3. 主进程调用 cancel_task(task_id) 置位该 Event（内部工具，不给模型看）；
4. 工具在轮询/步骤边界检查 Event，主动终止子进程或跳出循环。

为什么必须"登记 Event + 独立工具"而不能靠传输层取消：FastMCP 把同步工具派发到工作
线程执行（run_in_thread），asyncio 无法中断线程中阻塞的同步代码，请求取消后线程仍会
跑下去。所以需要工具自己观察信号。cancel_task 是瞬时返回的短操作，不会被长任务挡住。

本模块被 MCP 服务进程导入，**不得**依赖主进程侧的模块（否则 import 环）。
"""

from __future__ import annotations

import threading
from typing import Any


class CancelRegistry:
    """task_id -> 取消信号（Event）。dict/lock 操作受 GIL 保护，跨线程安全。"""

    def __init__(self) -> None:
        self._events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def register(self, task_id: str) -> threading.Event | None:
        """开始执行时登记；task_id 为空（不带取消能力）时返回 None。"""
        if not task_id:
            return None
        event = threading.Event()
        with self._lock:
            self._events[task_id] = event
        return event

    def unregister(self, task_id: str) -> None:
        with self._lock:
            self._events.pop(task_id, None)

    def cancel(self, task_id: str) -> bool:
        """置位取消信号；返回 False 表示任务不存在或已结束。"""
        with self._lock:
            event = self._events.get(task_id)
        if event is None:
            return False
        event.set()
        return True


def attach_cancel_tool(mcp: Any, registry: CancelRegistry) -> None:
    """把 cancel_task 工具注册到 MCP 服务上（供主进程调用，模型不可见）。"""

    @mcp.tool()
    def cancel_task(task_id: str) -> str:
        """取消一个正在执行的长耗时任务。

        内部工具：由主进程在客户端断开/点击停止时调用，不建议由模型直接调用。

        参数:
            task_id: 目标任务的 ID（即该次工具调用的 tool_call_id）

        返回:
            已发送取消信号 / 任务不存在或已结束
        """
        return "已发送取消信号" if registry.cancel(task_id) else "任务不存在或已结束"
