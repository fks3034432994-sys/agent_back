"""服务端协作式取消的注册表（MCP 服务进程内）。

这是"点了停止能不能真的停下正在跑的 shell/GUI 任务"的核心状态机，必须逐个分支测。
"""

import threading

from src.agents.mcps.cancel_registry import CancelRegistry, attach_cancel_tool
from src.agents.mcps.registry import CANCEL_TOOL_NAME


def test_register_only_allocates_event_with_task_id():
    reg = CancelRegistry()
    # 没有 task_id 的调用不可取消，返回 None
    assert reg.register("") is None
    assert isinstance(reg.register("t1"), threading.Event)


def test_cancel_sets_event_and_is_idempotent():
    reg = CancelRegistry()
    event = reg.register("t1")

    assert reg.cancel("t1") is True
    assert event.is_set()
    # 工具还没退出，重复取消不应报错
    assert reg.cancel("t1") is True


def test_cancel_unknown_task_returns_false():
    assert CancelRegistry().cancel("nope") is False


def test_unregister_stops_further_cancellation():
    reg = CancelRegistry()
    reg.register("t1")
    reg.unregister("t1")

    assert reg.cancel("t1") is False


def test_cancel_is_scoped_per_task():
    reg = CancelRegistry()
    first = reg.register("a")
    second = reg.register("b")

    reg.cancel("a")

    assert first.is_set()
    assert not second.is_set(), "取消一个任务不能波及另一个任务"


def test_attach_cancel_tool_exposes_working_tool():
    """attach 出来的 cancel_task 必须真的能置位信号（主进程靠它发取消）。"""
    captured = {}

    class FakeMCP:
        def tool(self):
            def decorator(fn):
                captured[fn.__name__] = fn
                return fn

            return decorator

    reg = CancelRegistry()
    attach_cancel_tool(FakeMCP(), reg)

    assert CANCEL_TOOL_NAME in captured
    task = reg.register("t9")

    assert captured[CANCEL_TOOL_NAME]("t9") == "已发送取消信号"
    assert task.is_set()
    assert captured[CANCEL_TOOL_NAME]("t9-不存在") == "任务不存在或已结束"
