"""主进程"本请求在飞工具"记录。

核心不变量：取消时只能看到**本请求**正在执行的工具，不能把其它会话的任务一起取消。
这里用并发任务验证 ContextVar 的隔离性——这正是当初不用全局 dict 的原因。
"""

import asyncio

from src.agents.mcps import tool_registry as tr
from src.asyncio_runner import run_async


def test_register_outside_request_context_is_safe_noop():
    # 没有 start_recording（例如模块被直接导入、非请求路径）时不应抛异常
    tr.register_running_tool("c1", "read_file")
    tr.unregister_running_tool("c1")


def test_recording_container_is_live_and_cleared_on_unregister():
    recording = tr.start_recording()

    tr.register_running_tool("c1", "read_file")
    assert recording == {"c1": "read_file"}

    tr.unregister_running_tool("c1")
    assert recording == {}


def test_unregister_unknown_id_is_harmless():
    recording = tr.start_recording()
    tr.unregister_running_tool("never-registered")
    assert recording == {}


def test_records_are_isolated_between_concurrent_requests():
    async def worker(call_id: str, tool_name: str) -> dict[str, str]:
        recording = tr.start_recording()
        tr.register_running_tool(call_id, tool_name)
        await asyncio.sleep(0.01)  # 让两个请求真正重叠
        return dict(recording)

    async def main():
        return await asyncio.gather(
            worker("c1", "run_shell_command"),
            worker("c2", "search_knowledge"),
        )

    first, second = run_async(main())

    assert first == {"c1": "run_shell_command"}
    assert second == {"c2": "search_knowledge"}
