"""SSE 事件流（api/chat.py 的 _stream_events）。

用假 agent 喂脚本化事件，覆盖：token 去重、工具事件、ask_user 中断、图异常、心跳、
以及客户端断开时请求取消工具。**不依赖数据库/MCP/大模型**。

这里包含两条回归用例，对应实际踩过的坑：
- 流正常结束时不能让 StopAsyncIteration 逃逸（Python 会把它包成 RuntimeError，
  客户端表现为"响应提前结束"、收不到 done）；
- 客户端断开时必须发出取消请求（长耗时工具跑在 MCP 服务的工作线程里，协程取消不了
  它，只能靠服务内的 cancel_task 协作式终止）。

"客户端断开"在测试里的模拟方式与线上一致：**取消正在消费事件流的那个任务**
（Starlette 的 listen_for_disconnect 就是这么干的），而不是去模拟 is_disconnected()。
"""

import asyncio
import contextlib
import json

from langchain.messages import AIMessage, AIMessageChunk, ToolMessage

from src.api import chat
from src.api.schemas import ChatRequest
from src.asyncio_runner import run_async


class FakeSnapshot:
    def __init__(self, messages=(), interrupts=(), tasks=(), task_completed=False):
        self.values = {"messages": list(messages), "task_completed": task_completed}
        self.interrupts = tuple(interrupts)
        self.tasks = tuple(tasks)


class FakeAgent:
    """按给定事件序列产出 (stream_mode, payload)；error 在流末尾抛出。"""

    def __init__(self, events, snapshot=None, error=None):
        self._events = list(events)
        self._snapshot = snapshot or FakeSnapshot()
        self._error = error

    async def aget_state(self, config):
        return self._snapshot

    def astream(self, graph_input, config=None, stream_mode=None):
        return self._iterate()

    async def _iterate(self):
        for event in self._events:
            yield event
        if self._error is not None:
            raise self._error


def _parse(frame: str):
    """把一帧 SSE 文本解析成 (event, data)；心跳返回 ("ping", None)。"""
    if frame.startswith(":"):
        return ("ping", None)
    event = data = None
    for line in frame.strip().split("\n"):
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: "):
            data = json.loads(line[6:])
    return event, data


async def _no_reference(uris):
    return ""


def _patch_stream(monkeypatch, agent):
    monkeypatch.setattr(chat, "get_main_agent", lambda: agent)
    monkeypatch.setattr(chat, "_THREAD_GATE", chat._ThreadGate())
    # 不测参考资料注入，短路避免连 MCP resource
    monkeypatch.setattr(chat, "load_reference_text", _no_reference)


def run_stream(monkeypatch, agent, *, req=None):
    """正常消费完整条流，返回解析后的事件列表。"""
    _patch_stream(monkeypatch, agent)

    async def drain():
        frames = [
            frame
            async for frame in chat._stream_events(req or ChatRequest(message="hi"))
        ]
        await asyncio.sleep(0.05)  # 收尾里发起的独立任务留出执行机会
        return frames

    return [_parse(frame) for frame in run_async(drain())]


def run_stream_then_cancel(monkeypatch, agent, *, req=None, cancel_after=0.05):
    """模拟客户端断开：消费中途取消消费任务（与 Starlette 的行为一致）。"""
    _patch_stream(monkeypatch, agent)

    async def drain():
        frames: list[str] = []

        async def consume():
            async for frame in chat._stream_events(req or ChatRequest(message="hi")):
                frames.append(frame)

        task = asyncio.create_task(consume())
        await asyncio.sleep(cancel_after)  # 让图先跑起来
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.05)  # 给"独立发送取消请求"的任务一点时间
        return frames

    return [_parse(frame) for frame in run_async(drain())]


def test_tokens_are_streamed_once_and_done_is_emitted(monkeypatch):
    agent = FakeAgent(
        [
            ("messages", (AIMessageChunk(content="上"), {})),
            ("messages", (AIMessageChunk(content="午好"), {})),
            # 流式块之后的完整消息内容重复，必须跳过
            ("messages", (AIMessage(content="上午好"), {})),
        ],
        snapshot=FakeSnapshot(messages=[AIMessage(content="上午好")]),
    )

    events = run_stream(monkeypatch, agent)
    names = [name for name, _ in events]

    assert names.count("token") == 2, "完整消息不应被当成第三个 token 重复发出"
    assert "".join(data["content"] for name, data in events if name == "token") == "上午好"
    assert events[-1][0] == "done"
    assert events[-1][1]["awaiting_user"] is False


def test_normal_stream_end_does_not_leak_stop_async_iteration(monkeypatch):
    """回归：StopAsyncIteration 一旦从 async generator 逃逸，Python 会包成
    RuntimeError("async generator raised StopAsyncIteration")，前端只看到连接断掉。"""
    agent = FakeAgent([("messages", (AIMessageChunk(content="x"), {}))])

    events = run_stream(monkeypatch, agent)

    assert events[-1][0] == "done"


def test_done_reports_task_completed_and_final_content(monkeypatch):
    final = AIMessage(content="做好了")
    agent = FakeAgent([], snapshot=FakeSnapshot(messages=[final], task_completed=True))

    events = run_stream(monkeypatch, agent)

    name, data = events[-1]
    assert name == "done"
    assert data["task_completed"] is True
    assert data["content"] == "做好了"


def test_non_streaming_model_content_is_fallen_back(monkeypatch):
    """模型不支持流式时（只来一条完整消息），仍要保证前端拿到正文。"""
    agent = FakeAgent([("messages", (AIMessage(content="整段回复"), {}))])

    events = run_stream(monkeypatch, agent)
    tokens = [data["content"] for name, data in events if name == "token"]

    assert tokens == ["整段回复"]


def test_tool_call_and_result_events(monkeypatch):
    ai = AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "a.txt"}, "id": "c1"}])
    tool = ToolMessage(content="文件内容", tool_call_id="c1", name="read_file")
    agent = FakeAgent([("updates", {"main_model_call": {"messages": [ai, tool]}})])

    events = run_stream(monkeypatch, agent)

    assert ("tool_call", {"name": "read_file", "args": {"path": "a.txt"}, "id": "c1"}) in events
    assert (
        "tool_result",
        {"name": "read_file", "content": "文件内容", "tool_call_id": "c1"},
    ) in events


def test_tool_message_in_messages_mode_is_not_emitted_as_token(monkeypatch):
    tool = ToolMessage(content="工具结果", tool_call_id="c1", name="read_file")
    agent = FakeAgent([("messages", (tool, {}))])

    events = run_stream(monkeypatch, agent)

    assert not [name for name, _ in events if name == "token"]


def test_interrupt_emits_ask_user_and_awaiting_done(monkeypatch):
    class Interrupt:
        value = {"question": "要删哪个文件？"}
        id = "i-9"

    agent = FakeAgent([("updates", {chat.INTERRUPT_KEY: (Interrupt(),)})])

    events = run_stream(monkeypatch, agent)

    assert ("ask_user", {"question": "要删哪个文件？", "interrupt_id": "i-9"}) in events
    name, data = events[-1]
    assert name == "done"
    assert data["awaiting_user"] is True
    assert data["content"] == "", "中断时本轮没有最终正文"


def test_graph_exception_becomes_error_event(monkeypatch):
    agent = FakeAgent([("messages", (AIMessageChunk(content="半句"), {}))], error=RuntimeError("图炸了"))

    events = run_stream(monkeypatch, agent)
    names = [name for name, _ in events]

    assert names[-1] == "error", "图内异常要转成 error 事件，而不是直接断流"
    assert "图炸了" in events[-1][1]["content"]
    assert "done" not in names


def test_heartbeat_ping_while_waiting_for_graph(monkeypatch):
    monkeypatch.setattr(chat, "HEARTBEAT_INTERVAL", 0.01)

    class SlowAgent(FakeAgent):
        async def _iterate(self):
            await asyncio.sleep(0.05)
            yield ("messages", (AIMessageChunk(content="慢"), {}))

    events = run_stream(monkeypatch, SlowAgent([]))
    names = [name for name, _ in events]

    assert "ping" in names, "长时间无数据时必须发心跳，否则会被代理掐断"
    assert names[-1] == "done"


class HangingAgent(FakeAgent):
    """图一直不返回（模拟正在跑的长耗时工具）。"""

    async def _iterate(self):
        await asyncio.sleep(5)
        yield ("messages", (AIMessageChunk(content="late"), {}))


def test_disconnect_skips_done_and_requests_tool_cancellation(monkeypatch):
    """断开后：不发 done，并且要请求 MCP 服务取消仍在执行的工具。"""
    monkeypatch.setattr(chat, "start_recording", lambda: {"c-shell": "run_shell_command"})

    cancel_calls: list[dict[str, str]] = []

    async def fake_cancel_running_tasks(calls):
        cancel_calls.append(dict(calls))
        return {}

    monkeypatch.setattr(chat, "cancel_running_tasks", fake_cancel_running_tasks)

    events = run_stream_then_cancel(monkeypatch, HangingAgent([]))
    names = [name for name, _ in events]

    assert "done" not in names, "客户端已断开，不应再发 done"
    assert cancel_calls == [{"c-shell": "run_shell_command"}]


def test_cancel_failure_does_not_escape(monkeypatch):
    """取消请求本身失败（MCP 服务正好挂了）不能把异常抛出来影响响应收尾。"""
    monkeypatch.setattr(chat, "start_recording", lambda: {"c1": "execute_gui_task"})

    async def boom(calls):
        raise RuntimeError("MCP 不可达")

    monkeypatch.setattr(chat, "cancel_running_tasks", boom)

    events = run_stream_then_cancel(monkeypatch, HangingAgent([]))

    assert [name for name, _ in events] == []
