"""对话接口里的纯函数：消息序列化、中断判定、会话串行化。

这些函数看着零碎，但每一个都对应一个踩过的坑（例如"用 snapshot.next 判断等待回答
会把未跑完的会话误报成在等用户"）。
"""

import asyncio
from datetime import datetime

import pytest
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.api import chat
from src.asyncio_runner import run_async


class _Snapshot:
    """极简状态快照：只提供被测函数会读到的字段。"""

    def __init__(self, messages=(), interrupts=(), tasks=()):
        self.values = {"messages": list(messages)}
        self.interrupts = tuple(interrupts)
        self.tasks = tuple(tasks)


class _Interrupt:
    def __init__(self, value, id="i-1"):
        self.value = value
        self.id = id


def test_split_uris_routes_text_and_image():
    text, image = chat._split_uris([
        "upload://note.txt", "upload://%E5%9B%BE.png", "upload://code.py",
    ])
    assert text == ["upload://note.txt", "upload://code.py"]
    assert image == ["upload://%E5%9B%BE.png"]


def test_split_uris_handles_none_and_empty():
    assert chat._split_uris(None) == ([], [])
    assert chat._split_uris([]) == ([], [])


def test_interrupts_of_none_snapshot():
    assert chat._interrupts_of(None) == ()


def test_interrupts_of_prefers_snapshot_interrupts():
    intr = _Interrupt({"question": "要删哪个？"})
    assert chat._interrupts_of(_Snapshot(interrupts=[intr])) == (intr,)


def test_interrupts_of_ignores_unfinished_session():
    """next 非空只说明"还有任务没跑完"，不代表正在等用户回答。

    客户端断开、进程被杀都会留下这种会话；把它当成"等待回答"会让前端一直提示
    用户回答一个并不存在的问题。
    """

    class Unfinished(_Snapshot):
        next = ("main_model_call",)

    assert chat._interrupts_of(Unfinished()) == ()


def test_interrupts_of_falls_back_to_tasks():
    """个别版本只把中断挂在 tasks 上，需要兜底。"""
    intr = _Interrupt("问题")

    class Task:
        interrupts = (intr,)

    class Snapshot:
        interrupts = ()
        tasks = (Task(),)

    assert chat._interrupts_of(Snapshot()) == (intr,)


def test_interrupt_question_extracts_text():
    assert chat._interrupt_question(_Interrupt({"question": "选哪个"})) == "选哪个"
    assert chat._interrupt_question(_Interrupt("直接是文本")) == "直接是文本"
    assert "other" in chat._interrupt_question(_Interrupt({"other": 1}))


def test_content_to_text_flattens_multimodal_blocks():
    content = [
        {"type": "text", "text": "第一部分"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "text", "text": "第二部分"},
    ]
    assert chat._content_to_text(content) == "第一部分\n第二部分"
    assert chat._content_to_text("直接是字符串") == "直接是字符串"


def test_content_to_text_of_image_only_message_is_empty():
    assert chat._content_to_text([{"type": "image_url", "image_url": {}}]) == ""


def test_image_names_of_collects_refs_only():
    content = [{"type": "image_ref", "name": "a.png"}, {"type": "text", "text": "x"}]
    assert chat._image_names_of(content) == ["a.png"]
    assert chat._image_names_of("纯文本") == []


def test_serialize_message_user_with_images():
    msg = HumanMessage(content=[
        {"type": "image_ref", "name": "a.png"},
        {"type": "text", "text": "看看这张图"},
    ])
    assert chat._serialize_message(msg) == {
        "role": "user", "content": "看看这张图", "images": ["a.png"],
    }


def test_serialize_message_user_without_images_uses_none():
    assert chat._serialize_message(HumanMessage(content="你好")) == {
        "role": "user", "content": "你好", "images": None,
    }


def test_serialize_message_assistant_with_tool_calls():
    msg = AIMessage(content="", tool_calls=[
        {"name": "read_file", "args": {"path": "a.txt"}, "id": "c1"},
    ])
    assert chat._serialize_message(msg) == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "name": "read_file", "args": {"path": "a.txt"}}],
    }


def test_serialize_message_tool_keeps_call_id():
    msg = ToolMessage(content="内容", tool_call_id="c1", name="read_file")
    assert chat._serialize_message(msg) == {
        "role": "tool", "content": "内容", "name": "read_file", "tool_call_id": "c1",
    }


def test_serialize_message_skips_system_message():
    assert chat._serialize_message(SystemMessage(content="你是助手")) is None


def test_uuid7_checkpoint_id_yields_its_timestamp():
    millis = 1_700_000_000_000
    checkpoint_id = f"{millis:012x}-1a2b-7c3d-8e4f-0123456789ab"

    parsed = chat._uuid7_time(checkpoint_id)

    assert parsed is not None
    expected = datetime.fromtimestamp(millis / 1000).astimezone()
    assert datetime.fromisoformat(parsed) == expected


@pytest.mark.parametrize("bad", ["", "not-hex-at-all!", "zzzz-1a2b"])
def test_uuid7_checkpoint_id_invalid_returns_none(bad):
    assert chat._uuid7_time(bad) is None


def test_thread_gate_serializes_same_thread():
    """同一会话的请求必须串行：检查点没有乐观锁，并发写会互相覆盖。"""
    order: list[str] = []

    async def job(tag: str, delay: float) -> None:
        async with chat._THREAD_GATE.hold("tid"):
            order.append(f"{tag}-in")
            await asyncio.sleep(delay)
            order.append(f"{tag}-out")

    async def main():
        await asyncio.gather(job("a", 0.02), job("b", 0.0))

    run_async(main())

    assert order == ["a-in", "a-out", "b-in", "b-out"]


def test_thread_gate_allows_parallel_across_threads():
    order: list[str] = []

    async def job(tid: str) -> None:
        async with chat._THREAD_GATE.hold(tid):
            order.append(f"{tid}-in")
            await asyncio.sleep(0.02)
            order.append(f"{tid}-out")

    async def main():
        await asyncio.gather(job("t1"), job("t2"))

    run_async(main())

    # 两个会话互不阻塞
    assert order == ["t1-in", "t2-in", "t1-out", "t2-out"]


def test_thread_gate_recycles_lock_after_use():
    """锁必须在无人等待时回收，否则长跑服务里会无限增长。"""
    gate = chat._ThreadGate()

    async def main():
        async with gate.hold("t-x"):
            pass
        return dict(gate._locks), dict(gate._holders)

    locks, holders = run_async(main())

    assert locks == {}
    assert holders == {}
