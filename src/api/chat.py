"""对话接口：SSE 流式输出 + 会话历史管理。

执行模型（Route A 之后）：整条链路真异步——图用 agent.astream 跑在 FastAPI 的事件
循环里，检查点读写走异步 psycopg。Windows 上这要求 SelectorEventLoop（见 src/main.py）。

取消：
- 客户端断开 → 用 asyncio 原生取消中断 agent.astream（真正中断图执行，不再依赖
  "检查标志位后主动退出"）；
- 但长耗时工具跑在 MCP 服务的**工作线程**里，线程无法被取消，所以额外调用服务内的
  cancel_task 做协作式取消（见 main_graph.cancel_running_tasks）。
"""

import asyncio
import contextlib
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command
from psycopg.rows import dict_row

from src.agents.checkpoint import (
    CONSTRAINT_NAMESPACE,
    get_async_checkpointer,
    get_async_pool,
    get_async_store,
)
from src.agents.graph.main_graph import cancel_running_tasks, get_main_agent
from src.agents.mcps.resource_client import load_reference_text
from src.agents.mcps.tool_registry import start_recording
from src.agents.mcps.upload_store import is_image, name_from_uri
from src.api.schemas import ChatRequest
from src.api.sse import sse_event, sse_ping

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])

DEFAULT_SYSTEM_PROMPT = "你是智能助手，你可以调用工具来帮助用户完成任务。如果可以使用其他工具完成任务，尽量不使用GUI工具,你有一个知识库工具：search_knowledge(query, top_k)。当用户询问规章制度、公司政策、产品说明等需要查阅文档的问题时，请调用该工具获取相关信息。当缺少必要信息或需要用户确认时，调用 ask_user 工具向用户提问，不要凭空猜测。"

# 资料类记忆不随系统提示词注入，需要显式引导模型先检索
USER_INFO_HINT = (
    "回答涉及用户自身的身份、背景、经历、偏好等资料时，"
    "先调用 get_store_by_embedding 查询资料记忆，不要凭空猜测。"
)

HEARTBEAT_INTERVAL = 15  # 心跳间隔（秒）

# asearch 的 limit 默认只有 10，不显式给会静默截断约束类记忆
CONSTRAINT_INJECT_LIMIT = 50

# langgraph 动态中断在 updates 流中的保留键
INTERRUPT_KEY = "__interrupt__"


class _ThreadGate:
    """按 thread_id 串行化同一会话的请求。

    LangGraph 的检查点没有乐观锁：同一 thread 并发写会互相覆盖/交错（前端虽然是
    单飞，但多标签页、重试请求都可能撞上）。这里用"每会话一把锁 + 引用计数"实现，
    计数归零才回收锁对象（回收必须在无人等待时做，否则等待方会持有旧锁、新请求拿到
    新锁，串行化失效）。
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._holders: dict[str, int] = {}

    @asynccontextmanager
    async def hold(self, thread_id: str) -> AsyncIterator[None]:
        lock = self._locks.setdefault(thread_id, asyncio.Lock())
        self._holders[thread_id] = self._holders.get(thread_id, 0) + 1
        try:
            async with lock:
                yield
        finally:
            self._holders[thread_id] -= 1
            if self._holders[thread_id] <= 0:
                self._holders.pop(thread_id, None)
                self._locks.pop(thread_id, None)


_THREAD_GATE = _ThreadGate()

# 后台任务要持有引用，否则可能被 GC 回收
_BACKGROUND_TASKS: set[asyncio.Task] = set()


def _spawn_cleanup(coro) -> None:
    """启动一个脱离当前取消作用域的后台任务。

    为什么不用 await：客户端断开时 Starlette 通过 anyio 的取消作用域取消本协程
    （uvicorn 声明 ASGI spec_version 2.3 < 2.4，所以走 listen_for_disconnect 分支），
    而 anyio 的取消是 level-triggered——作用域内**任何** await 都会被立即再次取消，
    直接 await 的清理动作等于没做。新建的 asyncio 任务不属于那个作用域，能正常跑完。
    """
    task = asyncio.create_task(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


async def _report_cancel(calls: dict[str, str]) -> None:
    try:
        summary = await cancel_running_tasks(list(calls.items()))
        if summary:
            logger.info("客户端断开，已请求取消工具任务：%s", summary)
    except Exception:
        logger.exception("取消 MCP 工具任务失败")


def _interrupt_question(intr) -> str:
    """从 Interrupt 对象中提取要展示给用户的问题文本。"""
    value = getattr(intr, "value", None)
    if isinstance(value, dict):
        return str(value.get("question") or value)
    return value if isinstance(value, str) else str(value)


def _interrupts_of(snapshot) -> tuple:
    """取状态快照中的中断对象，无中断返回空元组。

    不能用 snapshot.next 判断：next 表示"还有未执行的任务"，它既包含被 ask_user 中断的任务，
    也包含"执行到一半被放弃"的会话（客户端断开、进程退出）——后者没有中断对象。
    用 next 判断会把这类未完成的会话误报成"正在等待你的回答"。
    """
    if not snapshot:
        return ()
    ints = tuple(getattr(snapshot, "interrupts", ()) or ())
    if ints:
        return ints
    # 兜底：个别版本仅把中断挂在 tasks 上
    return tuple(
        i
        for task in (getattr(snapshot, "tasks", ()) or ())
        for i in (getattr(task, "interrupts", ()) or ())
    )


def _split_uris(uris: list[str] | None) -> tuple[list[str], list[str]]:
    """把本轮附件 URI 按文本/图片分流。

    图片必须走多模态消息：把它拼进参考资料文本，模型只会看到一串 base64。
    """
    text_uris: list[str] = []
    image_uris: list[str] = []
    for uri in uris or []:
        # 用解码后的文件名判断，与 _image_blocks 保持同一口径
        if is_image(name_from_uri(uri)):
            image_uris.append(uri)
        else:
            text_uris.append(uri)
    return text_uris, image_uris


async def _load_constraints() -> str:
    """读出约束类长期记忆，拼成可注入系统提示词的文本块。

    约束类是"要求一直遵守"的规则（称呼、语言、回复风格、禁忌等），会话开始时整体
    注入一次即可——那条 system message 会随每轮请求发给模型，摘要触发后也仍保留在
    summarized_messages[0]（SummarizationNode 不摘 system）。读取不带 query → 走
    namespace 前缀索引，成本与记忆总条数无关。
    读取失败只记日志：少一段约束不该让整轮对话失败。
    """
    try:
        items = await get_async_store().asearch(
            CONSTRAINT_NAMESPACE, limit=CONSTRAINT_INJECT_LIMIT
        )
    except Exception:  # noqa: BLE001 - 约束读不到不该让对话失败
        logger.exception("读取约束类长期记忆失败，本轮不注入约束")
        return ""

    lines = []
    for item in items:
        content = str((item.value or {}).get("value", "")).strip()
        if content:
            lines.append(f"- {content}")
    return "\n".join(lines)


async def _build_input(req: ChatRequest, snapshot) -> dict:
    """构造本轮图输入。"""
    # 首次进入会话时注入系统提示词（后续轮次沿用检查点中的历史消息）
    has_history = bool(snapshot and snapshot.values.get("messages"))
    messages = []
    if not has_history:
        # 约束类记忆随系统提示词注入一次；资料类不注入，需引导模型自行检索
        system_prompt = req.system_prompt or DEFAULT_SYSTEM_PROMPT
        system_prompt = f"{system_prompt}\n\n{USER_INFO_HINT}"
        constraints = await _load_constraints()
        if constraints:
            system_prompt = f"{system_prompt}\n\n<constraints>\n{constraints}\n</constraints>"
        messages.append(SystemMessage(content=system_prompt))

    # 附件作为本轮输入注入：文本走 MCP resource，图片只留文件名引用
    # 用 HumanMessage 而不是新增 SystemMessage：多个 system 消息会被部分模型拒绝
    text_uris, image_uris = _split_uris(req.resource_uris)
    reference = await load_reference_text(text_uris)
    if reference:
        messages.append(HumanMessage(content=reference))

    image_names = [name_from_uri(uri) for uri in image_uris]
    if image_names:
        content: list = [{"type": "image_ref", "name": name} for name in image_names]
        content.append({"type": "text", "text": req.message})
        messages.append(HumanMessage(content=content))
    else:
        messages.append(HumanMessage(content=req.message))

    return {"messages": messages, "task_completed": False}


async def _stream_events(req: ChatRequest) -> AsyncIterator[str]:
    """SSE 事件流：token / tool_call / tool_result / ask_user / done / error + 心跳。

    客户端断开由 Starlette 负责：它会取消本协程，CancelledError 从消费循环的 await
    处抛出、直接进入 finally 收尾（先快照在飞工具，再发起取消请求），所以这里既不
    需要 request 对象，也不需要自己轮询 is_disconnected()。
    """
    thread_id = req.thread_id or uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}
    agent = get_main_agent()

    async with _THREAD_GATE.hold(thread_id):
        snapshot = await agent.aget_state(config)
        # resume=True 且图停在 ask_user 时，本条消息就是用户回答
        if req.resume and _interrupts_of(snapshot):
            graph_input: Any = Command(resume=req.message)
        else:
            graph_input = await _build_input(req, snapshot)

        # 本轮在跑的工具调用（ContextVar，请求间隔离）
        recording = start_recording()
        pending: asyncio.Future | None = None

        # messages 流里只有 AIMessageChunk 是正文，完整 AIMessage 与 ToolMessage 跳过
        emitted_token = False
        last_full_content = ""

        stream = agent.astream(
            graph_input, config=config, stream_mode=["messages", "updates"]
        )

        # 把 StopAsyncIteration 与图内异常收敛成返回值：前者逃出会变成 RuntimeError，客户端表现为响应提前结束
        async def _next_event() -> tuple[str, Any]:
            try:
                return "event", await stream.__anext__()
            except StopAsyncIteration:
                return "end", None
            except Exception as exc:  # noqa: BLE001 - 收敛成 error 事件
                return "error", exc

        try:
            awaiting_user = False
            while True:
                # 包 Future 让超时只作用于这一次等待，超时后退化成发心跳
                pending = asyncio.ensure_future(_next_event())
                while not pending.done():
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(pending), timeout=HEARTBEAT_INTERVAL
                        )
                    except asyncio.TimeoutError:
                        # 定期发注释行保活，避免被代理/网关掐断
                        yield sse_ping()
                tag, value = pending.result()
                if tag == "end":
                    break
                if tag == "error":
                    yield sse_event("error", {"content": f"{type(value).__name__}: {value}"})
                    return
                kind, payload = value

                if kind == "messages":
                    msg_chunk, _meta = payload
                    if isinstance(msg_chunk, ToolMessage):
                        continue
                    content = msg_chunk.content
                    if not (isinstance(content, str) and content):
                        continue
                    if isinstance(msg_chunk, AIMessageChunk):
                        emitted_token = True
                        yield sse_event("token", {"content": content})
                    else:
                        # 完整 AIMessage：与流式块重复，不发出，仅作为非流式模型的兜底
                        last_full_content = content
                elif kind == "updates":
                    # 动态中断以 "__interrupt__" 键出现在 updates 中，值是 Interrupt 元组
                    interrupts = payload.get(INTERRUPT_KEY) or ()
                    if interrupts:
                        awaiting_user = True
                        for intr in interrupts:
                            yield sse_event("ask_user", {
                                "question": _interrupt_question(intr),
                                "interrupt_id": getattr(intr, "id", ""),
                            })
                    for _node, update in payload.items():
                        if not isinstance(update, dict):
                            continue
                        for msg in update.get("messages", []):
                            if isinstance(msg, AIMessage) and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    yield sse_event("tool_call", {
                                        "name": tc["name"],
                                        "args": tc["args"],
                                        "id": tc["id"],
                                    })
                            elif isinstance(msg, ToolMessage):
                                yield sse_event("tool_result", {
                                    "name": getattr(msg, "name", ""),
                                    "content": msg.content,
                                    "tool_call_id": getattr(msg, "tool_call_id", ""),
                                })

            if awaiting_user:
                yield sse_event("done", {
                    "task_completed": False,
                    "awaiting_user": True,
                    "content": "",
                    "thread_id": thread_id,
                })
                return

            # 非流式模型没有流式块，用完整内容兜底
            if not emitted_token and last_full_content:
                yield sse_event("token", {"content": last_full_content})

            final = await agent.aget_state(config)
            final_messages = (final.values.get("messages") or []) if final else []
            last = final_messages[-1] if final_messages else None
            yield sse_event("done", {
                "task_completed": bool(final.values.get("task_completed", False)) if final else False,
                "awaiting_user": False,
                "content": getattr(last, "content", "") if last else "",
                "thread_id": thread_id,
            })
        finally:
            # 顺序不能换：先取快照再取消，取消会注销运行中工具的登记
            calls = dict(recording)

            # shield 让 pending 不受本协程取消的影响，所以要显式取消它
            if pending is not None and not pending.done():
                pending.cancel()
            if calls:
                # 客户端断开：请求 MCP 服务协作式取消长耗时任务
                _spawn_cleanup(_report_cancel(calls))
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()


@router.post("/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        _stream_events(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用代理缓冲，保证流式实时性
        },
    )


async def _state_messages(thread_id: str) -> list:
    """读取指定会话检查点中的 messages。

    此版本 CheckpointTuple 无 .value 属性；checkpoint 的 channel_values 中
    messages 已是反序列化后的 BaseMessage 对象，直接使用。
    """
    tup = await get_async_checkpointer().aget_tuple(
        {"configurable": {"thread_id": thread_id}}
    )
    if not tup:
        return []
    return (tup.checkpoint.get("channel_values") or {}).get("messages", [])


def _uuid7_time(checkpoint_id: str) -> str | None:
    """从 uuid7 格式的 checkpoint_id 提取时间（前 48 位为毫秒时间戳）。"""
    try:
        ms = int(checkpoint_id.replace("-", "")[:12], 16)
        return datetime.fromtimestamp(ms / 1000).astimezone().isoformat()
    except (ValueError, OSError, OverflowError):
        return None


def _image_names_of(content) -> list[str]:
    """取出消息 content 里的图片引用文件名（供前端按名加载图片实体）。"""
    if not isinstance(content, list):
        return []
    names: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image_ref":
            names.append(str(block.get("name", "")))
    return names


def _content_to_text(content) -> str:
    """把消息 content 压成纯文本。

    多模态消息的 content 是 content block 列表，直接 str() 会得到一段 Python repr
    （前端只能看到 ``[{'type': 'text', ...}]`` 这样的噪声），因此按块提取文本；
    图片块本身不产出文字，由前端依据 images 字段单独渲染。
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(part for part in parts if part)


def _serialize_message(msg) -> dict | None:
    """将检查点中的消息序列化为可 JSON 化结构，供前端渲染历史对话。"""
    content = _content_to_text(msg.content)
    if isinstance(msg, HumanMessage):
        # 图片只回传文件名，实体按需从 attachments 接口取
        images = _image_names_of(msg.content)
        return {"role": "user", "content": content, "images": images or None}
    if isinstance(msg, AIMessage):
        tool_calls = [
            {"id": tc.get("id", ""), "name": tc.get("name", ""), "args": tc.get("args", {})}
            for tc in (msg.tool_calls or [])
        ]
        return {"role": "assistant", "content": content, "tool_calls": tool_calls or None}
    if isinstance(msg, ToolMessage):
        return {
            "role": "tool",
            "content": content,
            "name": getattr(msg, "name", "") or None,
            "tool_call_id": getattr(msg, "tool_call_id", None),
        }
    return None


async def _pending_interrupt(thread_id: str) -> dict | None:
    """会话若确实停在 ask_user 的中断点，返回待回答的问题，否则 None。

    用于重新打开会话时恢复"等待回答"状态（否则该会话既无法继续也无法回答）。
    只有真正存在中断对象（snapshot.interrupts）才算等待回答：
    未跑完就中断的会话（客户端断开、进程退出）虽然 next 非空，但并不在等用户，
    误报会让前端一直提示"正在等待你的回答"。
    """
    try:
        snapshot = await get_main_agent().aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
    except Exception:
        logger.exception("读取会话中断状态失败")
        return None
    ints = _interrupts_of(snapshot)
    if not ints:
        return None
    intr = ints[0]
    return {
        "question": _interrupt_question(intr),
        "interrupt_id": getattr(intr, "id", ""),
    }


@router.get("/threads")
async def list_threads(limit: int = 50):
    """列出所有会话线程，按最近活跃时间倒序，附首条用户消息预览。

    注意：这里对每个会话都单独读一次检查点（N+1 查询），会话多时值得改成一条
    聚合 SQL（group by thread_id + 取首条 HumanMessage）。
    """
    saver = get_async_checkpointer()
    pool = get_async_pool()
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT thread_id, max(checkpoint_id) AS last_checkpoint "
                "FROM checkpoints WHERE checkpoint_ns = '' "
                "GROUP BY thread_id ORDER BY last_checkpoint DESC LIMIT %s",
                (limit,),
            )
            rows = await cur.fetchall()

    result = []
    for row in rows:
        tid = row["thread_id"]
        info = {
            "thread_id": tid,
            "last_active": None,
            "preview": "",
            "message_count": 0,
        }
        try:
            tup = await saver.aget_tuple({"configurable": {"thread_id": tid}})
            if tup:
                info["last_active"] = (
                        tup.checkpoint.get("ts")  # 检查点写入时间（ISO 字符串）
                        or (tup.metadata or {}).get("timestamp")
                        or _uuid7_time(row["last_checkpoint"])
                )
                msgs = await _state_messages(tid)
                info["message_count"] = len(msgs)
                for m in msgs:
                    if isinstance(m, HumanMessage):
                        info["preview"] = _content_to_text(m.content)[:60]
                        break
        except Exception:
            pass  # 单个会话加载失败不影响列表输出
        result.append(info)
    return result


@router.get("/threads/{thread_id}/messages")
async def get_thread_messages(thread_id: str):
    """查询指定会话的全部历史消息（已序列化），并附带是否停在待回答的中断点。"""
    saver = get_async_checkpointer()
    tup = await saver.aget_tuple({"configurable": {"thread_id": thread_id}})
    if not tup:
        raise HTTPException(status_code=404, detail="会话不存在")
    items = []
    for m in (tup.checkpoint.get("channel_values") or {}).get("messages", []):
        item = _serialize_message(m)
        if item:
            items.append(item)
    return {
        "thread_id": thread_id,
        "messages": items,
        "interrupt": await _pending_interrupt(thread_id),
    }


@router.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """删除指定会话的全部检查点。"""
    await get_async_checkpointer().adelete_thread(thread_id)
    return {"deleted": thread_id}
