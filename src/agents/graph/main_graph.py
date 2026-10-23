"""主对话 agent：LLM + MCP 工具循环。

执行模型（Route A 之后）：
- MCP 是常驻 HTTP 服务，本模块只是 HTTP 客户端 → 不需要创建子进程，节点可以全部写成
  async，直接在 FastAPI 的事件循环里 await，不再占用线程池；
- 检查点/Store 用异步版本（见 checkpoint.postgresql_saver）；
- 图与工具在应用启动时（lifespan 调用 init_agent）一次性初始化，运行期只读全局字典。

取消：客户端断开时事件循环会取消正在等待的 astream，但**长耗时工具跑在 MCP 服务的
工作线程里，线程无法被取消**，所以额外提供 cancel_running_tasks() 做协作式取消。
"""

import asyncio
import base64
import logging
import os
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.messages import AnyMessage, AIMessage, HumanMessage, ToolMessage
from langchain.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from typing_extensions import Annotated, TypedDict

from src.agents.checkpoint import get_async_checkpointer, get_async_store
from src.agents.mcps.registry import CANCEL_TOOL_NAME, SERVERS, url_of
from src.agents.mcps.tool_registry import (
    register_running_tool,
    unregister_running_tool,
)
from src.agents.mcps.upload_store import guess_media_type, read_upload_bytes
from src.agents.models import main_model, main_model_breaker
from src.asyncio_runner import run_async
from src.rag import query_rewrite
from langmem.short_term import SummarizationNode

logger = logging.getLogger(__name__)

# GUI 任务常常跑几分钟，而 streamable HTTP 默认超时只有 30s，必须放大
MCP_TOOL_TIMEOUT = float(os.getenv("MCP_TOOL_TIMEOUT", "3600"))

mcp_client = MultiServerMCPClient({
    name: {
        "transport": "streamable_http",
        "url": url_of(name),
        "timeout": MCP_TOOL_TIMEOUT,
        "sse_read_timeout": MCP_TOOL_TIMEOUT,
    }
    for name in SERVERS
})

# 支持协作式取消的工具在执行前会注入 tool_call_id 作为 task_id
CANCELLABLE_TOOLS = {"execute_gui_task", "run_shell_command", "run_python_code"}

# 知识库检索前先把 query 改写成可独立检索的问题
RAG_TOOL_NAME = "search_knowledge"

# 查询改写开关（RAG_REWRITE=0 关闭）与渲染历史保留的条数
RAG_REWRITE_ENABLED = os.getenv("RAG_REWRITE", "1") != "0"
RAG_REWRITE_HISTORY_MESSAGES = int(os.getenv("RAG_REWRITE_HISTORY_MESSAGES", "6"))

#   HISTORY_SUMMARY=0                    关闭（图中不挂摘要节点）
#   HISTORY_SUMMARY_MAX_TOKENS           摘要后给模型的历史预算（token）
#   HISTORY_SUMMARY_BEFORE_TOKENS        触发摘要的阈值（默认等于上面的预算）
#   HISTORY_SUMMARY_MAX_SUMMARY_TOKENS   摘要文本自身的上限，必须小于历史预算
HISTORY_SUMMARY_ENABLED = os.getenv("HISTORY_SUMMARY", "1") != "0"
HISTORY_SUMMARY_MAX_TOKENS = int(os.getenv("HISTORY_SUMMARY_MAX_TOKENS", "8000"))
HISTORY_SUMMARY_BEFORE_TOKENS = int(
    os.getenv("HISTORY_SUMMARY_BEFORE_TOKENS", str(HISTORY_SUMMARY_MAX_TOKENS))
)
# main_model 是推理模型，这个上限同时卡住思维链，给太小摘要会变成空壳
HISTORY_SUMMARY_MAX_SUMMARY_TOKENS = int(
    os.getenv("HISTORY_SUMMARY_MAX_SUMMARY_TOKENS", "2000")
)


@tool
def ask_user(question: str) -> str:
    """
    当你需要向用户提问以获取必要信息时调用此工具。
    例如：需要用户确认操作、补充缺失的参数、选择选项等。

    参数:
        question: 你要向用户提出的问题
    """
    # 这个工具不真正执行，只是信号，逻辑在 ask_user_question 节点
    return ""


@tool
def view_image(name: str) -> list:
    """查看此前上传过的图片。

    当用户针对某张历史图片追问细节，而该图在当前上下文中只以
    ［图片：文件名］占位标记出现时（原图已移出上下文以节省空间），
    调用本工具按文件名把图片重新读回来查看。

    参数:
        name: 图片文件名，取自占位标记 ［图片：文件名］
    """
    # 只声明引用，实体由 _resolve_images 在发请求前加载
    return [
        {"type": "text", "text": f"以下是图片 {name} 的内容："},
        {"type": "image_ref", "name": name},
    ]


def _load_image_block(name: str) -> dict:
    """把图片文件名读成多模态内容块；读取失败时返回一句文字说明。

    不抛异常：单张图缺失不该让整轮对话失败，交给模型一句说明更合适。
    """
    data = read_upload_bytes(name)
    if not data:
        return {"type": "text", "text": f"［图片加载失败：{name}（文件不存在或已被清理）］"}
    encoded = base64.b64encode(data).decode()
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{guess_media_type(name)};base64,{encoded}"},
    }


def _stale_image_note(name: str) -> dict:
    """历史轮次里的图片引用降级成的文字占位。"""
    if name:
        label = f"［图片：{name}］"
    else:
        label = "［图片］"
    return {
        "type": "text",
        "text": f"{label}（已移出上下文，可用 view_image 工具重新查看）",
    }


def last_human_index(messages: list[AnyMessage]) -> int:
    """最后一条 HumanMessage 的下标（没有则 -1）。

    "本轮"的唯一定义处：图片加载（_resolve_images）与摘要后补回当前提问
    （build_summarize_node）都按它划分，避免两处口径漂移。
    """
    last_human = -1
    for idx, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            last_human = idx
    return last_human


def _resolve_images(messages: list[AnyMessage]) -> list[AnyMessage]:
    """把消息里的图片引用按需加载成真实图片，历史轮次的引用降级成文字占位。

    检查点里只存 {"type": "image_ref", "name": ...}，图片实体到发请求前才从 uploads
    目录读出并编码，所以 state 与数据库里不会长期压着 MB 级的 base64（一张 1.5MB 的
    图编码后是 200 万字符量级）。只加载"本轮"用到的图，更早的降级成文字，并提示模型
    可用 view_image 重新取回。

    "本轮"以最后一条 HumanMessage 为界：它本身（用户刚发的图）和它之后的消息
    （模型本轮用 view_image 取回的图）都加载，更早的一律降级。
    resume 续答时最后一条 HumanMessage 仍是那条带图的消息，加载它正合语义——用户
    多半还在说同一张图。

    本函数是纯 CPU/文件 IO（base64 编码 + 读盘），调用方要放到线程池里执行，
    避免阻塞事件循环。
    """
    last_human = last_human_index(messages)

    result: list[AnyMessage] = []
    for index, message in enumerate(messages):
        if not isinstance(message.content, list):
            result.append(message)
            continue

        converted: list = []
        has_ref = False
        for block in message.content:
            if not (isinstance(block, dict) and block.get("type") == "image_ref"):
                converted.append(block)
                continue
            has_ref = True
            name = str(block.get("name", ""))
            if index >= last_human:
                converted.append(_load_image_block(name))
            else:
                converted.append(_stale_image_note(name))

        if has_ref:
            result.append(message.model_copy(update={"content": converted}))
        else:
            result.append(message)
    return result


def replace_messages_with_ids(
    _left: list[AnyMessage], right: list[AnyMessage]
) -> list[AnyMessage]:
    """summarized_messages 通道的 reducer：整体替换 + 给缺 id 的消息补 uuid。

    两件事的动机不同：

    - **整体替换**（忽略 left、返回 right）是**必须**的：SummarizationNode 每轮返回的是
      *完整的*"给模型看的历史"（摘要 + 近期若干条），不是增量。若沿用 messages 那套
      add_messages 累加语义，这份历史会一轮一轮叠上去，摘要就白做了。
    - **补 id** 是**防御性**的：SummarizationNode 的摘要是新建的 SystemMessage、没有 id，
      但不补也不会报错 —— 真正要求消息带 id 的 _preprocess_messages，其抛
      ValueError("Messages are required to have ID field") 的位置读的是 messages 通道
      （那里的 id 由 add_messages 负责）。这里补上只是让通道内每条消息都自带身份，
      避免将来有代码按 id 比较/去重时踩空。
    """
    for message in right or []:
        if hasattr(message, "id") and message.id is None:
            message.id = str(uuid.uuid4())
    return right


class MainMessagesState(TypedDict):
    # 完整转录供前端历史接口与图片引用使用，不因摘要裁剪
    # 用 add_messages：它会补 uuid，SummarizationNode 要求每条消息都有 id
    messages: Annotated[list[AnyMessage], add_messages]
    # 发给模型的历史：摘要 + 近期若干条；未触发摘要时为空，回退到 messages
    summarized_messages: Annotated[list[AnyMessage], replace_messages_with_ids]
    context: dict[str, Any]
    task_completed: bool
    pending_question: str  # 待向用户提问的内容
    pending_tool_call_id: str  # 待向用户提问的工具调用 ID


_llm_with_tools: Any = None
_tool_map: dict[str, Any] = {}
_tool_server: dict[str, str] = {}  # 工具名 -> 所属 MCP 服务，用于取消时定位服务
_agent: CompiledStateGraph | None = None


async def init_agent() -> None:
    """连接 MCP 服务、绑定工具、编译图（由 FastAPI lifespan 调用一次）。"""
    global _llm_with_tools, _tool_map, _tool_server, _agent
    if _agent is not None:
        return

    tools: list[Any] = []
    tool_server: dict[str, str] = {}
    # 记录工具所属的服务，取消长耗时任务时按服务名发起请求
    for name in SERVERS:
        server_tools = await mcp_client.get_tools(server_name=name)
        for server_tool in server_tools:
            tool_server[server_tool.name] = name
        # cancel_task 是内部工具，不暴露给模型
        tools.extend(t for t in server_tools if t.name != CANCEL_TOOL_NAME)
    tools.extend([ask_user, view_image])

    _tool_map = {t.name: t for t in tools}
    _tool_server = tool_server
    _llm_with_tools = main_model.bind_tools(tools)
    _agent = agent_builder.compile(
        checkpointer=get_async_checkpointer(),
        store=get_async_store(),
    )
    logger.info("主 agent 初始化完成：%d 个工具来自 %d 个 MCP 服务", len(tools), len(SERVERS))


def get_main_agent() -> CompiledStateGraph:
    """返回已编译的 main_agent（必须先经过 init_agent）。"""
    if _agent is None:
        raise RuntimeError("agent 未初始化：请确认 FastAPI lifespan 已执行 init_agent()")
    return _agent


def get_llm_and_tools() -> tuple[Any, dict[str, Any]]:
    """返回 (绑定工具的模型, 工具名->工具映射)。"""
    if _llm_with_tools is None:
        raise RuntimeError("agent 未初始化：请确认 FastAPI lifespan 已执行 init_agent()")
    return _llm_with_tools, _tool_map


async def cancel_running_tasks(calls: list[tuple[str, str]]) -> dict[str, str]:
    """请求 MCP 服务取消仍在执行的长耗时任务，返回 {tool_call_id: 结果}。

    参数 calls 必须由调用方在**取消之前**取快照传入（见 api/chat.py 中 _stream_events
    的收尾逻辑）：取消会让节点的 finally 立刻注销运行中工具，事后查不到。

    取消是协作式的：HTTP 常驻服务不能被整体杀掉（会连带杀死其它会话的任务），
    所以调用服务内的 cancel_task 置位取消事件，工具在轮询/步骤边界自行终止。
    """
    summary: dict[str, str] = {}
    for call_id, tool_name in calls:
        server = _tool_server.get(tool_name)
        if not server:
            summary[call_id] = "非 MCP 工具，跳过"
            continue
        try:
            async with mcp_client.session(server) as session:
                await session.call_tool(CANCEL_TOOL_NAME, {"task_id": call_id})
            summary[call_id] = f"{server}: 已发送取消"
        except Exception as e:  # noqa: BLE001 - 取消失败不应影响断开清理流程
            summary[call_id] = f"{server}: 取消失败 {e}"
            logger.warning("取消工具任务失败 call_id=%s tool=%s: %s", call_id, tool_name, e)
    return summary


def build_summarize_node(
    model: Any = None,
) -> Callable[[MainMessagesState], Awaitable[dict[str, Any]]] | None:
    """构造对话历史摘要节点；HISTORY_SUMMARY=0 时返回 None（此时图中不挂该节点）。

    为什么要包一层而不是直接把 SummarizationNode 挂进图：它每轮都会把"给模型看的历史"
    整份写进 summarized_messages，**即使这一轮根本没触发摘要**（那只是 messages 的一份
    拷贝），会让检查点在摘要生效前一直存两份历史。这里只在确实生成了摘要（返回值里带
    context）时才落库，否则不产生任何通道更新。

    model 仅供测试注入，默认用 main_model。
    """
    if not HISTORY_SUMMARY_ENABLED:
        logger.info("对话历史摘要已关闭（HISTORY_SUMMARY=0）")
        return None

    summarization = SummarizationNode(
        # 摘要上限也绑到模型上，让服务端按上限生成而不是生成完再截
        model=(model or main_model).bind(max_tokens=HISTORY_SUMMARY_MAX_SUMMARY_TOKENS),
        max_tokens=HISTORY_SUMMARY_MAX_TOKENS,
        max_tokens_before_summary=HISTORY_SUMMARY_BEFORE_TOKENS,
        max_summary_tokens=HISTORY_SUMMARY_MAX_SUMMARY_TOKENS,
        input_messages_key="messages",
        output_messages_key="summarized_messages",
    )

    async def summarize_history(state: MainMessagesState) -> dict[str, Any]:
        update = await summarization.ainvoke(state)
        # 没生成摘要就不写通道，避免每轮把完整历史拷进检查点
        if "context" not in update:
            return {}

        running_summary = (update.get("context") or {}).get("running_summary")
        if running_summary is None or not (running_summary.summary or "").strip():
            # 摘要为空时不要覆盖已有的好摘要，直接放弃本轮更新
            logger.warning(
                "摘要模型返回了空摘要，本轮跳过压缩（可调大 "
                "HISTORY_SUMMARY_MAX_SUMMARY_TOKENS，推理模型的思维链也占用这个额度）"
            )
            return {}

        # 首次触发摘要会连最新一条消息一起折进去，这里把本轮提问补回来
        messages = state["messages"]
        summarized = list(update["summarized_messages"])
        present = {m.id for m in summarized if getattr(m, "id", None)}
        current_turn = messages[last_human_index(messages) :]
        missing = [m for m in current_turn if m.id not in present]
        if missing:
            logger.info("摘要把当前提问一并折入，补回 %d 条消息", len(missing))
            update = {**update, "summarized_messages": summarized + missing}
        return update

    return summarize_history


async def condense_question(question: str, history: list[AnyMessage]) -> str:
    """把带指代的追问改写成可独立检索的问题；不需要改写或改写失败时原样返回。

    为什么在 agent 侧做：``search_knowledge`` 跑在独立的 MCP 服务进程里，只能看到
    传进去的 query 字符串，拿不到对话历史。而模型自己生成的 query 参数经常是
    「那这个要提前几天」这种依赖上文的话，直接拿去嵌入会检索到无关内容。

    代价控制：用 needs_rewrite 做粗筛，只有「有历史 且（含指代 或 提问过短）」时才
    多一次模型往返；判漏的后果只是回到"不改写"的行为，不会比现状更差。
    """
    if not RAG_REWRITE_ENABLED:
        return question

    history_text = query_rewrite.render_history(history, RAG_REWRITE_HISTORY_MESSAGES)
    if not query_rewrite.needs_rewrite(question, history_text):
        return question

    try:
        response = await main_model.ainvoke(
            query_rewrite.build_condense_messages(question, history_text)
        )
    except Exception:  # noqa: BLE001 - 改写失败不该让本轮检索失败
        logger.exception("查询改写失败，改用原始问题检索")
        return question

    condensed = query_rewrite.parse_condensed(
        query_rewrite.text_of(response.content), question
    )
    if condensed != question:
        logger.info("检索查询改写：%r -> %r", question, condensed)
    return condensed


async def main_model_call(state: MainMessagesState, runtime: Runtime) -> Command:
    llm_with_tools, tool_map = get_llm_and_tools()
    # 摘要生效后只发摘要 + 近期消息；完整转录仍在 messages 里
    history: list[AnyMessage] = state.get("summarized_messages") or state["messages"]
    # 图片解码是 CPU/IO，放线程池
    messages = await asyncio.to_thread(_resolve_images, history)
    async with main_model_breaker.guard():
        response = await llm_with_tools.ainvoke(messages)

    if not response.tool_calls:
        return Command(
            update={
                "messages": [AIMessage(content=response.content)],
                "task_completed": True,
            },
            goto=END,
        )

    results: dict[str, Any] = {}
    runnable: list[dict] = []
    deferred: set[str] = set()  # 由中断节点负责补 ToolMessage 的 tool_call_id
    ask_question: str | None = None
    ask_tool_call_id = ""

    for tc in response.tool_calls:
        tool = tool_map.get(tc["name"])
        if tool is not None and tool.name == "ask_user":
            # ask_user 只是信号，ToolMessage 由 ask_user_question 节点补；同轮多次提问只保留第一个
            if ask_question is None:
                ask_question = tc["args"].get("question", "请提供更多信息")
                ask_tool_call_id = tc["id"]
                deferred.add(tc["id"])
                continue
            results[tc["id"]] = "同一轮已向用户提出一个更早的问题，请等待用户回答"
            continue
        if tool is None:
            results[tc["id"]] = f"错误：未找到工具 '{tc['name']}'"
            continue
        runnable.append(tc)

    async def _run_one(tc: dict) -> None:
        """执行单个工具调用；异常收敛成文本结果，不中断本轮其它工具。"""
        args = dict(tc["args"])
        if tc["name"] == RAG_TOOL_NAME:
            # 只改本次调用的参数副本，不动 response.tool_calls
            args["query"] = await condense_question(
                str(args.get("query", "")), messages
            )
        if tc["name"] in CANCELLABLE_TOOLS:
            args["task_id"] = tc["id"]
        register_running_tool(tc["id"], tc["name"])
        try:
            results[tc["id"]] = await tool_map[tc["name"]].ainvoke(args)
        except asyncio.CancelledError:
            raise  # 客户端断开：让取消继续向上传播，但 finally 保证注销登记
        except Exception as e:  # noqa: BLE001 - 工具错误要回给模型，而不是让整轮失败
            results[tc["id"]] = f"工具调用失败: {e}"
        finally:
            unregister_running_tool(tc["id"])

    if runnable:
        # 同一轮的多个工具调用并行执行
        await asyncio.gather(*(_run_one(tc) for tc in runnable))

    # 按 tool_calls 原始顺序组装 ToolMessage
    tool_messages: list[ToolMessage] = []
    for tc in response.tool_calls:
        if tc["id"] in deferred:
            # 被 ask_user 拦下的调用不在这里补 ToolMessage，否则续答时同一 id 有两条 tool 消息，模型报 400
            continue
        result = results.get(tc["id"], "")
        # 工具可能返回 content block（如 view_image），必须原样传，str() 会毁掉图片块
        content = result if isinstance(result, list) else str(result)
        tool_messages.append(
            ToolMessage(content=content, tool_call_id=tc["id"], name=tc["name"])
        )

    update: dict[str, Any] = {"messages": [response] + tool_messages}
    if ask_question is not None:
        update.update({
            "pending_question": ask_question,  # 传给 ask_user_question
            "pending_tool_call_id": ask_tool_call_id,  # 传给 ask_user_question
        })
        return Command(update=update, goto="ask_user_question")
    return Command(update=update, goto="main_model_call")


async def ask_user_question(state: MainMessagesState, runtime: Runtime) -> Command:
    question = state.get("pending_question", "请提供更多信息")
    tool_call_id = state.get("pending_tool_call_id", "")

    # interrupt() 恢复时节点会从头重放，之前不能有副作用
    answer = interrupt(question)

    # 无论用户答没答，都必须返回 ToolMessage
    content = str(answer).strip() if str(answer).strip() else "用户没有提供回答"

    return Command(
        update={
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
            "pending_question": "",
            "pending_tool_call_id": "",
        },
        goto="main_model_call"
    )


agent_builder = StateGraph(MainMessagesState)
agent_builder.add_node("main_model_call", main_model_call)
agent_builder.add_node("ask_user_question", ask_user_question)

# 摘要只挂在用户轮次入口，工具循环和 ask_user 续答不重复摘要
_summarize_history = build_summarize_node()
if _summarize_history is not None:
    agent_builder.add_node("summarize_history", _summarize_history)
    agent_builder.add_edge(START, "summarize_history")
    agent_builder.add_edge("summarize_history", "main_model_call")
else:
    agent_builder.add_edge(START, "main_model_call")

agent_builder.add_edge("main_model_call", END)


if __name__ == "__main__":
    # 独立调试入口：需要先启动 MCP 服务（scripts/start_mcp_servers.py）
    async def _demo():
        from src.agents.checkpoint import init_async_persistence

        await init_async_persistence()
        await init_agent()
        agent = get_main_agent()
        result = await agent.ainvoke(
            {
                "messages": [HumanMessage(content="你好")],
                "summarized_messages": [],
                "context": {},
                "task_completed": False,
                "pending_question": "",
                "pending_tool_call_id": "",
            },
            config={"configurable": {"thread_id": "user_session"}},
        )
        print("\n=== 最终结果 ===")
        print("任务完成状态:", result.get("task_completed", False))
        if result["messages"]:
            print("最后消息:", result["messages"][-1].content)

    # Windows 上必须走 SelectorEventLoop，否则异步 psycopg 直接不可用
    run_async(_demo())
