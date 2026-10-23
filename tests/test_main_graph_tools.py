"""主 agent 的工具循环：图片上下文处理 + 工具调用编排。

不连 MCP、不调模型：把 get_llm_and_tools 换成假 LLM 与假工具。
重点覆盖两处容易回归的逻辑：
- 图片引用（image_ref）只在"本轮"加载，历史轮次降级为文字占位（控 token 与存储）；
- 被 ask_user 拦下的 tool_call **不能**在这里补 ToolMessage——否则用户回答后续答时，
  同一个 tool_call_id 会出现两条 tool 消息，模型端直接 400。
"""

import asyncio
import base64

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.constants import END

from src.agents.graph import main_graph
from src.asyncio_runner import run_async


class FakeTool:
    def __init__(self, name, result=None, error=None, on_call=None):
        self.name = name
        self._result = result
        self._error = error
        self._on_call = on_call
        self.calls: list[dict] = []

    async def ainvoke(self, args):
        self.calls.append(dict(args))
        if self._on_call is not None:
            await self._on_call(args)
        if self._error is not None:
            raise self._error
        return self._result


class FakeLLM:
    def __init__(self, response):
        self._response = response
        self.calls: list = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self._response


def patch_llm(monkeypatch, response, tools):
    monkeypatch.setattr(main_graph, "get_llm_and_tools", lambda: (FakeLLM(response), tools))


def call_node(response, tools):
    return run_async(main_graph.main_model_call({"messages": [HumanMessage(content="x")]}, None))


def tool_calls_msg(*calls) -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": call_id} for name, args, call_id in calls
    ])


def tool_messages_of(command) -> list[ToolMessage]:
    return [m for m in command.update["messages"] if isinstance(m, ToolMessage)]


def test_resolve_images_loads_current_turn_image(monkeypatch):
    monkeypatch.setattr(main_graph, "read_upload_bytes", lambda name: b"\x89PNG")
    monkeypatch.setattr(main_graph, "guess_media_type", lambda name: "image/png")
    msg = HumanMessage(content=[
        {"type": "image_ref", "name": "a.png"},
        {"type": "text", "text": "看这张图"},
    ])

    resolved = main_graph._resolve_images([msg])

    assert resolved[0] is not msg
    assert resolved[0].content[0] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(b"\x89PNG").decode()},
    }
    assert resolved[0].content[1] == {"type": "text", "text": "看这张图"}


def test_resolve_images_downgrades_history_images(monkeypatch):
    """历史轮次的图片必须降级成文字占位，否则每轮都重发 base64。"""
    monkeypatch.setattr(main_graph, "read_upload_bytes", lambda name: b"\x89PNG")
    monkeypatch.setattr(main_graph, "guess_media_type", lambda name: "image/png")
    history = HumanMessage(content=[{"type": "image_ref", "name": "old.png"}])
    current = HumanMessage(content="换个话题")

    resolved = main_graph._resolve_images([history, current])

    placeholder = resolved[0].content[0]["text"]
    assert "［图片：old.png］" in placeholder
    assert "view_image" in placeholder
    assert resolved[1] is current


def test_resolve_images_keeps_plain_messages_untouched():
    plain = HumanMessage(content="你好")

    resolved = main_graph._resolve_images([plain])

    assert resolved[0] is plain, "没有图片引用时不该复制消息对象"


def test_resolve_images_reports_missing_image(monkeypatch):
    monkeypatch.setattr(main_graph, "read_upload_bytes", lambda name: None)
    msg = HumanMessage(content=[{"type": "image_ref", "name": "gone.png"}])

    resolved = main_graph._resolve_images([msg])

    assert "图片加载失败" in resolved[0].content[0]["text"]


def test_answer_without_tool_calls_ends_the_graph(monkeypatch):
    response = AIMessage(content="答完了")
    patch_llm(monkeypatch, response, {})

    command = call_node(response, {})

    assert command.goto == END
    assert command.update["task_completed"] is True
    assert command.update["messages"][0].content == "答完了"


def test_ask_user_is_deferred_to_interrupt_node(monkeypatch):
    response = tool_calls_msg(
        ("ask_user", {"question": "选哪个？"}, "c-ask"),
        ("read_file", {"path": "a.txt"}, "c-read"),
    )
    ask_tool = FakeTool("ask_user")
    read_tool = FakeTool("read_file", result="文件内容")
    patch_llm(monkeypatch, response, {"ask_user": ask_tool, "read_file": read_tool})

    command = call_node(response, {})

    assert command.goto == "ask_user_question"
    assert command.update["pending_question"] == "选哪个？"
    assert command.update["pending_tool_call_id"] == "c-ask"
    assert [m.tool_call_id for m in tool_messages_of(command)] == ["c-read"]
    assert tool_messages_of(command)[0].content == "文件内容"
    assert tool_messages_of(command)[0].name == "read_file"
    assert ask_tool.calls == [], "ask_user 只是提问信号，不应被执行"


def test_second_ask_user_in_same_round_gets_placeholder(monkeypatch):
    """同一轮问两次：只保留第一个，第二个补占位，保证每个 tool_call 都有配对消息。"""
    response = tool_calls_msg(
        ("ask_user", {"question": "第一个问题"}, "c1"),
        ("ask_user", {"question": "第二个问题"}, "c2"),
    )
    patch_llm(monkeypatch, response, {"ask_user": FakeTool("ask_user")})

    command = call_node(response, {})

    assert command.update["pending_tool_call_id"] == "c1"
    placeholders = tool_messages_of(command)
    assert len(placeholders) == 1
    assert placeholders[0].tool_call_id == "c2"
    assert "等待用户回答" in placeholders[0].content


def test_unknown_tool_is_reported_as_text(monkeypatch):
    response = tool_calls_msg(("nope", {}, "c1"))
    patch_llm(monkeypatch, response, {})

    command = call_node(response, {})

    assert command.goto == "main_model_call"
    assert tool_messages_of(command)[0].content == "错误：未找到工具 'nope'"


def test_tool_exception_is_converted_to_text(monkeypatch):
    """单个工具失败要让模型知道并继续，而不是整轮对话失败。"""
    response = tool_calls_msg(("boom", {}, "c1"))
    patch_llm(monkeypatch, response, {"boom": FakeTool("boom", error=RuntimeError("炸了"))})

    command = call_node(response, {})

    assert tool_messages_of(command)[0].content.startswith("工具调用失败:")
    assert "炸了" in tool_messages_of(command)[0].content


def test_tools_in_one_round_run_in_parallel(monkeypatch):
    """同一轮多个互不依赖的调用应并行，延迟取最大值而不是求和。"""
    ends: list[str] = []
    entries: dict[str, int] = {}

    def make_tool(name: str, delay: float) -> FakeTool:
        async def on_call(args):
            entries[name] = len(ends)
            await asyncio.sleep(delay)
            ends.append(name)

        return FakeTool(name, result=name, on_call=on_call)

    response = tool_calls_msg(("slow", {}, "c1"), ("fast", {}, "c2"))
    patch_llm(monkeypatch, response, {
        "slow": make_tool("slow", 0.05),
        "fast": make_tool("fast", 0.0),
    })

    call_node(response, {})

    assert entries["fast"] == 0, "fast 开始时 slow 尚未结束，说明两者并行"
    assert ends == ["fast", "slow"]


def test_tool_messages_follow_model_order(monkeypatch):
    """并行执行后仍要按模型给出的顺序组装 ToolMessage（协议要求成对且有序）。"""
    def make_tool(name: str, delay: float) -> FakeTool:
        async def on_call(args):
            await asyncio.sleep(delay)

        return FakeTool(name, result=name, on_call=on_call)

    response = tool_calls_msg(("t1", {}, "c1"), ("t2", {}, "c2"), ("t3", {}, "c3"))
    patch_llm(monkeypatch, response, {
        "t1": make_tool("t1", 0.05),
        "t2": make_tool("t2", 0.02),
        "t3": make_tool("t3", 0.0),
    })

    command = call_node(response, {})

    assert [m.tool_call_id for m in tool_messages_of(command)] == ["c1", "c2", "c3"]


def test_cancellable_tool_receives_tool_call_id_as_task_id(monkeypatch):
    """取消要靠 task_id 定位"这一次"调用，注入值必须等于 tool_call_id。"""
    shell = FakeTool("run_shell_command", result="ok")
    response = tool_calls_msg(("run_shell_command", {"command": "pwd"}, "c-shell"))
    patch_llm(monkeypatch, response, {"run_shell_command": shell})

    call_node(response, {})

    assert shell.calls == [{"command": "pwd", "task_id": "c-shell"}]


def test_other_tool_args_are_not_modified(monkeypatch):
    read = FakeTool("read_file", result="ok")
    response = tool_calls_msg(("read_file", {"path": "a.txt"}, "c1"))
    patch_llm(monkeypatch, response, {"read_file": read})

    call_node(response, {})

    assert read.calls == [{"path": "a.txt"}]


def test_content_block_result_passes_through_unchanged(monkeypatch):
    """view_image 返回的是 content block 列表，不能被 str() 拍平成文本。"""
    blocks = [
        {"type": "text", "text": "以下是图片"},
        {"type": "image_ref", "name": "a.png"},
    ]
    response = tool_calls_msg(("view_image", {"name": "a.png"}, "c1"))
    patch_llm(monkeypatch, response, {"view_image": FakeTool("view_image", result=blocks)})

    command = call_node(response, {})

    assert tool_messages_of(command)[0].content == blocks


class FailingLLM:
    """改写失败（模型报错）时用。"""

    def __init__(self, error):
        self._error = error
        self.calls: list = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        raise self._error


HISTORY_THEN_ANAPHORA = [
    HumanMessage(content="公司请假制度是什么"),
    AIMessage(content="事假全年不超过 15 天"),
    HumanMessage(content="那这个要提前几天"),
]


def run_tool_node(monkeypatch, response, tools, messages):
    """按给定 state 跑一次工具循环。"""
    patch_llm(monkeypatch, response, tools)
    return run_async(main_graph.main_model_call({"messages": messages}, None))


def patch_rewrite_model(monkeypatch, reply="请假需要提前几天申请"):
    """替换改写用的上游模型（main_model），并记录调用。"""
    fake = FakeLLM(AIMessage(content=reply))
    monkeypatch.setattr(main_graph, "main_model", fake)
    return fake


def test_anaphoric_query_is_rewritten_before_retrieval(monkeypatch):
    search = FakeTool("search_knowledge", result="片段")
    response = tool_calls_msg(
        ("search_knowledge", {"query": "那这个要提前几天", "top_k": 3}, "c1")
    )
    rewrite = patch_rewrite_model(monkeypatch)

    run_tool_node(
        monkeypatch, response, {"search_knowledge": search}, HISTORY_THEN_ANAPHORA
    )

    assert search.calls == [{"query": "请假需要提前几天申请", "top_k": 3}]
    # 改写提示里要带历史，否则无从消解指代
    assert "公司请假制度是什么" in rewrite.calls[0][1].content


def test_rewrite_does_not_mutate_model_tool_calls(monkeypatch):
    """只改"这一次执行用的参数"：检查点与前端看到的仍是模型原话。"""
    search = FakeTool("search_knowledge", result="片段")
    response = tool_calls_msg(("search_knowledge", {"query": "那这个要提前几天"}, "c1"))
    patch_rewrite_model(monkeypatch)

    run_tool_node(monkeypatch, response, {"search_knowledge": search}, HISTORY_THEN_ANAPHORA)

    assert response.tool_calls[0]["args"]["query"] == "那这个要提前几天"


def test_self_contained_query_skips_rewrite(monkeypatch):
    """自包含的问题不该多花一次模型往返。"""
    search = FakeTool("search_knowledge", result="片段")
    response = tool_calls_msg(
        ("search_knowledge", {"query": "费用报销流程是什么"}, "c1")
    )
    rewrite = patch_rewrite_model(monkeypatch)

    run_tool_node(
        monkeypatch,
        response,
        {"search_knowledge": search},
        [
            HumanMessage(content="你好"),
            AIMessage(content="有什么可以帮你"),
            HumanMessage(content="费用报销流程是什么"),
        ],
    )

    assert rewrite.calls == []
    assert search.calls == [{"query": "费用报销流程是什么"}]


def test_first_turn_skips_rewrite(monkeypatch):
    """没有历史时连"短问题"也不改写：没有可消解的对象。"""
    search = FakeTool("search_knowledge", result="片段")
    response = tool_calls_msg(("search_knowledge", {"query": "年假呢"}, "c1"))
    rewrite = patch_rewrite_model(monkeypatch)

    run_tool_node(monkeypatch, response, {"search_knowledge": search}, [HumanMessage(content="年假呢")])

    assert rewrite.calls == []
    assert search.calls == [{"query": "年假呢"}]


def test_rewrite_failure_falls_back_to_original_query(monkeypatch):
    """改写是辅助步骤：模型报错也要照原问题检索，不能让本轮工具失败。"""
    search = FakeTool("search_knowledge", result="片段")
    response = tool_calls_msg(("search_knowledge", {"query": "那这个要提前几天"}, "c1"))
    monkeypatch.setattr(main_graph, "main_model", FailingLLM(RuntimeError("model down")))

    command = run_tool_node(
        monkeypatch, response, {"search_knowledge": search}, HISTORY_THEN_ANAPHORA
    )

    assert search.calls == [{"query": "那这个要提前几天"}]
    assert tool_messages_of(command)[0].content == "片段"


def test_rewrite_falls_back_when_model_answers_instead_of_rewriting(monkeypatch):
    """模型偶尔会"回答"而不是"改写"，输出过长时回退原问题。"""
    search = FakeTool("search_knowledge", result="片段")
    response = tool_calls_msg(("search_knowledge", {"query": "那这个要提前几天"}, "c1"))
    patch_rewrite_model(monkeypatch, reply="根据公司规定，事假应提前 1 天申请……" * 20)

    run_tool_node(monkeypatch, response, {"search_knowledge": search}, HISTORY_THEN_ANAPHORA)

    assert search.calls == [{"query": "那这个要提前几天"}]


def test_rewrite_can_be_disabled(monkeypatch):
    search = FakeTool("search_knowledge", result="片段")
    response = tool_calls_msg(("search_knowledge", {"query": "那这个要提前几天"}, "c1"))
    rewrite = patch_rewrite_model(monkeypatch)
    monkeypatch.setattr(main_graph, "RAG_REWRITE_ENABLED", False)

    run_tool_node(monkeypatch, response, {"search_knowledge": search}, HISTORY_THEN_ANAPHORA)

    assert rewrite.calls == []
    assert search.calls == [{"query": "那这个要提前几天"}]


def test_non_retrieval_tools_are_not_rewritten(monkeypatch):
    """只有检索工具的 query 需要改写，别的工具的入参不能被动。"""
    view = FakeTool("view_image", result="图片")
    response = tool_calls_msg(("view_image", {"name": "那这个.png"}, "c1"))
    rewrite = patch_rewrite_model(monkeypatch)

    run_tool_node(
        monkeypatch, response, {"view_image": view}, HISTORY_THEN_ANAPHORA
    )

    assert rewrite.calls == []
    assert view.calls == [{"name": "那这个.png"}]
