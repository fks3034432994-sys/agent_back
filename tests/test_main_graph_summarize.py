"""对话历史摘要（langmem SummarizationNode 接入 main_graph）。

不连库、不连 MCP：摘要模型用假对象注入，覆盖三件事：
1. messages / summarized_messages 两个通道的 reducer 契约；
2. 摘要节点"没触发摘要就不写通道"的取舍；
3. main_model_call 优先用摘要后的历史，回退到完整转录。
"""

from typing import Any

import pytest
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages

from src.agents.graph import main_graph
from src.asyncio_runner import run_async

# 摘要触发用的长文本：约 39 token/条，两条就超过 50 token 阈值但又不至于装不进预算
LONG = "用户的问题内容" * 20


class FakeSummaryModel:
    """只回一段摘要文本的假模型（SummarizationNode 会先调 bind）。"""

    def __init__(self, reply="这是历史摘要"):
        self.reply = reply
        self.calls: list[Any] = []
        self.bound: dict[str, Any] = {}

    def bind(self, **kwargs):
        self.bound = kwargs
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.reply)

    async def ainvoke(self, messages):
        return self.invoke(messages)


class RecordingLLM:
    """记录收到的消息，用来断言 main_model_call 到底把哪份历史发给了模型。"""

    def __init__(self, response):
        self._response = response
        self.calls: list[Any] = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self._response


def msg(content="x", **kwargs):
    """带 id 的消息：SummarizationNode 硬性要求每条消息都有 id。"""
    kwargs.setdefault("id", f"id-{abs(hash(content))}")
    return HumanMessage(content=content, **kwargs)


@pytest.fixture()
def summarize_config(monkeypatch):
    """把摘要阈值调小，让几轮短对话就能触发。"""
    monkeypatch.setattr(main_graph, "HISTORY_SUMMARY_ENABLED", True)
    monkeypatch.setattr(main_graph, "HISTORY_SUMMARY_MAX_TOKENS", 100)
    monkeypatch.setattr(main_graph, "HISTORY_SUMMARY_BEFORE_TOKENS", 50)
    monkeypatch.setattr(main_graph, "HISTORY_SUMMARY_MAX_SUMMARY_TOKENS", 20)


def test_replace_reducer_assigns_missing_ids():
    """摘要 SystemMessage 是新建对象、没有 id，必须在这里补上。"""
    plain = SystemMessage(content="摘要")

    out = main_graph.replace_messages_with_ids([], [plain])

    assert out == [plain]
    assert out[0].id


def test_replace_reducer_keeps_existing_ids():
    message = msg("已有 id", id="keep-me")

    out = main_graph.replace_messages_with_ids([], [message])

    assert out[0].id == "keep-me"


def test_replace_reducer_replaces_instead_of_appending():
    """SummarizationNode 每轮返回完整历史（摘要+近期），累加会越滚越大。"""
    old = [msg("旧"), msg("更旧")]
    new = [msg("摘要后的第一条")]

    out = main_graph.replace_messages_with_ids(old, new)

    assert out == new
    assert len(out) == 1


@pytest.mark.parametrize("right", [None, []])
def test_replace_reducer_tolerates_empty_input(right):
    assert main_graph.replace_messages_with_ids([msg("旧")], right) == right


def test_replace_reducer_skips_non_message_items():
    """通道里万一混进 dict（LangGraph 允许）也不能崩。"""
    out = main_graph.replace_messages_with_ids([], [{"type": "text"}])

    assert out == [{"type": "text"}]


def test_messages_channel_uses_add_messages():
    """messages 必须是 add_messages：它负责自动补 id，摘要节点依赖这一点。"""
    annotation = main_graph.MainMessagesState.__annotations__["messages"]

    assert getattr(annotation, "__metadata__", None) == (add_messages,)


def test_summarize_node_disabled_by_config(monkeypatch):
    monkeypatch.setattr(main_graph, "HISTORY_SUMMARY_ENABLED", False)

    assert main_graph.build_summarize_node(model=FakeSummaryModel()) is None


def test_summarize_node_is_wired_before_model_call():
    """摘要只在用户轮次入口执行，不能嵌进工具循环。"""
    assert "summarize_history" in main_graph.agent_builder.nodes
    assert ("__start__", "summarize_history") in main_graph.agent_builder.edges
    assert ("summarize_history", "main_model_call") in main_graph.agent_builder.edges


def test_summarize_node_binds_summary_token_cap(summarize_config, monkeypatch):
    model = FakeSummaryModel()

    main_graph.build_summarize_node(model=model)

    assert model.bound == {"max_tokens": main_graph.HISTORY_SUMMARY_MAX_SUMMARY_TOKENS}


def test_summarize_node_skips_update_below_threshold(summarize_config):
    """没触发摘要就不写通道：否则每轮都把完整历史拷一份进检查点。"""
    node = main_graph.build_summarize_node(model=FakeSummaryModel())

    update = run_async(node({"messages": [msg("短", id="m1")]}))

    assert update == {}


def test_summarize_node_summarizes_when_over_threshold(summarize_config):
    model = FakeSummaryModel(reply="用户在咨询请假制度")
    node = main_graph.build_summarize_node(model=model)
    state = {
        "messages": [
            HumanMessage(content=LONG, id="m1"),
            HumanMessage(content=LONG, id="m2"),
            HumanMessage(content="最新一句", id="m3"),
        ]
    }

    update = run_async(node(state))

    assert model.calls, "超过阈值必须调用摘要模型"
    assert set(update) == {"summarized_messages", "context"}
    summarized = update["summarized_messages"]
    assert isinstance(summarized[0], SystemMessage)
    assert "用户在咨询请假制度" in summarized[0].content
    assert summarized[-1].id == "m3"
    assert update["context"]["running_summary"].summary == "用户在咨询请假制度"


def apply_update(state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """按通道 reducer 把节点返回的 update 合进 state（模拟 LangGraph 的合并）。

    摘要 SystemMessage 的 id 是**在 reducer 里**补的，所以这个测试不能直接
    state.update(update)，否则就等于绕过了被验证的那一环。
    """
    if "messages" in update:
        state["messages"] = add_messages(state.get("messages", []), update["messages"])
    if "summarized_messages" in update:
        state["summarized_messages"] = main_graph.replace_messages_with_ids(
            state.get("summarized_messages"), update["summarized_messages"]
        )
    if "context" in update:
        state["context"] = update["context"]
    return state


def test_summarize_node_keeps_working_after_repeated_runs(summarize_config):
    """连续多轮：摘要被扩展而不是每次从零重算，且消息始终带 id。"""
    model = FakeSummaryModel()
    node = main_graph.build_summarize_node(model=model)
    state: dict[str, Any] = {
        "messages": [
            HumanMessage(content=LONG, id="m1"),
            HumanMessage(content=LONG, id="m2"),
        ]
    }

    for i in range(5):
        apply_update(state, {"messages": [HumanMessage(content=LONG, id=f"n{i}")]})
        apply_update(state, run_async(node(state)))

        assert all(m.id for m in state["summarized_messages"])
        assert state["summarized_messages"][0].content.startswith("Summary")
        assert state["summarized_messages"][-1].id == f"n{i}"

    # 摘要对象跨轮保留，不每轮重算
    assert model.calls, "至少触发过一次摘要"
    assert state["context"]["running_summary"].summarized_message_ids
    # 不变量：完整转录在涨，发给模型的历史被压住
    assert len(state["messages"]) > len(state["summarized_messages"])


def test_summarize_node_keeps_current_question_when_folded_into_summary(summarize_config):
    """摘要首次触发时会把当前提问一起折进摘要（只剩一条 summary），必须补回来。

    这是实测行为：2 条消息刚到阈值时，langmem 返回的 summarized_messages 只有摘要本身，
    模型将看不到用户当前问题的原文。
    """
    node = main_graph.build_summarize_node(model=FakeSummaryModel())
    state = {
        "messages": [
            HumanMessage(content=LONG, id="m1"),
            HumanMessage(content=LONG, id="m2"),  # 当前提问
        ]
    }

    update = run_async(node(state))

    assert update["summarized_messages"][-1].id == "m2"
    assert update["summarized_messages"][-1].content == LONG


def test_summarize_node_ignores_empty_summary(summarize_config):
    """模型返回空摘要时放弃压缩。

    真实场景：main_model 是推理模型，绑定的 max_tokens 被思维链吃光后正文返回空串
    （finish_reason=length）。此时若照常写通道，不但没有压缩效果，还会把已有的好摘要
    覆盖成空壳 —— 必须退回"用完整转录"。
    """
    node = main_graph.build_summarize_node(model=FakeSummaryModel(reply="   "))
    state = {
        "messages": [
            HumanMessage(content=LONG, id="m1"),
            HumanMessage(content=LONG, id="m2"),
        ]
    }

    update = run_async(node(state))

    assert update == {}


def test_summarize_node_does_not_duplicate_kept_question(summarize_config):
    """已经在尾部里的当前提问不能补第二遍（补重复会让模型看到两遍同一句）。"""
    node = main_graph.build_summarize_node(model=FakeSummaryModel())
    state = {
        "messages": [
            HumanMessage(content=LONG, id="m1"),
            HumanMessage(content=LONG, id="m2"),
            HumanMessage(content=LONG, id="m3"),
        ]
    }

    update = run_async(node(state))

    ids = [m.id for m in update["summarized_messages"]]
    assert ids.count("m3") == 1


def test_main_model_call_prefers_summarized_history(monkeypatch):
    summarized = [
        SystemMessage(content="历史摘要", id="s1"),
        HumanMessage(content="最新问题", id="h1"),
    ]
    raw = [
        HumanMessage(content="很久以前的问题", id="old1"),
        HumanMessage(content="很久以前的回复", id="old2"),
    ] + summarized
    llm = RecordingLLM(AIMessage(content="好的"))
    monkeypatch.setattr(main_graph, "get_llm_and_tools", lambda: (llm, {}))

    run_async(
        main_graph.main_model_call(
            {"messages": raw, "summarized_messages": summarized}, None
        )
    )

    assert llm.calls[0] == summarized


@pytest.mark.parametrize("summarized", [None, []])
def test_main_model_call_falls_back_to_full_transcript(monkeypatch, summarized):
    """尚未触发摘要时 summarized_messages 为空，必须回退到完整转录。"""
    raw = [HumanMessage(content="问题", id="h1")]
    llm = RecordingLLM(AIMessage(content="好的"))
    monkeypatch.setattr(main_graph, "get_llm_and_tools", lambda: (llm, {}))

    state = {"messages": raw}
    if summarized is not None:
        state["summarized_messages"] = summarized
    run_async(main_graph.main_model_call(state, None))

    assert llm.calls[0] == raw
