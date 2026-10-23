"""检索查询改写的纯逻辑（src/rag/query_rewrite.py）。

不调模型、不连库：这里只覆盖"判断要不要改、历史怎么渲染、提示怎么拼、输出怎么清洗"
四件事。真正的模型调用在 tests/test_main_graph_tools.py 里用假模型覆盖。
"""

import pytest
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.rag import query_rewrite as qr


def test_text_of_collapses_whitespace():
    assert qr.text_of("  请假\n\n 几天 ") == "请假 几天"


def test_text_of_extracts_only_text_blocks():
    """多模态内容里图片块不产出文本，直接 str() 会得到 repr 噪声。"""
    content = [
        {"type": "image_ref", "name": "a.png"},
        {"type": "text", "text": "看这张图"},
    ]

    assert qr.text_of(content) == "看这张图"


def test_text_of_handles_none_and_other_types():
    assert qr.text_of(None) == ""
    assert qr.text_of(123) == "123"


def test_render_history_excludes_current_question():
    """本轮提问会单独作为"最新提问"传入，不能又在历史里重复一遍。"""
    messages = [
        SystemMessage(content="你是助手"),
        HumanMessage(content="公司请假制度是什么"),
        AIMessage(content="事假全年不超过 15 天"),
        HumanMessage(content="那这个要提前几天"),
    ]

    out = qr.render_history(messages)

    assert "公司请假制度是什么" in out
    assert "事假全年不超过 15 天" in out
    assert "那这个要提前几天" not in out


def test_render_history_skips_system_and_tool_messages():
    messages = [
        SystemMessage(content="全局设定"),
        HumanMessage(content="问题一"),
        AIMessage(content="回答一"),
        ToolMessage(content="工具结果", tool_call_id="c1"),
        HumanMessage(content="问题二"),
    ]

    out = qr.render_history(messages)

    assert out == "用户：问题一\n助手：回答一"


def test_render_history_keeps_only_most_recent_messages():
    messages = [HumanMessage(content=f"问题{i}") for i in range(10)]
    messages.append(HumanMessage(content="最新"))

    out = qr.render_history(messages, max_messages=3)

    assert out == "用户：问题7\n用户：问题8\n用户：问题9"


def test_render_history_skips_empty_assistant_messages():
    """纯 tool_call 的 AIMessage 内容为空，渲染进去只会多两行噪声。"""
    messages = [
        HumanMessage(content="问题"),
        AIMessage(content=""),
        AIMessage(content="真正的回答"),
        HumanMessage(content="追问"),
    ]

    out = qr.render_history(messages)

    assert out == "用户：问题\n助手：真正的回答"


def test_render_history_truncates_long_messages():
    messages = [HumanMessage(content="啊" * 500), HumanMessage(content="追问")]

    out = qr.render_history(messages)

    assert len(out) < qr.HISTORY_MESSAGE_MAX_CHARS + 10
    assert out.startswith("用户：啊")


def test_render_history_without_prior_turns_is_empty():
    messages = [SystemMessage(content="设定"), HumanMessage(content="第一句话")]

    assert qr.render_history(messages) == ""


HISTORY = "用户：公司请假制度是什么\n助手：事假全年不超过 15 天"


@pytest.mark.parametrize(
    "question",
    [
        "那这个要提前几天",       # 指代
        "年假呢",                # 指代 + 过短
        "继续",                  # 指代 + 过短
        "考勤制度",              # 过短（信息不足）
        "还有其他规定吗",         # 指代
    ],
)
def test_needs_rewrite_for_anaphoric_or_short_questions(question):
    assert qr.needs_rewrite(question, HISTORY) is True


@pytest.mark.parametrize(
    "question",
    [
        "公司请假需要提前几天申请",
        "费用报销流程是什么",
        "新员工入职需要准备哪些材料",
    ],
)
def test_does_not_rewrite_self_contained_questions(question):
    assert qr.needs_rewrite(question, HISTORY) is False


def test_needs_rewrite_requires_history():
    """没有历史时一律不改：没有可消解的对象，改了也是白问一次模型。"""
    assert qr.needs_rewrite("那这个要提前几天", "") is False
    assert qr.needs_rewrite("年假呢", "   ") is False


def test_needs_rewrite_ignores_blank_question():
    assert qr.needs_rewrite("", HISTORY) is False
    assert qr.needs_rewrite("   ", HISTORY) is False


def test_build_condense_messages_carries_history_and_question():
    messages = qr.build_condense_messages("那这个要提前几天", HISTORY)

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert "独立检索" in messages[0].content
    body = messages[1].content
    assert HISTORY in body
    assert "那这个要提前几天" in body


def test_parse_condensed_keeps_plain_output():
    assert qr.parse_condensed("请假需要提前几天申请", "原问题") == "请假需要提前几天申请"


@pytest.mark.parametrize(
    "raw",
    [
        "改写后：请假需要提前几天申请",
        "独立问题: 请假需要提前几天申请",
        "以下是改写后的独立问题：请假需要提前几天申请",
    ],
)
def test_parse_condensed_strips_label_prefix(raw):
    assert qr.parse_condensed(raw, "原问题") == "请假需要提前几天申请"


@pytest.mark.parametrize(
    "raw",
    [
        "「请假需要提前几天申请」",
        '"请假需要提前几天申请"',
        "“请假需要提前几天申请”",
        "《请假需要提前几天申请》",
    ],
)
def test_parse_condensed_strips_wrapping_quotes(raw):
    assert qr.parse_condensed(raw, "原问题") == "请假需要提前几天申请"


def test_parse_condensed_keeps_only_first_line():
    raw = "请假需要提前几天申请\n\n这是因为公司规定……"

    assert qr.parse_condensed(raw, "原问题") == "请假需要提前几天申请"


def test_parse_condensed_does_not_mangle_time_like_questions():
    """「9:00 上班吗」不是"标签：内容"，不能把 9 当成前缀剥掉。"""
    assert qr.parse_condensed("9:00 上班吗", "原问题") == "9:00 上班吗"


@pytest.mark.parametrize("raw", ["", "   ", None, "\n\n"])
def test_parse_condensed_falls_back_when_empty(raw):
    assert qr.parse_condensed(raw, "原问题") == "原问题"


def test_parse_condensed_falls_back_when_too_long():
    """明显在回答问题而不是改写时回退：改写结果不该比问题长一大截。"""
    answer = "根据公司规定，事假应提前 1 天申请，病假需当日 9:00 前报备……" * 10

    assert qr.parse_condensed(answer, "那这个呢") == "那这个呢"


def test_parse_condensed_falls_back_when_only_a_label():
    assert qr.parse_condensed("改写后：", "原问题") == "原问题"
