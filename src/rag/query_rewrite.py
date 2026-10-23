"""检索查询改写（纯逻辑部分，不含模型调用）。

要解决的问题：多轮对话里的追问大量依赖指代——「那这个要提前几天」「年假呢」。
这类文本直接拿去嵌入，检索到的是字面相似的内容，与用户真正想问的无关。而
``search_knowledge`` 跑在**独立的 MCP 服务进程**里，拿不到对话历史，所以消解
只能在 agent 侧（拿到 messages 的那一层）完成。

本模块只负责"判断要不要改、拼提示词、清洗模型输出"这些可单测的纯逻辑；
真正调模型在 ``main_graph.condense_question`` 里做。

两个安全前提：
- 判断"需要改写"只是**粗筛**，宁可多改：多改的代价是多一次模型调用，判漏的代价
  只是回到"不改写"的旧行为；
- 输出清洗失败一律回退到原始问题——改写是辅助步骤，不该因为它出问题就让检索失败。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from langchain.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

# 指代/省略类标记：出现其一且存在历史时，认为问题不自包含
ANAPHORA_MARKERS = (
    "这个",
    "那个",
    "这些",
    "那些",
    "它",
    "上述",
    "上面",
    "前面",
    "刚才",
    "之前",
    "继续",
    "还有",
    "另外",
    "其他",
    "其中",
    "同样",
    "一样",
    "呢",
)

# 短于该长度且有历史时，认为信息不足，需要结合上下文补全
MIN_SELF_CONTAINED_CHARS = 8

# 单条历史消息渲染时截断的长度，避免把长回答整段塞进改写提示
HISTORY_MESSAGE_MAX_CHARS = 200

# 改写结果的长度上限：超过它基本可以断定模型在回答而不是改写
MAX_QUERY_CHARS = 200

CONDENSE_SYSTEM_PROMPT = """你是检索查询改写器。把用户的最新提问改写成可**独立检索**的问题。

规则：
1. 补全指代（如"这个/那个/它/上面说的"）所指代的具体对象；
2. 补全被省略的主语或对象，措辞可从对话历史里取；
3. 只输出改写后的问句本身，不要回答问题、不要解释、不要加引号或前缀；
4. 若最新提问本身已经能独立检索，原样输出。"""

_LABEL_PREFIX = re.compile(r"^(?!\d)[^：:\n]{0,12}[：:]\s*")
_WRAPPING = "「」『』\"'“”‘’《》 \t"


def text_of(content: Any) -> str:
    """把消息 content 压成纯文本并折叠空白。

    多模态消息的 content 是 content block 列表（图片块不产出文本），
    直接 str() 会得到一段 Python repr 噪声。
    """
    if isinstance(content, str):
        raw = content
    elif isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        raw = " ".join(parts)
    else:
        raw = "" if content is None else str(content)
    return re.sub(r"\s+", " ", raw).strip()


def render_history(messages: Sequence[AnyMessage], max_messages: int = 6) -> str:
    """把对话历史渲染成"用户：… / 助手：…"的纯文本。

    只取**最后一轮用户提问之前**的部分：本轮提问会单独作为"最新提问"传进去，
    若也留在历史里就会在提示中重复一次。这也让本函数对"同一轮里的第二次检索"
    表现一致（此时 messages 末尾是工具结果而不是用户提问）。

    System / Tool 消息不参与改写——前者是全局设定，后者是检索结果，都不是
    "用户想说什么"的证据。返回空串表示没有可用历史。
    """
    end = len(messages)
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            end = index
            break

    lines: list[str] = []
    for message in messages[:end]:
        if isinstance(message, HumanMessage):
            role = "用户"
        elif isinstance(message, AIMessage):
            role = "助手"
        else:
            continue
        text = text_of(message.content)[:HISTORY_MESSAGE_MAX_CHARS]
        if text:
            lines.append(f"{role}：{text}")

    if max_messages > 0:
        lines = lines[-max_messages:]
    return "\n".join(lines)


def needs_rewrite(question: str, history: str) -> bool:
    """是否需要改写。history 用 render_history 的结果（空串表示没有历史）。

    粗筛规则：有历史 且（含指代标记 或 提问过短）。没有任何历史时一律不改：
    没有可消解的对象，问了也白问。
    """
    if not history.strip():
        return False
    text = (question or "").strip()
    if not text:
        return False
    if len(text) < MIN_SELF_CONTAINED_CHARS:
        return True
    return any(marker in text for marker in ANAPHORA_MARKERS)


def build_condense_messages(question: str, history: str) -> list[AnyMessage]:
    """拼改写提示。history 是 render_history 的结果，避免重复渲染。"""
    body = f"对话历史：\n{history}\n\n最新提问：{question}"
    return [
        SystemMessage(content=CONDENSE_SYSTEM_PROMPT),
        HumanMessage(content=body),
    ]


def parse_condensed(text: str, original: str) -> str:
    """清洗改写模型的输出；不可用时返回原问题。

    模型偶尔会带上"改写后：""独立问题："这类前缀、包一层引号，或者干脆多写一段
    解释——只取第一行并剥掉这些外壳；结果为空或过长（明显在回答而非改写）时回退。
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return original

    line = cleaned.splitlines()[0].strip()
    line = _LABEL_PREFIX.sub("", line).strip()
    line = line.strip(_WRAPPING).strip()

    if not line or len(line) > MAX_QUERY_CHARS:
        return original
    return line
