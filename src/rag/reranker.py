"""Rerank（交叉编码器重排）。

为什么需要：向量检索用的是**双塔**模型——query 与 document 各自独立编码，只能比较
"整体语义是否接近"；交叉编码器把两者拼在一起过一遍模型，判断的是"这段文字是否真的
回答了这个问题"，区分度明显更高。

同一批语料上的实测对比（本项目知识库，10 个片段，模型 BAAI/bge-reranker-v2-m3）：

    相关提问「请假需要提前几天申请」：命中片段 0.579，其余 0.0001 ~ 0.124
    无关提问「今天北京的天气怎么样」：全库最高 0.00005
    6 组问答里，正确答案的 top1 得分 0.16 ~ 0.87（最低那组说明语料里确实没写清）

对照余弦相似度（命中 0.64 vs 噪声 0.53，只差 1.2 倍），重排把正确答案与噪声拉开了
4~5 倍，"低于阈值即丢弃"才真正可用。

设计原则：**重排失败不能影响检索**。未配置模型/密钥、超时、返回异常时一律返回 None，
调用方退回向量检索的原始结果（fail-open）。重排是锦上添花的一步，不该因为它不可用
就让整个检索失败。

配置（都可选，默认即可用）：
    RAG_RERANK_MODEL     重排模型，默认 BAAI/bge-reranker-v2-m3；置空则关闭重排
    RAG_RERANK_URL       重排端点，默认取 EMBEDDING_MODEL_URL + /rerank（与嵌入同源）
    RAG_RERANK_API_KEY   重排密钥，默认复用 EMBEDDING_MODEL_API_KEY
    RAG_RERANK_TIMEOUT   单次调用超时秒数，默认 30
"""

from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


def model_name() -> str:
    """当前使用的重排模型；返回空串表示关闭重排。"""
    return (os.getenv("RAG_RERANK_MODEL", DEFAULT_MODEL) or "").strip()


def endpoint() -> str:
    """重排端点。硅基流动的 rerank 与 embeddings 同源，所以默认复用同一个 base url。"""
    base = os.getenv("RAG_RERANK_URL") or os.getenv("EMBEDDING_MODEL_URL") or ""
    return base.rstrip("/") + "/rerank"


def api_key() -> str:
    """重排密钥；未单独配置时复用嵌入服务的密钥。"""
    return os.getenv("RAG_RERANK_API_KEY") or os.getenv("EMBEDDING_MODEL_API_KEY") or ""


def timeout() -> float:
    return float(os.getenv("RAG_RERANK_TIMEOUT", "30"))


def rerank(query: str, documents: list[str]) -> list[tuple[int, float]] | None:
    """给 documents 逐个打相关性分，返回 [(原下标, 分数), ...] 按分数降序。

    返回 None 表示"重排不可用"，调用方应退回向量检索顺序（fail-open）。
    """
    model = model_name()
    if not model or not documents:
        return None

    key = api_key()
    if not key:
        logger.warning(
            "未配置 rerank 密钥（RAG_RERANK_API_KEY / EMBEDDING_MODEL_API_KEY），跳过重排"
        )
        return None

    try:
        response = requests.post(
            endpoint(),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "query": query,
                "documents": documents,
                # 显式要全部：最强候选之外的分数在"未命中"提示里还要用（报告全局最高分）
                "top_n": len(documents),
            },
            timeout=timeout(),
        )
    except Exception:
        logger.exception("rerank 请求异常，退回向量检索顺序")
        return None

    if response.status_code != 200:
        logger.warning(
            "rerank 返回 %s，退回向量检索顺序：%s",
            response.status_code,
            (response.text or "")[:200],
        )
        return None

    try:
        results = response.json().get("results") or []
    except Exception:
        logger.exception("rerank 响应不是合法 JSON，退回向量检索顺序")
        return None

    scored: list[tuple[int, float]] = []
    for item in results:
        try:
            scored.append((int(item["index"]), float(item["relevance_score"])))
        except (KeyError, TypeError, ValueError):
            logger.warning("rerank 返回了无法解析的条目，退回向量检索顺序：%s", item)
            return None

    if not scored:
        return None

    scored.sort(key=lambda pair: -pair[1])
    return scored
