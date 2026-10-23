"""RAG 检索服务（MCP）。

检索流程：**向量召回 -> 交叉编码器重排 -> 阈值过滤 -> 格式化**。

- 召回：用 `similarity_search_with_score` 取 RECALL_K 条候选（远多于最终返回数，
  给重排留出挑选空间）。它返回的是余弦**距离**（pgvector 的 `<=>`），换算成
  相似度 = 1 - 距离；
- 重排：`src.rag.reranker` 用交叉编码器对候选重新打分，按相关度降序取 top_k。
  重排不可用时自动退回"余弦相似度 + 余弦阈值"的旧行为（fail-open）；
- 阈值：两种分数不同量纲，因此各有一个阈值——重排分数用 RAG_MIN_RERANK_SCORE，
  余弦相似度用 RAG_MIN_SCORE。低于阈值的片段直接丢弃，一条都不剩时明确告知模型
  "知识库未命中"，避免它把噪声当依据。

阈值依据（本项目语料实测）：
    余弦相似度：相关 top1 ≈ 0.64~0.71，无关 top1 ≈ 0.24~0.28  -> 默认 0.4
    重排相关度：相关 top1 ≈ 0.16~0.87，无关全库最高 0.00005   -> 默认 0.1
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# 先补 sys.path：直接运行本文件时 sys.path[0] 是 mcps/，否则找不到 src 包
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from fastmcp import FastMCP
from langchain_core.documents import Document
from langchain_postgres import PGVector
from src.agents.models import SiliconFlowEmbeddings
from src.rag import reranker

load_dotenv()

mcp = FastMCP("RAG")

COLLECTION_NAME = "knowledge_base"

DEFAULT_TOP_K = 3

# top_k 上限：模型可能给出很大的 k，不设限会把整个知识库拼进上下文
MAX_TOP_K = int(os.getenv("RAG_MAX_TOP_K", "10"))

# 召回条数：候选太少重排没意义，调大更贵也更慢
RECALL_K = max(DEFAULT_TOP_K, int(os.getenv("RAG_RECALL_K", "20")))

# 余弦相似度阈值（取值 [-1, 1]）：仅用于"重排不可用"时的退回路径
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.4"))

# 重排相关度阈值（取值 [0, 1]）：交叉编码器分数与余弦相似度不是一个量纲，不能共用阈值
MIN_RERANK_SCORE = float(os.getenv("RAG_MIN_RERANK_SCORE", "0.1"))

_vectorstore: PGVector | None = None


def get_vectorstore() -> PGVector:
    """懒加载向量库实例（进程内复用）。

    不在 import 时创建：DB_URI 缺失或写错应该在真正检索时报出可读错误，而不是让整个
    MCP 服务进程起不来；同时便于单测替换成假实现。
    """
    global _vectorstore
    if _vectorstore is None:
        db_uri = os.getenv("DB_URI")
        if not db_uri:
            raise RuntimeError("未配置 DB_URI，无法连接知识库")
        _vectorstore = PGVector(
            connection=db_uri.replace("postgresql://", "postgresql+psycopg://"),
            embeddings=SiliconFlowEmbeddings(),
            collection_name=COLLECTION_NAME,
        )
    return _vectorstore


def score_of(distance: float) -> float:
    """余弦距离 -> 余弦相似度（pgvector 的 `<=>` 距离 = 1 - 余弦相似度）。"""
    return 1.0 - float(distance)


def format_results(
    hits: list[tuple[Document, float]],
    min_score: float = MIN_RERANK_SCORE,
    limit: int | None = None,
    label: str = "相关度",
) -> str:
    """过滤 + 排序 + 截断 + 格式化成给模型的文本。

    hits 里的分数语义统一为"越大越相关"（重排相关度或余弦相似度，由调用方换算好）。
    一条都不剩时明确说明"未命中"并给出候选池里的最高分，让模型知道可以照常回答，
    但不应把结果当作知识库依据。
    """
    kept = [(doc, score) for doc, score in hits if score >= min_score]
    best = max((score for _, score in hits), default=None)

    if not kept:
        if best is None:
            return "知识库中未找到相关内容。可基于通用知识回答，并说明该结论不来自知识库。"
        return (
            f"知识库中未找到相关内容（最高{label} {best:.3f}，低于阈值 {min_score:g}）。"
            "可基于通用知识回答，并说明该结论不来自知识库。"
        )

    # 显式按分数降序，不依赖底层返回顺序
    kept.sort(key=lambda item: -item[1])
    if limit is not None:
        kept = kept[:limit]

    return "\n\n".join(
        f"【片段 {i}】{label} {score:.3f}（来源：{doc.metadata.get('source', '未知来源')}）\n"
        f"{doc.page_content}"
        for i, (doc, score) in enumerate(kept, 1)
    )


@mcp.tool()
def search_knowledge(query: str, top_k: int = DEFAULT_TOP_K) -> str:
    """
    在知识库中检索与查询相关的内容。

    参数:
        query: 用户的问题或查询文本
        top_k: 返回最相关的文档片段数量（默认3，上限10）

    返回:
        按相关度从高到低排列的文档片段，每段标注相关度与来源；
        若所有片段都低于阈值，则明确返回"未找到相关内容"。
    """
    query = (query or "").strip()
    if not query:
        return "查询内容为空，请提供要检索的问题。"

    k = max(1, min(int(top_k), MAX_TOP_K))
    try:
        recalled = get_vectorstore().similarity_search_with_score(
            query, k=max(k, RECALL_K)
        )
    except Exception as e:
        return f"检索失败: {e}"

    documents = [doc for doc, _ in recalled]
    ranked = reranker.rerank(query, [doc.page_content for doc in documents])

    if ranked is None:
        return format_results(
            [(doc, score_of(distance)) for doc, distance in recalled],
            min_score=MIN_SCORE,
            limit=k,
            label="相似度",
        )

    # 重排只回下标，映射回 Document；越界条目忽略
    hits = [
        (documents[index], score)
        for index, score in ranked
        if 0 <= index < len(documents)
    ]
    return format_results(
        hits, min_score=MIN_RERANK_SCORE, limit=k, label="相关度"
    )


if __name__ == "__main__":
    from src.agents.mcps.registry import RAG_SERVER, serve

    serve(mcp, RAG_SERVER)
