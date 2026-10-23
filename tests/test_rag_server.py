"""RAG 检索工具：召回 -> 重排 -> 阈值过滤 -> 格式化成文本。

（src/agents/mcps/rag_mcp_server.py）

不连数据库、不联网：向量库与 reranker 都换成假实现，只验证契约：
分数换算、阈值过滤、降序与截断、重排优先、重排不可用时的退回、top_k 上限、失败不外抛。
"""

import pytest
from langchain_core.documents import Document

from src.agents.mcps import rag_mcp_server as rag



def make_doc(source: str, text: str = "片段内容") -> Document:
    return Document(page_content=text, metadata={"source": source})


class FakeVectorStore:
    """记录调用参数的假向量库；results 是 (Document, 余弦距离) 列表。"""

    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.calls: list[dict] = []

    def similarity_search_with_score(self, query, k=4):
        self.calls.append({"query": query, "k": k})
        if self.error is not None:
            raise self.error
        return self.results[:k]


@pytest.fixture()
def store(monkeypatch):
    def _install(fake: FakeVectorStore) -> FakeVectorStore:
        monkeypatch.setattr(rag, "get_vectorstore", lambda: fake)
        return fake

    return _install


@pytest.fixture()
def rerank(monkeypatch):
    """替换 reranker.rerank；不装它时默认返回 None（等价于"重排不可用"）。"""

    def _install(ranked, calls=None):
        def _fake(query, documents):
            if calls is not None:
                calls.append({"query": query, "documents": documents})
            return ranked

        monkeypatch.setattr(rag.reranker, "rerank", _fake)

    monkeypatch.setattr(rag.reranker, "rerank", lambda query, documents: None)
    return _install



def test_score_of_converts_cosine_distance_to_similarity():
    """pgvector 的 <=> 距离 = 1 - 余弦相似度，取值 [0, 2]。"""
    assert rag.score_of(0.0) == 1.0
    assert rag.score_of(0.5) == pytest.approx(0.5)
    assert rag.score_of(1.0) == 0.0
    assert rag.score_of(2.0) == pytest.approx(-1.0)



def test_format_results_drops_below_threshold():
    hits = [
        (make_doc("knowledge_base/请假管理制度.txt", "正文A"), 0.8),
        (make_doc("knowledge_base/考勤制度.txt", "正文B"), 0.3),
        (make_doc("knowledge_base/入职须知.txt", "正文C"), -0.9),
    ]

    out = rag.format_results(hits, min_score=0.4)

    assert "正文A" in out
    assert "正文B" not in out
    assert "正文C" not in out


def test_format_results_keeps_boundary_score():
    """恰好等于阈值保留：阈值语义是"最低可接受"，不是"必须大于"。"""
    out = rag.format_results([(make_doc("s.txt", "边界"), 0.4)], min_score=0.4)

    assert "边界" in out


def test_format_results_sorts_descending_and_truncates():
    hits = [
        (make_doc("low.txt", "低分"), 0.2),
        (make_doc("high.txt", "高分"), 0.9),
        (make_doc("mid.txt", "中分"), 0.5),
    ]

    out = rag.format_results(hits, min_score=0.0, limit=2)

    assert out.index("高分") < out.index("中分")
    assert "低分" not in out  # 先排序再截断，被截掉的一定是最低分
    assert out.count("【片段") == 2


def test_format_results_reports_best_score_when_nothing_passes():
    hits = [
        (make_doc("a.txt"), 0.05),
        (make_doc("b.txt"), 0.01),
    ]

    out = rag.format_results(hits, min_score=0.1, label="相关度")

    assert "未找到相关内容" in out
    assert "最高相关度 0.050" in out
    assert "阈值 0.1" in out
    assert "a.txt" not in out  # 未命中时不泄露任何片段内容


def test_format_results_handles_empty_candidates():
    out = rag.format_results([], min_score=0.1)

    assert "未找到相关内容" in out
    assert "最高" not in out  # 没有任何候选时不编造分数


def test_format_results_includes_label_source_and_content():
    hits = [(make_doc("knowledge_base/费用报销流程.txt", "报销需要发票"), 0.7)]

    out = rag.format_results(hits, min_score=0.1, label="相关度")

    assert "【片段 1】相关度 0.700（来源：knowledge_base/费用报销流程.txt）" in out
    assert "报销需要发票" in out


def test_format_results_falls_back_to_unknown_source():
    doc = Document(page_content="无来源", metadata={})

    assert "来源：未知来源" in rag.format_results([(doc, 0.5)], min_score=0.1)



def test_rerank_result_overrides_vector_order(store, rerank):
    """余弦 top1 是 A，但重排认为 B 更相关时，输出必须以重排为准。"""
    store(
        FakeVectorStore(
            [
                (make_doc("a.txt", "余弦第一"), 0.1),  # 相似度 0.9
                (make_doc("b.txt", "重排第一"), 0.4),  # 相似度 0.6
            ]
        )
    )
    rerank([(1, 0.95), (0, 0.20)])

    out = rag.search_knowledge("问题")

    assert out.index("重排第一") < out.index("余弦第一")
    assert "相关度 0.950" in out


def test_rerank_receives_all_recalled_contents(store, rerank):
    store(
        FakeVectorStore(
            [(make_doc("a.txt", "内容A"), 0.1), (make_doc("b.txt", "内容B"), 0.2)]
        )
    )
    calls: list[dict] = []
    rerank([(0, 0.5)], calls=calls)

    rag.search_knowledge("请假流程")

    assert calls[0]["query"] == "请假流程"
    assert calls[0]["documents"] == ["内容A", "内容B"]


def test_rerank_applies_rerank_threshold(store, rerank):
    """两条都过了余弦阈值，但重排分低于重排阈值时同样丢弃。"""
    store(
        FakeVectorStore(
            [(make_doc("a.txt", "强相关"), 0.1), (make_doc("b.txt", "弱相关"), 0.15)]
        )
    )
    rerank([(0, 0.9), (1, 0.01)])

    out = rag.search_knowledge("问题")

    assert "强相关" in out
    assert "弱相关" not in out


def test_rerank_ignores_out_of_range_index(store, rerank):
    """防御异常响应：越界下标不能让检索崩掉。"""
    store(FakeVectorStore([(make_doc("a.txt", "唯一片段"), 0.1)]))
    rerank([(0, 0.8), (99, 0.7)])

    out = rag.search_knowledge("问题")

    assert "唯一片段" in out
    assert out.count("【片段") == 1



def test_falls_back_to_cosine_with_its_own_threshold(store):
    """重排不可用 -> 用余弦相似度 + 余弦阈值，标签也换成"相似度"。"""
    store(
        FakeVectorStore(
            [
                (make_doc("a.txt", "相关内容"), 0.3),  # 相似度 0.7，过阈值
                (make_doc("b.txt", "无关内容"), 0.72),  # 相似度 0.28，不过阈值
            ]
        )
    )

    out = rag.search_knowledge("问题")

    assert "相关内容" in out
    assert "无关内容" not in out
    assert "相似度 0.700" in out


def test_fallback_ignores_rerank_threshold(store):
    """退回路径用的是余弦阈值（0.4），不该错误地套用重排阈值（0.1）。"""
    store(FakeVectorStore([(make_doc("a.txt", "余弦 0.5 的片段"), 0.5)]))

    out = rag.search_knowledge("问题")

    assert "余弦 0.5 的片段" in out



def test_recall_k_is_used_even_when_top_k_is_small(store):
    """重排要从更多候选里挑，所以召回数取 max(top_k, RECALL_K)。"""
    fake = store(FakeVectorStore([(make_doc("a.txt"), 0.1)]))

    rag.search_knowledge("问题", top_k=1)

    assert fake.calls[0]["k"] == rag.RECALL_K


def test_top_k_is_clamped_to_max(store, rerank):
    fake = store(FakeVectorStore([(make_doc(f"d{i}.txt", f"片段{i}"), 0.1) for i in range(20)]))
    rerank([(i, 0.9 - i * 0.01) for i in range(20)])

    out = rag.search_knowledge("问题", top_k=100)

    assert out.count("【片段") == rag.MAX_TOP_K
    assert fake.calls[0]["k"] == max(rag.MAX_TOP_K, rag.RECALL_K)


def test_top_k_is_clamped_to_at_least_one(store, rerank):
    store(FakeVectorStore([(make_doc("a.txt", "片段"), 0.1)]))
    rerank([(0, 0.9)])

    out = rag.search_knowledge("问题", top_k=0)

    assert out.count("【片段") == 1


def test_search_knowledge_strips_query(store):
    fake = store(FakeVectorStore([]))

    rag.search_knowledge("  请假流程  ")

    assert fake.calls[0]["query"] == "请假流程"



def test_blank_query_never_touches_store_or_rerank(store, rerank):
    fake = store(FakeVectorStore([]))
    calls: list[dict] = []
    rerank([], calls=calls)

    out = rag.search_knowledge("   ")

    assert "查询内容为空" in out
    assert fake.calls == []
    assert calls == []


def test_store_failure_is_converted_to_text(store, rerank):
    """工具异常要收敛成文本：MCP 工具直接抛错会让整轮工具调用失败。"""
    store(FakeVectorStore(error=RuntimeError("connection refused")))

    out = rag.search_knowledge("问题")

    assert out.startswith("检索失败: ")
    assert "connection refused" in out


def test_irrelevant_query_is_filtered_end_to_end(store, rerank):
    """端到端：无关提问的重排最高分（实测 0.00005）应被阈值挡掉。"""
    store(
        FakeVectorStore(
            [
                (make_doc("knowledge_base/考勤制度.txt", "无关片段"), 0.72),
                (make_doc("knowledge_base/入职须知.txt", "也无关"), 0.76),
            ]
        )
    )
    rerank([(0, 0.00005), (1, 0.00002)])

    out = rag.search_knowledge("今天北京的天气怎么样")

    assert "未找到相关内容" in out
    assert "无关片段" not in out
