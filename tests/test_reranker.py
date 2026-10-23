"""Rerank 客户端（src/rag/reranker.py）。

不联网：把 reranker.requests 换成记录器（不是改全局 requests 模块，避免影响其它测试）。
重点覆盖 fail-open 契约——任何失败都必须返回 None，由调用方退回向量检索顺序。
"""

import pytest

import src.rag.reranker as reranker


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {
            "results": [{"index": 1, "relevance_score": 0.9}]
        }
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeRequests:
    def __init__(self, response=None, error=None):
        self.response = response or FakeResponse()
        self.error = error
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture()
def http(monkeypatch):
    """装好假 requests 与一套可用的默认配置（与 conftest 的占位环境变量一致）。"""
    monkeypatch.setenv("EMBEDDING_MODEL_URL", "https://api.example.com/v1")
    monkeypatch.setenv("EMBEDDING_MODEL_API_KEY", "test-key")
    for key in ("RAG_RERANK_URL", "RAG_RERANK_API_KEY", "RAG_RERANK_MODEL", "RAG_RERANK_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)

    def _install(fake: FakeRequests) -> FakeRequests:
        monkeypatch.setattr(reranker, "requests", fake)
        return fake

    return _install



def test_rerank_returns_pairs_sorted_by_score(http):
    fake = http(
        FakeRequests(
            FakeResponse(
                payload={
                    "results": [
                        {"index": 3, "relevance_score": 0.12},
                        {"index": 7, "relevance_score": 0.87},
                        {"index": 1, "relevance_score": 0.45},
                    ]
                }
            )
        )
    )

    out = reranker.rerank("问题", ["a", "b", "c"])

    assert out == [(7, 0.87), (1, 0.45), (3, 0.12)]
    assert len(fake.calls) == 1


def test_rerank_builds_the_expected_request(http):
    fake = http(FakeRequests())

    reranker.rerank("请假几天", ["片段一", "片段二"])

    call = fake.calls[0]
    # rerank 与 embeddings 同源，端点由 EMBEDDING_MODEL_URL 推导
    assert call["url"] == "https://api.example.com/v1/rerank"
    assert call["headers"]["Authorization"] == "Bearer test-key"
    assert call["json"] == {
        "model": reranker.DEFAULT_MODEL,
        "query": "请假几天",
        "documents": ["片段一", "片段二"],
        # 显式要全部候选：最强候选之外的分数在"未命中"提示里还要用
        "top_n": 2,
    }


def test_rerank_uses_configured_model_and_timeout(http, monkeypatch):
    monkeypatch.setenv("RAG_RERANK_MODEL", "custom/reranker")
    monkeypatch.setenv("RAG_RERANK_TIMEOUT", "5")
    fake = http(FakeRequests())

    reranker.rerank("q", ["d"])

    assert fake.calls[0]["json"]["model"] == "custom/reranker"
    assert fake.calls[0]["timeout"] == 5.0


def test_rerank_default_timeout_is_bounded(http):
    """必须带超时：rerank 挂住会让整轮工具调用一直卡着。"""
    fake = http(FakeRequests())

    reranker.rerank("q", ["d"])

    assert fake.calls[0]["timeout"] == 30.0


def test_explicit_url_and_key_win(http, monkeypatch):
    monkeypatch.setenv("RAG_RERANK_URL", "https://rerank.example.com/v2/")
    monkeypatch.setenv("RAG_RERANK_API_KEY", "own-key")
    fake = http(FakeRequests())

    reranker.rerank("q", ["d"])

    assert fake.calls[0]["url"] == "https://rerank.example.com/v2/rerank"  # 末尾斜杠被规整
    assert fake.calls[0]["headers"]["Authorization"] == "Bearer own-key"



def test_rerank_returns_none_when_model_disabled(http, monkeypatch):
    monkeypatch.setenv("RAG_RERANK_MODEL", "")
    fake = http(FakeRequests())

    assert reranker.rerank("q", ["d"]) is None
    assert fake.calls == []  # 关闭时不应发出请求


def test_rerank_returns_none_without_api_key(http, monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL_API_KEY", raising=False)
    fake = http(FakeRequests())

    assert reranker.rerank("q", ["d"]) is None
    assert fake.calls == []


def test_rerank_returns_none_without_documents(http):
    fake = http(FakeRequests())

    assert reranker.rerank("q", []) is None
    assert fake.calls == []


def test_rerank_returns_none_on_http_error(http):
    http(FakeRequests(FakeResponse(status_code=401, text="invalid api key")))

    assert reranker.rerank("q", ["d"]) is None


def test_rerank_returns_none_on_request_exception(http):
    http(FakeRequests(error=TimeoutError("timed out")))

    assert reranker.rerank("q", ["d"]) is None


def test_rerank_returns_none_on_invalid_json(http):
    http(FakeRequests(FakeResponse(payload=ValueError("not json"))))

    assert reranker.rerank("q", ["d"]) is None


@pytest.mark.parametrize(
    "payload",
    [
        {},                                                  # 没有 results
        {"results": []},                                     # 空结果
        {"results": [{"index": 0}]},                         # 缺分数
        {"results": [{"relevance_score": 0.5}]},             # 缺下标
        {"results": [{"index": "x", "relevance_score": 0.5}]},  # 下标不可解析
    ],
)
def test_rerank_returns_none_on_unparsable_payload(http, payload):
    """任何解析不了的响应都当成"不可用"，而不是让异常冒到检索层。"""
    http(FakeRequests(FakeResponse(payload=payload)))

    assert reranker.rerank("q", ["d"]) is None
