"""向量化客户端（agents/models/models.py）。

只验证"请求长什么样、批量怎么分批、失败怎么重试"——不发真实网络请求：把
models.requests 换成记录器（不是改全局 requests 模块，避免影响其它测试）。
"""

import os

import pytest

# src.agents.models 是包，实现在包内 models.py，__init__ 只导出模型对象
import src.agents.models.models as models


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}]
        }
        self.text = text

    def json(self):
        return self._payload


class FakeRequests:
    """记录调用并**按顺序**消费 script：FakeResponse 作为响应返回，异常实例则抛出。

    最后一个元素会被复用，方便写"一直失败"和"重试若干次后成功"两种场景。
    """

    def __init__(self, script=None):
        self.script = list(script or [FakeResponse()])
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture()
def fake_requests(monkeypatch):
    """装好假 requests；同时把重试退避换成记录器，避免测试真的 sleep。"""

    def _install(fake: FakeRequests) -> FakeRequests:
        monkeypatch.setattr(models, "requests", fake)
        monkeypatch.setattr(models, "EMBEDDING_MAX_RETRIES", 2)
        monkeypatch.setattr(models, "EMBEDDING_RETRY_BACKOFF", 0.5)
        monkeypatch.setattr(models.time, "sleep", lambda seconds: sleeps.append(seconds))
        return fake

    sleeps: list[float] = []
    _install.sleeps = sleeps
    return _install


def test_embed_text_builds_the_expected_request(fake_requests):
    fake = fake_requests(FakeRequests())

    vector = models.embed_text("你好")

    assert vector == [0.1, 0.2, 0.3]
    call = fake.calls[0]
    assert call["url"] == os.getenv("EMBEDDING_MODEL_URL") + "/embeddings"
    # 批量与单条都走数组形式的同一条路径
    assert call["json"] == {"model": os.getenv("EMBEDDING_MODEL_ID"), "input": ["你好"]}
    assert call["headers"]["Authorization"] == f"Bearer {os.getenv('EMBEDDING_MODEL_API_KEY')}"
    assert call["headers"]["Content-Type"] == "application/json"


def test_embed_text_sets_a_timeout(fake_requests):
    """没有超时时嵌入服务不响应会让检索/入库永久挂住。"""
    fake = fake_requests(FakeRequests())

    models.embed_text("你好")

    assert fake.calls[0]["timeout"] == models.EMBEDDING_TIMEOUT


def test_trailing_slash_in_base_url_is_normalized(fake_requests, monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL_URL", "https://api.example.com/v1/")
    fake = fake_requests(FakeRequests())

    models.embed_text("x")

    assert fake.calls[0]["url"] == "https://api.example.com/v1/embeddings"


def test_embed_texts_returns_one_vector_per_text(fake_requests):
    fake_requests(
        FakeRequests(
            [
                FakeResponse(
                    payload={
                        "data": [
                            {"embedding": [1.0], "index": 0},
                            {"embedding": [2.0], "index": 1},
                        ]
                    }
                )
            ]
        )
    )

    assert models.embed_texts(["a", "b"]) == [[1.0], [2.0]]


def test_embed_texts_aligns_by_index_not_response_order(fake_requests):
    """响应顺序错位时必须按 index 对齐：错位会把 A 的向量存成 B 的，且不会报错。"""
    fake_requests(
        FakeRequests(
            [
                FakeResponse(
                    payload={
                        "data": [
                            {"embedding": [2.0], "index": 1},
                            {"embedding": [1.0], "index": 0},
                        ]
                    }
                )
            ]
        )
    )

    assert models.embed_texts(["a", "b"]) == [[1.0], [2.0]]


def test_embed_texts_falls_back_to_positional_order(fake_requests):
    """老接口/替身可能不带 index，此时按位置顺序对齐。"""
    fake_requests(
        FakeRequests([FakeResponse(payload={"data": [{"embedding": [7.0]}]})])
    )

    assert models.embed_texts(["a"]) == [[7.0]]


def test_embed_texts_raises_when_vectors_are_missing(fake_requests):
    """少返回一条不能静默补空，否则入库会错位且无感知。"""
    fake_requests(FakeRequests([FakeResponse(payload={"data": []})]))

    with pytest.raises(Exception, match="缺少"):
        models.embed_texts(["a", "b"])


def test_embed_texts_of_empty_list_does_not_call_api(fake_requests):
    fake = fake_requests(FakeRequests())

    assert models.embed_texts([]) == []
    assert fake.calls == []


def test_embed_documents_splits_into_batches(fake_requests, monkeypatch):
    """批量化：7 条文本、每批 3 条 -> 3 次请求（3+3+1），而不是 7 次。"""
    monkeypatch.setattr(models, "EMBEDDING_BATCH_SIZE", 3)

    def _response_for(count):
        return FakeResponse(
            payload={
                "data": [
                    {"embedding": [float(i)], "index": i} for i in range(count)
                ]
            }
        )

    fake_requests(
        FakeRequests(
            [_response_for(3), _response_for(3), _response_for(1)]
        )
    )

    vectors = models.SiliconFlowEmbeddings().embed_documents(
        [f"文档{i}" for i in range(7)]
    )

    assert len(vectors) == 7
    # 最后一次脚本元素会被复用，所以断言请求次数与每批的条数
    assert [len(c["json"]["input"]) for c in models.requests.calls] == [3, 3, 1]


def test_embed_documents_without_texts_does_not_call_api(fake_requests, monkeypatch):
    monkeypatch.setattr(models, "EMBEDDING_BATCH_SIZE", 3)
    fake = fake_requests(FakeRequests())

    assert models.SiliconFlowEmbeddings().embed_documents([]) == []
    assert fake.calls == []


def test_embed_query_uses_single_text_request(fake_requests):
    fake = fake_requests(FakeRequests())

    assert models.SiliconFlowEmbeddings().embed_query("问题") == [0.1, 0.2, 0.3]
    assert fake.calls[0]["json"]["input"] == ["问题"]


def test_embed_text_raises_with_response_body(fake_requests):
    fake = fake_requests(FakeRequests([FakeResponse(status_code=401, text="bad key")]))

    with pytest.raises(Exception, match="bad key"):
        models.embed_text("x")

    assert len(fake.calls) == 1  # 4xx 重试无意义，必须立即失败


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_retryable_statuses_are_retried(fake_requests, status):
    fake = fake_requests(
        FakeRequests([FakeResponse(status_code=status, text="busy"), FakeResponse()])
    )

    assert models.embed_text("x") == [0.1, 0.2, 0.3]
    assert len(fake.calls) == 2


def test_retry_backoff_grows(fake_requests):
    fake_requests(
        FakeRequests([FakeResponse(status_code=503, text="busy")])  # 一直失败
    )

    with pytest.raises(Exception, match="重试 2 次后仍失败"):
        models.embed_text("x")

    # 退避 0.5s、1s，前一次等待应短于后一次
    assert fake_requests.sleeps == [0.5, 1.0]


def test_connection_errors_are_retried(fake_requests):
    fake = fake_requests(
        FakeRequests([TimeoutError("timed out"), FakeResponse()])
    )

    assert models.embed_text("x") == [0.1, 0.2, 0.3]
    assert len(fake.calls) == 2


def test_exhausted_retries_report_the_last_error(fake_requests):
    fake_requests(FakeRequests([TimeoutError("timed out")]))

    with pytest.raises(Exception, match="timed out"):
        models.embed_text("x")


def test_no_retry_when_disabled(fake_requests, monkeypatch):
    fake = fake_requests(FakeRequests([FakeResponse(status_code=503, text="busy")]))
    # 必须装在 fixture 之后，fixture 会重置重试次数
    monkeypatch.setattr(models, "EMBEDDING_MAX_RETRIES", 0)

    with pytest.raises(Exception, match="busy"):
        models.embed_text("x")

    assert len(fake.calls) == 1


def test_embed_documents_returns_one_vector_per_text(monkeypatch):
    monkeypatch.setattr(models, "embed_texts", lambda texts: [[float(len(t))] for t in texts])

    embeddings = models.SiliconFlowEmbeddings()

    assert embeddings.embed_documents(["a", "bbb"]) == [[1.0], [3.0]]


def test_embed_query_returns_one_vector(monkeypatch):
    monkeypatch.setattr(models, "embed_texts", lambda texts: [[float(len(t))] for t in texts])

    assert models.SiliconFlowEmbeddings().embed_query("ab") == [2.0]
