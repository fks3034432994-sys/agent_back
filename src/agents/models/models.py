import os
import time
from src.agents.utils import ActionParam
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import requests
from langchain_core.embeddings import Embeddings

from .circuit_breaker import CircuitBreaker

load_dotenv()

click_model = init_chat_model(
    model=os.getenv("TARS_MODEL_ID"),
    model_provider="openai",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=os.getenv("TARS_MODEL_API_KEY")
).with_structured_output(ActionParam)

check_model = init_chat_model(
    model=os.getenv("TARS_MODEL_ID"),
    model_provider="openai",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=os.getenv("TARS_MODEL_API_KEY")
)

task_planning_model = init_chat_model(
    model=os.getenv("TASK_PLANNING_MODEL_ID"),
    model_provider="openai",
    base_url=os.getenv("SILI_CONFLOW_URL"),
    api_key=os.getenv("SILI_CONFLOW_API_KEY")
)

# 不显式传就吃 SDK 默认的 600s 超时 + 重试 2 次，模型挂住会占满会话锁
# 流式下这是两次数据之间的超时，60s 不会打断长回答
MAIN_MODEL_TIMEOUT = float(os.getenv("MAIN_MODEL_TIMEOUT", "60"))
MAIN_MODEL_MAX_RETRIES = int(os.getenv("MAIN_MODEL_MAX_RETRIES", "2"))

# 熔断：连续失败达阈值直接拒绝，冷却后放一个探测请求；计数是进程内的
MAIN_MODEL_FAIL_THRESHOLD = int(os.getenv("MAIN_MODEL_FAIL_THRESHOLD", "3"))
MAIN_MODEL_RESET_SECONDS = float(os.getenv("MAIN_MODEL_RESET_SECONDS", "30"))

main_model = init_chat_model(
    model=os.getenv("MAIN_MODEL_ID"),
    model_provider="openai",
    base_url=os.getenv("MAIN_MODEL_URL"),
    api_key=os.getenv("MAIN_MODEL_API_KEY"),
    # 显式开启流式：否则拿不到 AIMessageChunk，前端就没有逐字效果
    streaming=True,
    timeout=MAIN_MODEL_TIMEOUT,
    max_retries=MAIN_MODEL_MAX_RETRIES,
)

# 主模型调用的熔断器（用法见 main_graph.main_model_call 的 guard）
main_model_breaker = CircuitBreaker(
    fail_threshold=MAIN_MODEL_FAIL_THRESHOLD,
    reset_timeout=MAIN_MODEL_RESET_SECONDS,
)

# 超时必须有：嵌入服务不响应，同步 HTTP 会永久占住线程
EMBEDDING_TIMEOUT = float(os.getenv("EMBEDDING_TIMEOUT", "60"))

# 只重试连接失败/超时/5xx/429，4xx 重试无意义
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "2"))
EMBEDDING_RETRY_BACKOFF = float(os.getenv("EMBEDDING_RETRY_BACKOFF", "0.5"))
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

# 批量请求把 N 次 RTT 压成 N/批次
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))


def _parse_embeddings(payload: dict, count: int) -> list[list[float]]:
    """把响应里的 data 按 index 对齐成与入参同序的向量列表。

    按 index 对齐而不是按返回顺序：批量请求时顺序错位会把 A 文档的向量存成 B 的，
    而且不会报错，属于最难查的一类 bug。响应没有 index 时退回位置顺序。
    """
    items = (payload or {}).get("data") or []
    vectors: list[list[float] | None] = [None] * count
    for position, item in enumerate(items):
        index = item.get("index", position)
        if not isinstance(index, int) or not 0 <= index < count:
            continue
        vectors[index] = [float(value) for value in item["embedding"]]

    missing = [i for i, vector in enumerate(vectors) if vector is None]
    if missing:
        raise Exception(f"Embedding API error: 响应缺少第 {missing} 条向量（共请求 {count} 条）")
    return vectors


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量获取文本向量（一次请求多条），返回顺序与入参严格对应。"""
    if not texts:
        return []

    url = os.getenv("EMBEDDING_MODEL_URL", "").rstrip("/") + "/embeddings"
    headers = {
        "Authorization": f"Bearer {os.getenv('EMBEDDING_MODEL_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {"model": os.getenv("EMBEDDING_MODEL_ID"), "input": list(texts)}

    last_message = ""
    for attempt in range(EMBEDDING_MAX_RETRIES + 1):
        if attempt:
            time.sleep(EMBEDDING_RETRY_BACKOFF * 2 ** (attempt - 1))
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=EMBEDDING_TIMEOUT
            )
        except Exception as exc:  # noqa: BLE001 - 连接失败/超时都值得重试
            last_message = f"{type(exc).__name__}: {exc}"
            continue

        if response.status_code == 200:
            return _parse_embeddings(response.json(), len(texts))
        if response.status_code not in _RETRYABLE_STATUS:
            raise Exception(f"Embedding API error: {response.text}")
        last_message = f"HTTP {response.status_code}: {response.text}"

    raise Exception(
        f"Embedding API error: 重试 {EMBEDDING_MAX_RETRIES} 次后仍失败（{last_message}）"
    )


def embed_text(text: str) -> list[float]:
    """使用硅基流动的 API 获取单条文本的向量。"""
    return embed_texts([text])[0]


class SiliconFlowEmbeddings(Embeddings):
    """硅基流动的嵌入适配器：按 EMBEDDING_BATCH_SIZE 分批，一次请求多条。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入文档"""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            vectors.extend(embed_texts(texts[start : start + EMBEDDING_BATCH_SIZE]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """嵌入单个查询文本"""
        return embed_text(text)


if __name__ == "__main__":
    vector = embed_text("测试文本")
    print(f"向量维度: {len(vector)}")
    print(f"前5个值: {vector[:5]}")