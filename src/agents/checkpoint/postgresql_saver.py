"""LangGraph 检查点 / Store 在 PostgreSQL 上的持久化。

同时提供**同步**与**异步**两套工厂，分别服务于不同的运行环境：

- 异步（AsyncPostgresSaver / AsyncPostgresStore）：主进程（FastAPI）使用。
  图在事件循环里异步执行，检查点读写也异步 → 不再占用线程池，客户端断开时可以用
  asyncio 原生取消真正中断执行。
  前提：psycopg 的异步连接依赖 loop.add_reader，Windows 上必须跑在
  SelectorEventLoop（见 src/main.py；这也是旧方案被迫用同步图的根因）。

- 同步（PostgresSaver / PostgresStore）：MCP store 服务进程使用。
  它跑在独立常驻进程里，工具是同步函数（FastMCP 派发到工作线程执行），同步 psycopg
  在这里最简单，也不会影响主进程的事件循环。

两边共用同一份连接池参数，避免同步/异步配置漂移。
"""

import os
import threading

from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.store.postgres import PostgresStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool

from src.rag import get_embeddings

load_dotenv()

POOL_MAX_SIZE = int(os.getenv("DB_POOL_MAX_SIZE", "10"))


def _db_uri() -> str:
    """读取 DB_URI；未配置时报出可读错误。

    不内置默认连接串：默认值会带着本地口令进入代码与日志，且失败点会延后到建池阶段。
    """
    uri = os.getenv("DB_URI")
    if not uri:
        raise RuntimeError(
            "未配置 DB_URI，无法连接 PostgreSQL（检查点 / 长期记忆 / 向量库）"
        )
    return uri

# 同步/异步共用的连接参数
_POOL_KWARGS = {
    "autocommit": True,  # 检查点是"写完即生效"的语义，需要自动提交
    "row_factory": dict_row,  # 确保返回字典格式的行
    "options": "-c search_path=public",  # 指定默认模式
}

# Store 的语义检索索引：对 value 字段做 1024 维嵌入（与 SiliconFlow 模型一致）
_STORE_INDEX = {"embed": get_embeddings(), "dims": 1024, "fields": ["value"]}

# 长期记忆的 namespace：constraint 常驻注入不检索，material 按需语义检索
# 两级 namespace 让约束类走 btree 前缀索引，成本与记忆总量无关
CONSTRAINT_NAMESPACE = ("user", "constraint")
MATERIAL_NAMESPACE = ("user", "material")

# RunningSummary 不在 langgraph 安全名单里，显式登记以消除反序列化警告
_EXTRA_MSGPACK_TYPES = [
    ("langmem.short_term.summarization", "RunningSummary"),
]


def _serde() -> JsonPlusSerializer:
    """构造检查点序列化器。

    注意不能改用 saver.with_allowlist()：该方法在"宽松模式（默认 True）"下直接返回原
    对象、不生效，必须在构造时就给定 allowlist。
    """
    return JsonPlusSerializer(allowed_msgpack_modules=list(_EXTRA_MSGPACK_TYPES))



_checkpointer = None
_checkpointer_lock = threading.Lock()


def get_checkpointer() -> PostgresSaver:
    global _checkpointer
    if _checkpointer is None:
        with _checkpointer_lock:
            if _checkpointer is None:
                pool = ConnectionPool(
                    _db_uri(),
                    max_size=POOL_MAX_SIZE,
                    kwargs=_POOL_KWARGS,
                )
                checkpointer = PostgresSaver(pool, serde=_serde())
                checkpointer.setup()
                _checkpointer = checkpointer
    return _checkpointer


_store = None
_store_lock = threading.Lock()


def get_store() -> PostgresStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                pool = ConnectionPool(
                    _db_uri(),
                    max_size=POOL_MAX_SIZE,
                    kwargs=_POOL_KWARGS,
                )
                store = PostgresStore(pool, index=_STORE_INDEX)
                store.setup()
                _store = store
    return _store



_async_pool: AsyncConnectionPool | None = None
_async_checkpointer: AsyncPostgresSaver | None = None
_async_store: AsyncPostgresStore | None = None


async def init_async_persistence() -> None:
    """建立异步连接池并初始化检查点/Store 表（由 FastAPI lifespan 调用一次）。

    不做懒加载 + 双检锁：连接池必须绑定到运行中的事件循环，在 lifespan 里显式初始化
    既避免"第一次请求才建池"的抖动，也让启动失败能立刻暴露出来。
    """
    global _async_pool, _async_checkpointer, _async_store
    if _async_pool is not None:
        return

    pool = AsyncConnectionPool(
        _db_uri(),
        max_size=POOL_MAX_SIZE,
        open=False,  # 显式 open，避免构造函数里隐式开池
        kwargs=_POOL_KWARGS,
    )
    await pool.open()
    # 启动阶段自检，配置错误立刻暴露
    async with pool.connection() as conn:
        await conn.execute("SELECT 1")

    checkpointer = AsyncPostgresSaver(pool, serde=_serde())
    await checkpointer.setup()
    store = AsyncPostgresStore(pool, index=_STORE_INDEX)
    await store.setup()

    _async_pool, _async_checkpointer, _async_store = pool, checkpointer, store


async def close_async_persistence() -> None:
    """关闭异步连接池（应用退出时调用）。"""
    global _async_pool, _async_checkpointer, _async_store
    if _async_pool is None:
        return
    await _async_pool.close()
    _async_pool = _async_checkpointer = _async_store = None


def get_async_pool() -> AsyncConnectionPool:
    """取异步连接池（供业务接口写自定义 SQL，如会话列表）。"""
    if _async_pool is None:
        raise RuntimeError("异步持久化未初始化：请确认 FastAPI lifespan 已执行")
    return _async_pool


def get_async_checkpointer() -> AsyncPostgresSaver:
    if _async_checkpointer is None:
        raise RuntimeError("异步持久化未初始化：请确认 FastAPI lifespan 已执行")
    return _async_checkpointer


def get_async_store() -> AsyncPostgresStore:
    if _async_store is None:
        raise RuntimeError("异步持久化未初始化：请确认 FastAPI lifespan 已执行")
    return _async_store
