"""FastAPI 应用入口。

事件循环（Windows 上必须显式指定）：
- 异步 psycopg（AsyncPostgresSaver / AsyncConnectionPool）依赖 loop.add_reader，
  而 Windows 默认的 ProactorEventLoop 不实现它 → 必须用 SelectorEventLoop；
- 旧方案之所以被迫"Proactor + 同步图 + 线程池"，是因为 MCP 走 stdio 需要创建子进程
  （只有 Proactor 支持）。MCP 改为常驻 HTTP 服务后主进程不再创建子进程，这个死结
  就不存在了（见 agents/mcps/registry.py）。
- 注意 uvicorn 自身的行为：它在 Windows 上会**直接用 loop_factory 选 ProactorEventLoop**，
  绕过全局事件循环策略（uvicorn.loops.asyncio.asyncio_loop_factory）。所以这里不用
  uvicorn.run()，而是自建 Selector 循环后 await server.serve()。

启动依赖：先起 MCP 常驻服务（python scripts/start_mcp_servers.py），再起本服务。
"""

import contextlib
import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.agents.checkpoint import close_async_persistence, init_async_persistence
from src.agents.graph.main_graph import init_agent
from src.agents.mcps.registry import SERVERS, url_of
from src.api.attachments import router as attachments_router
from src.api.chat import router as chat_router
from src.api.knowledge import router as knowledge_router
from src.asyncio_runner import run_async

logger = logging.getLogger(__name__)


def _mcp_hint() -> str:
    return "、".join(f"{name}({url_of(name)})" for name in SERVERS)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动时初始化持久化与 agent，退出时关闭连接池。

    刻意让"连不上 MCP"直接失败启动：否则问题会推迟到用户第一次对话时才暴露。
    """
    await init_async_persistence()
    try:
        await init_agent()
    except Exception as e:
        raise RuntimeError(
            f"连接 MCP 服务失败：{e}\n"
            f"请先启动常驻服务：python scripts/start_mcp_servers.py\n"
            f"当前地址表：{_mcp_hint()}"
        ) from e
    yield
    await close_async_persistence()


app = FastAPI(title="Agent Backend", version="0.1.0", lifespan=lifespan)

# CORS：开发阶段放开所有来源，生产按需收紧
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(knowledge_router)
app.include_router(attachments_router)


@app.get("/api/v1/health")
async def health():
    return {"status": "ok"}


def _run(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    # 应用此前没有 logging 配置，logger.info 会被丢弃，这里给最小可用配置
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    # 不能直接 uvicorn.run()：Windows 上会换成 ProactorEventLoop，异步 psycopg 不可用
    run_async(server.serve())


if __name__ == "__main__":
    _run()
