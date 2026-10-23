"""MCP 服务的地址表与统一启动入口。

Route A：MCP 从"主进程的 stdio 子进程"改为**独立常驻的 HTTP 服务**。

为什么这么改：
1. 主进程不再创建子进程 → Windows 上摆脱"MCP stdio 需要 ProactorEventLoop、异步
   psycopg 需要 SelectorEventLoop"的死结（两者不可共存于同一循环），全链路才能真正
   异步化；旧方案只能把同步图塞进线程池来绕开（见 src/main.py 的说明）。
2. langchain-mcp-adapters 每次工具调用都会新建一个 session，stdio 下等价于每次调用
   重新起一个进程（进程启动 + 重复 import 的成本，GUI 服务还要连带 import torch）；
   HTTP 常驻后这项开销完全消失。
3. 代价有两处，都已做了对应改造：
   - 服务生命周期不再由主进程托管 → scripts/start_mcp_servers.py 一键拉起；
   - 取消机制不能再"按 PID 杀进程"（常驻服务被杀会连带杀死其它会话正在执行的任务）
     → 改为服务内协作式取消，见 cancel_registry.py。

服务名沿用原有字符串：store_mcp_sever 是历史拼写，保持不动——改名要同步改端口映射、
启动脚本与文档，收益为零。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

FILE_SERVER = "file_server"
GUI_SERVER = "gui_server"
PYTHON_SERVER = "python_server"
RAG_SERVER = "rag_server"
STORE_SERVER = "store_mcp_sever"

HOST = os.getenv("MCP_HOST", "127.0.0.1")
# fastmcp 的 streamable HTTP 默认路径（见 fastmcp.settings.streamable_http_path）
PATH = os.getenv("MCP_PATH", "/mcp")

# cancel_task 由服务自己调用，不暴露给模型
CANCEL_TOOL_NAME = "cancel_task"


@dataclass(frozen=True)
class ServerSpec:
    """一个 MCP 服务的模块与默认端口。"""

    module: str
    port: int


SERVERS: dict[str, ServerSpec] = {
    FILE_SERVER: ServerSpec("src.agents.mcps.file_mcp_server", 8761),
    PYTHON_SERVER: ServerSpec("src.agents.mcps.python_and_shell_mcp_server", 8762),
    RAG_SERVER: ServerSpec("src.agents.mcps.rag_mcp_server", 8763),
    STORE_SERVER: ServerSpec("src.agents.mcps.store_mcp_sever", 8764),
    GUI_SERVER: ServerSpec("src.agents.mcps.gui_mcp_server", 8765),
}


def port_of(name: str) -> int:
    """服务端口：环境变量 MCP_PORT_<服务名大写> 优先，便于一台机器上跑多套环境。"""
    return int(os.getenv(f"MCP_PORT_{name.upper()}", SERVERS[name].port))


def url_of(name: str) -> str:
    """服务的 streamable HTTP 端点地址（主进程客户端与服务自身都用它）。"""
    return f"http://{HOST}:{port_of(name)}{PATH}"


def serve(mcp: Any, name: str) -> None:
    """以 HTTP transport 常驻运行一个 MCP 服务（各服务 __main__ 的统一入口）。

    参数:
        mcp: FastMCP 实例
        name: SERVERS 中的服务名
    """
    if name not in SERVERS:
        raise ValueError(f"未知的 MCP 服务名：{name}（可选：{', '.join(SERVERS)}）")
    port = port_of(name)
    print(f"[mcp] {name} 启动，监听 {url_of(name)}", file=sys.stderr, flush=True)
    mcp.run(transport="http", host=HOST, port=port)
