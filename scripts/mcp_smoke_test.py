"""MCP 服务自检：确认常驻服务在线，并列出各自暴露的工具。

Route A 之后 MCP 是独立常驻服务，"服务没起来"是最常见的故障。与其等到第一次对话
才看到一个含糊的连接错误，不如先跑这个脚本定位：哪个端口不通、哪个服务工具列表为空。

用法：
    python scripts/mcp_smoke_test.py            # 端口 + 工具列表检查
    python scripts/mcp_smoke_test.py --agent    # 额外验证主进程能绑定工具并编译图（需数据库）

退出码：0 全部正常；1 有服务不可用。
"""

from __future__ import annotations

import argparse
import asyncio
import socket
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.graph.main_graph import mcp_client  # noqa: E402
from src.agents.mcps.registry import (  # noqa: E402
    CANCEL_TOOL_NAME,
    HOST,
    SERVERS,
    port_of,
    url_of,
)
from src.asyncio_runner import run_async  # noqa: E402


def _port_up(port: int, timeout: float = 1.0) -> bool:
    with socket.socket() as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((HOST, port)) == 0


async def _check_tools(name: str) -> list[str]:
    tools = await mcp_client.get_tools(server_name=name)
    return sorted(t.name for t in tools)


async def main() -> int:
    parser = argparse.ArgumentParser(description="MCP 服务自检")
    parser.add_argument("--agent", action="store_true", help="额外验证 agent 初始化（需数据库可用）")
    args = parser.parse_args()

    failed: list[str] = []
    for name in SERVERS:
        port = port_of(name)
        if not _port_up(port):
            print(f"[自检] {name:<16} 端口 {port} 不通  ({url_of(name)})")
            failed.append(name)
            continue
        try:
            tools = await _check_tools(name)
        except Exception as e:  # noqa: BLE001 - 自检脚本要把原因打出来
            print(f"[自检] {name:<16} 端口通但取工具失败：{e}")
            failed.append(name)
            continue
        exposed = [t for t in tools if t != CANCEL_TOOL_NAME]
        print(f"[自检] {name:<16} 正常，工具 {len(exposed)} 个：{', '.join(exposed)}")

    if args.agent:
        from src.agents.checkpoint import init_async_persistence
        from src.agents.graph.main_graph import init_agent

        try:
            await init_async_persistence()
            await init_agent()
            print("[自检] agent 初始化正常（数据库与工具绑定均可用）")
        except Exception as e:  # noqa: BLE001
            print(f"[自检] agent 初始化失败：{e}")
            failed.append("agent")

    if failed:
        print(f"[自检] 有 {len(failed)} 项异常：{', '.join(failed)}")
        print("[自检] 若服务未启动，请先运行：python scripts/start_mcp_servers.py")
        return 1
    print("[自检] 全部正常")
    return 0


if __name__ == "__main__":
    # 走统一入口：Windows 上普通 asyncio.run 会选到 ProactorEventLoop 而失败
    sys.exit(run_async(main()))
