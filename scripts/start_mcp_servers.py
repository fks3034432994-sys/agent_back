"""一键启动全部 MCP 常驻服务（HTTP transport）。

Route A 之后 MCP 是**独立常驻服务**，主进程不再用 stdio 拉起子进程，因此开发时需要
单独启动它们（生产环境可由 systemd / docker compose 各自拉起一个服务）。

用法：
    python scripts/start_mcp_servers.py                  # 启动全部
    python scripts/start_mcp_servers.py --only file_server,gui_server
    python scripts/start_mcp_servers.py --timeout 180    # 等待就绪的上限（秒）

Ctrl+C 结束全部子进程；Windows 上用 taskkill /T 连带子进程树。
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.mcps.registry import HOST, SERVERS, port_of  # noqa: E402


def _wait_ready(port: int, timeout: float) -> float | None:
    """等待端口可连接；返回耗时（秒），超时返回 None。"""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        with socket.socket() as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((HOST, port)) == 0:
                return time.monotonic() - start
        time.sleep(0.3)
    return None


def _terminate(proc: subprocess.Popen, name: str) -> None:
    """结束一个服务进程（含其子进程树）。"""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
        )
    else:
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    print(f"[启动器] 已结束 {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="启动 MCP 常驻服务（HTTP transport）")
    parser.add_argument("--only", default="", help="只启动指定服务（逗号分隔）")
    parser.add_argument("--timeout", type=float, default=120.0, help="等待就绪的上限秒数")
    args = parser.parse_args()

    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(SERVERS)
    unknown = [n for n in names if n not in SERVERS]
    if unknown:
        print(f"[启动器] 未知服务：{', '.join(unknown)}；可选：{', '.join(SERVERS)}")
        return 2

    # 显式传 PYTHONPATH，保证两种启动方式行为一致
    env = {
        **os.environ,
        "PYTHONPATH": str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }

    procs: list[tuple[str, subprocess.Popen, int]] = []
    for name in names:
        port = port_of(name)
        proc = subprocess.Popen(
            [sys.executable, "-m", SERVERS[name].module],
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        procs.append((name, proc, port))
        print(f"[启动器] 启动 {name}（pid={proc.pid}，端口 {port}）")

    try:
        failed: list[str] = []
        for name, proc, port in procs:
            cost = _wait_ready(port, args.timeout)
            if proc.poll() is not None:
                failed.append(name)
                print(f"[启动器] {name} 启动失败并已退出（退出码 {proc.returncode}），请单独运行它查看报错")
            elif cost is None:
                failed.append(name)
                print(f"[启动器] {name} 在 {args.timeout:.0f}s 内未就绪（端口 {port}）")
            else:
                print(f"[启动器] {name} 就绪（{cost:.1f}s）")

        if failed:
            print(f"[启动器] 有服务未就绪：{', '.join(failed)}")
            return 1

        print("[启动器] 全部服务已就绪，按 Ctrl+C 结束")
        while True:
            for name, proc, _port in procs:
                code = proc.poll()
                if code is not None:
                    print(f"[启动器] 服务 {name} 已退出（退出码 {code}）")
                    return 1
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[启动器] 收到中断，正在结束服务")
        return 0
    finally:
        for name, proc, _port in procs:
            _terminate(proc, name)


if __name__ == "__main__":
    sys.exit(main())
