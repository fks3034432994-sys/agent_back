"""shell / python 代码执行工具（MCP 服务）。

与旧版的两点区别：
1. HTTP 常驻服务：不再由主进程以 stdio 拉起（见 mcps/registry.py 的说明）；
2. **可取消**：旧版用 subprocess.run(timeout=...)，一旦跑起来就无法中断，客户端点
   "停止"只能粗暴地杀掉整个 MCP 进程。现在改成 Popen + 轮询 + 取消信号，收到取消就
   终止子进程树。

为什么必须用"轮询 + 信号"而不是直接靠 asyncio 取消：本工具是**同步函数**，FastMCP
会把它派发到工作线程执行（run_in_thread），而 asyncio 无法中断线程里阻塞的同步代码。
所以取消只能由工具自己在等待过程中观察信号（见 mcps/cancel_registry.py）。
"""

import contextlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# 先补 sys.path：直接运行本文件时 sys.path[0] 是 mcps/，否则找不到 src 包
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from fastmcp import FastMCP

from src.agents.mcps.cancel_registry import CancelRegistry, attach_cancel_tool

mcp = FastMCP("python_and_shell")

# 可取消任务注册表 + cancel_task（内部工具，只由主进程调用）
REGISTRY = CancelRegistry()
attach_cancel_tool(mcp, REGISTRY)

# 命令白名单，为空则允许所有；生产环境建议限定
ALLOWED_COMMANDS = ["ls", "pwd", "python", "pip", "where"]

# 取消/超时检查间隔，越小响应越快
POLL_INTERVAL = 0.5


def _kill_tree(proc: subprocess.Popen) -> None:
    """终止进程及其子进程树。

    Windows 用 taskkill /T（shell=True 起的命令会再派生一层子进程，只杀 shell 不管用）；
    POSIX 用进程组；随后把管道读干净，避免留下半读的管道。
    """
    if proc.poll() is None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
            )
        else:
            import signal

            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)
    with contextlib.suppress(Exception):
        proc.communicate(timeout=5)


def _format_result(returncode: int, out: str, err: str, label: str) -> str:
    if returncode == 0:
        output = out or ""
        if err:
            output += "\n[stderr]\n" + err
        return output or f"({label}执行成功，无输出)"
    return f"{label}返回非零码 ({returncode}):\n{err or out}"


def _wait_cancellable(proc: subprocess.Popen, cancel, timeout: int, label: str) -> str:
    """等待进程结束，期间周期性检查取消信号与超时。

    返回给模型的文本（正常输出 / 取消 / 超时）。
    """
    deadline = time.monotonic() + timeout
    while True:
        if cancel is not None and cancel.is_set():
            _kill_tree(proc)
            return f"已取消：用户中断了本次{label}执行"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _kill_tree(proc)
            return f"{label}执行超时（超过 {timeout} 秒）"
        try:
            out, err = proc.communicate(timeout=min(POLL_INTERVAL, remaining))
            return _format_result(proc.returncode, out, err, label)
        except subprocess.TimeoutExpired:
            continue  # 未结束：回到循环顶部检查取消/超时


@mcp.tool()
def run_shell_command(command: str, timeout: int = 60, task_id: str = "") -> str:
    """
    执行 shell 命令（独立子进程，带超时，可被用户中断）。

    参数:
        command: 要执行的命令
        timeout: 超时时间（秒），默认 60
        task_id: 内部参数，由主进程注入用于取消，模型无需填写

    返回:
        命令的输出结果或错误信息
    """
    if ALLOWED_COMMANDS:
        cmd_name = command.split()[0] if command.split() else ""
        if cmd_name not in ALLOWED_COMMANDS:
            return f"命令 '{cmd_name}' 不在允许列表中"

    cancel = REGISTRY.register(task_id)
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return _wait_cancellable(proc, cancel, timeout, "命令")
    except Exception as e:
        return f"执行失败：{e}"
    finally:
        REGISTRY.unregister(task_id)


@mcp.tool()
def run_python_code(code: str, timeout: int = 30, task_id: str = "") -> str:
    """
    在独立进程中执行 Python 代码，捕获输出，带超时，可被用户中断。

    参数:
        code: 要执行的 Python 代码字符串
        timeout: 超时时间（秒），默认 30
        task_id: 内部参数，由主进程注入用于取消，模型无需填写

    返回:
        代码的输出结果或错误信息
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        tmp_path = f.name

    cancel = REGISTRY.register(task_id)
    try:
        proc = subprocess.Popen(
            [sys.executable, tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return _wait_cancellable(proc, cancel, timeout, "代码")
    except Exception as e:
        return f"执行失败：{e}"
    finally:
        REGISTRY.unregister(task_id)
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


if __name__ == "__main__":
    from src.agents.mcps.registry import PYTHON_SERVER, serve

    serve(mcp, PYTHON_SERVER)
