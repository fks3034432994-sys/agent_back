"""异步入口的统一运行方式（Windows 上必须用 SelectorEventLoop）。

异步 psycopg（AsyncPostgresSaver / AsyncConnectionPool）依赖 loop.add_reader，而 Windows
默认的 ProactorEventLoop 不实现它，会直接报：

    Psycopg cannot use the 'ProactorEventLoop' to run in async mode.

因此本项目里所有"自己创建事件循环"的入口（Web 服务、自检脚本、模块调试入口）统一走
run_async()，避免以后新增脚本时再踩一遍。

另注：uvicorn 在 Windows 上会用 loop_factory 直接选 ProactorEventLoop（绕过全局事件循环
策略），所以 Web 服务也不能直接 uvicorn.run()，而要用 run_async(server.serve())。
"""

import asyncio
import sys
from collections.abc import Coroutine
from typing import Any


def run_async(main_coro: Coroutine[Any, Any, Any]) -> Any:
    """运行一个异步入口协程；Windows 上强制使用 SelectorEventLoop。"""
    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            return runner.run(main_coro)
    return asyncio.run(main_coro)
