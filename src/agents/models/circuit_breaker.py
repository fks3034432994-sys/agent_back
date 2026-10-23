from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


class CircuitOpen(RuntimeError):
    """熔断中：本次调用被直接拒绝，没有真正发往模型服务。"""


class CircuitBreaker:
    """按"连续失败次数"跳闸的熔断器。

    状态变更全在同步方法里完成（没有 await），asyncio 单线程下不会被打断，因此不需要
    额外的锁。
    """

    def __init__(self, fail_threshold: int = 3, reset_timeout: float = 30.0) -> None:
        self.fail_threshold = max(1, int(fail_threshold))
        self.reset_timeout = max(0.0, float(reset_timeout))
        self.state = CLOSED
        self.failures = 0
        self.opened_at = 0.0
        self.probing = False

    def _open_message(self) -> str:
        wait = max(0.0, self.reset_timeout - (time.monotonic() - self.opened_at))
        return (
            f"模型服务暂不可用（连续失败 {self.fail_threshold} 次），"
            f"约 {wait:.0f} 秒后自动重试"
        )

    def _admit(self) -> None:
        """放行则返回，否则抛 CircuitOpen。"""
        if self.state == CLOSED:
            return

        if self.state == OPEN:
            if time.monotonic() - self.opened_at < self.reset_timeout:
                raise CircuitOpen(self._open_message())
            # 冷却结束转半开，只放一个探测请求
            self.state = HALF_OPEN

        if self.probing:
            raise CircuitOpen("模型服务恢复探测中，请稍后重试")
        self.probing = True

    def _record_success(self) -> None:
        self.state = CLOSED
        self.failures = 0
        self.opened_at = 0.0
        self.probing = False

    def _record_failure(self) -> None:
        self.probing = False
        if self.state == HALF_OPEN:
            # 探测失败立即重新打开，不等阈值
            self.state = OPEN
            self.opened_at = time.monotonic()
            return

        self.failures += 1
        if self.failures >= self.fail_threshold:
            self.state = OPEN
            self.opened_at = time.monotonic()

    @asynccontextmanager
    async def guard(self) -> AsyncIterator[None]:
        """包住一次模型调用：进入时判定放行，退出时按结果更新状态。"""
        self._admit()
        try:
            yield
        except asyncio.CancelledError:
            # 客户端断开不计失败，只放掉探测位
            self.probing = False
            raise
        except Exception:
            self._record_failure()
            raise
        else:
            self._record_success()
