"""主模型熔断器。

三条不变量：
1. 连续失败达到阈值后**不再真的调用模型**——熔断的意义就是省掉那次"超时 + 重试"的等待；
2. 冷却后只放**一个**探测请求，否则恢复瞬间涌入的请求会变成又一轮雪崩；
3. 客户端断开（CancelledError）不算模型故障——那是用户跑了，不是服务挂了。
"""

import asyncio

import pytest

from src.agents.models.circuit_breaker import (
    CLOSED,
    HALF_OPEN,
    OPEN,
    CircuitBreaker,
    CircuitOpen,
)
from src.asyncio_runner import run_async


class Boom(RuntimeError):
    """模拟模型调用失败。"""


def test_opens_after_threshold_and_stops_calling_model():
    breaker = CircuitBreaker(fail_threshold=3, reset_timeout=60)
    calls: list[int] = []

    async def call() -> None:
        async with breaker.guard():
            calls.append(1)
            raise Boom

    async def main() -> None:
        for _ in range(3):
            with pytest.raises(Boom):
                await call()
        # 第 4 次起直接拒绝，没有真正发往模型
        with pytest.raises(CircuitOpen):
            await call()

    run_async(main())

    assert breaker.state == OPEN
    assert calls == [1, 1, 1]


def test_success_clears_failure_count():
    breaker = CircuitBreaker(fail_threshold=3, reset_timeout=60)

    async def main() -> None:
        for _ in range(2):
            with pytest.raises(Boom):
                async with breaker.guard():
                    raise Boom
        async with breaker.guard():  # 成功一次，计数清零
            pass
        for _ in range(2):
            with pytest.raises(Boom):
                async with breaker.guard():
                    raise Boom
        # 清零后只累计了 2 次，未达阈值
        assert breaker.state == CLOSED
        assert breaker.failures == 2

    run_async(main())


def test_half_open_allows_only_one_probe_and_recovers():
    breaker = CircuitBreaker(fail_threshold=1, reset_timeout=0)

    async def main() -> None:
        with pytest.raises(Boom):
            async with breaker.guard():
                raise Boom
        assert breaker.state == OPEN

        entered = asyncio.Event()
        release = asyncio.Event()

        async def probe() -> None:
            async with breaker.guard():
                entered.set()
                await release.wait()

        task = asyncio.create_task(probe())
        await entered.wait()
        assert breaker.state == HALF_OPEN

        # 探测在飞时后续请求必须被拒绝
        with pytest.raises(CircuitOpen):
            async with breaker.guard():
                pass

        release.set()
        await task
        assert breaker.state == CLOSED

    run_async(main())


def test_probe_failure_reopens_immediately():
    breaker = CircuitBreaker(fail_threshold=1, reset_timeout=0)

    async def main() -> None:
        with pytest.raises(Boom):
            async with breaker.guard():
                raise Boom
        # reset_timeout=0，第二次进入即半开探测；失败要立即重新打开并重新计时
        with pytest.raises(Boom):
            async with breaker.guard():
                raise Boom
        assert breaker.state == OPEN

    run_async(main())


def test_cancellation_is_not_counted_as_failure():
    breaker = CircuitBreaker(fail_threshold=2, reset_timeout=60)

    async def call_forever() -> None:
        async with breaker.guard():
            await asyncio.sleep(10)

    async def main() -> None:
        for _ in range(5):
            task = asyncio.create_task(call_forever())
            await asyncio.sleep(0)  # 让它真正进入 guard
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert breaker.state == CLOSED
        assert breaker.failures == 0

    run_async(main())
