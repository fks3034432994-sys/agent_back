"""项目量化指标基准。

覆盖六项指标，其中图片引用可完全离线测，其余按需连服务（见各函数 docstring）：
  1. bench_mcp_roundtrip   MCP 工具调用往返开销              [需 python_server]
  2. bench_cancel_latency  协作式取消延迟                    [需 python_server]
  3. bench_image_ref       图片引用存储 vs 全部内联           [离线]
  4. bench_compression     上下文压缩前后的 token 下降比例     [需 LLM API]
  5. bench_retrieval       知识库检索命中率                  [需 PG + embedding API]
  6. bench_e2e_latency     端到端延迟                        [需 PG + 5 个 MCP + LLM API]

需要 MCP 服务的几项会自己拉起服务、跑完关掉；已在运行的端口则直接复用、不动它。
测量原则：直接调用被测代码自己的函数（_resolve_images / build_summarize_node /
search_knowledge / run_shell_command），不另写一份模拟逻辑，否则测的是脚本而不是项目。
README 的「实测数据」一节即出自本脚本。

用法：
    uv run python scripts/bench_metrics.py                     # 全部
    uv run python scripts/bench_metrics.py image               # 只跑某一项
    uv run python scripts/bench_metrics.py roundtrip,cancel    # 顺带跑（逗号分隔）
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import random
import re
import socket
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # 直接运行本文件时 sys.path[0] 是 scripts/

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

# 图像 token 按固定 tokens_per_image 计（默认 85），不数 base64 字符
IMAGE_TOKENS_PER_BLOCK = 85


def _port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _wait_ports(ports: dict[str, int], timeout: float) -> list[str]:
    """等待所有端口可连接，返回未就绪的服务名。"""
    pending = dict(ports)
    deadline = time.monotonic() + timeout
    while pending and time.monotonic() < deadline:
        for name, port in list(pending.items()):
            if _port_open(port):
                del pending[name]
        if pending:
            time.sleep(0.5)
    return list(pending)


def _start_servers(names: list[str]) -> list[subprocess.Popen]:
    """按 registry 的服务表拉起指定 MCP 常驻服务，返回进程句柄。

    端口与服务名都取自 registry，不在这里另写一份地址表；已在运行的服务直接复用，
    也不在收尾时关掉它（那可能是用户自己起的）。
    """
    from src.agents.mcps import registry

    env = {
        **os.environ,
        "PYTHONPATH": str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    procs: list[subprocess.Popen] = []
    ports: dict[str, int] = {}
    for name in names:
        port = registry.port_of(name)
        if _port_open(port):
            print(f"    {name} 已在运行（端口 {port}），复用")
            continue
        procs.append(subprocess.Popen(
            [sys.executable, "-m", registry.SERVERS[name].module],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ))
        ports[name] = port
        print(f"    启动 {name}（端口 {port}）")

    not_ready = _wait_ports(ports, timeout=180)
    if not_ready:
        _stop_servers(procs)
        raise RuntimeError(f"这些服务 180s 内未就绪：{', '.join(not_ready)}")
    return procs


def _stop_servers(procs: list[subprocess.Popen]) -> None:
    for proc in procs:
        if proc.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
            else:
                proc.terminate()
    if procs:
        print(f"    已关闭 {len(procs)} 个自启的 MCP 服务")


# 必须在 shell 白名单内，且要跑几秒以上，否则取消信号容易撞上命令自然结束
ROUNDTRIP_COMMAND = "where cmd"
LONG_COMMAND = "where /r C:\\ *.sys"

ROUNDTRIP_SAMPLES = 20
CANCEL_SAMPLES = 8

# 取消前等待必须加抖动：固定值会与轮询边界相位锁定，把延迟系统性低估
CANCEL_WARMUP = 1.5


def _mcp_client(server_name: str):
    """按 registry 的地址表构造单服务 MCP 客户端。"""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    from src.agents.mcps import registry

    return MultiServerMCPClient({
        server_name: {
            "transport": "streamable_http",
            "url": registry.url_of(server_name),
            "timeout": 60,
        }
    })


async def _server_tools(server_name: str) -> dict[str, Any]:
    client = _mcp_client(server_name)
    return {tool.name: tool for tool in await client.get_tools(server_name=server_name)}


def _p95(sorted_samples: list[float]) -> float:
    return sorted_samples[max(0, int(len(sorted_samples) * 0.95) - 1)]


async def _measure_roundtrip(server_name: str) -> None:
    tools = await _server_tools(server_name)
    shell = tools["run_shell_command"]

    # 预热：首次调用含建连与进程启动
    await shell.ainvoke({"command": ROUNDTRIP_COMMAND, "timeout": 10})

    samples: list[float] = []
    for _ in range(ROUNDTRIP_SAMPLES):
        start = time.perf_counter()
        await shell.ainvoke({"command": ROUNDTRIP_COMMAND, "timeout": 10})
        samples.append((time.perf_counter() - start) * 1000)

    samples.sort()
    print(
        f"[往返开销] 跨进程 MCP 调用（{ROUNDTRIP_SAMPLES} 次 `{ROUNDTRIP_COMMAND}`）："
        f"中位数 {statistics.median(samples):.1f} ms / P95 {_p95(samples):.1f} ms / "
        f"最小 {samples[0]:.1f} ms"
    )


async def _measure_cancel(server_name: str) -> None:
    from src.agents.mcps.python_and_shell_mcp_server import POLL_INTERVAL

    tools = await _server_tools(server_name)
    shell, cancel = tools["run_shell_command"], tools["cancel_task"]

    end_to_end: list[float] = []
    detect_wait: list[float] = []
    for i in range(CANCEL_SAMPLES):
        task_id = f"bench-{os.getpid()}-{i}"
        task = asyncio.create_task(
            shell.ainvoke({"command": LONG_COMMAND, "timeout": 300, "task_id": task_id})
        )
        # 抖动等待，避免相位锁定
        await asyncio.sleep(CANCEL_WARMUP + random.uniform(0, POLL_INTERVAL))

        start = time.perf_counter()
        reply = await cancel.ainvoke({"task_id": task_id})  # 置位 Event，瞬时返回
        cancel_rtt = (time.perf_counter() - start) * 1000
        result = await task  # 等工具自己发现信号并终止进程树
        elapsed = (time.perf_counter() - start) * 1000

        text = result[0]["text"]
        if not text.startswith("已取消"):
            # 命令在等待期内跑完，cancel 会撞上 unregister，样本无效
            print(f"  第 {i + 1} 次：样本无效（命令已自然结束：{text[:40]}），跳过")
            continue

        end_to_end.append(elapsed)
        detect_wait.append(elapsed - cancel_rtt)
        print(
            f"  第 {i + 1} 次：端到端 {elapsed:.0f} ms"
            f" = cancel 往返 {cancel_rtt:.0f} ms + 服务端检测 {elapsed - cancel_rtt:.0f} ms"
            f" | {reply[0]['text']}"
        )

    if not end_to_end:
        print("[取消延迟] 全部样本无效，无法统计")
        return

    end_to_end.sort()
    detect_wait.sort()
    print(
        f"[取消延迟] 端到端 中位数 {statistics.median(end_to_end):.0f} ms / "
        f"P95 {_p95(end_to_end):.0f} ms / 最小 {end_to_end[0]:.0f} ms"
        f"（{len(end_to_end)} 次有效采样）"
    )
    print(
        f"[取消延迟] 只看服务端检测等待：均值 {statistics.fmean(detect_wait):.0f} ms / "
        f"中位数 {statistics.median(detect_wait):.0f} ms / 最大 {detect_wait[-1]:.0f} ms；"
        f"轮询间隔 {POLL_INTERVAL}s，理论期望为 0~{POLL_INTERVAL * 1000:.0f} ms "
        f"均匀分布（均值 250 ms）"
    )


def _with_python_server(measure) -> None:
    """拉起 python_server（或复用已在跑的），跑一次测量，再关掉自启的那个。"""
    from src.agents.mcps import registry

    procs = _start_servers([registry.PYTHON_SERVER])
    try:
        measure(registry.PYTHON_SERVER)
    finally:
        _stop_servers(procs)


def bench_mcp_roundtrip() -> None:
    """MCP 工具调用往返开销：常驻 HTTP 服务下每次工具调用的固定跨进程成本。"""
    from src.asyncio_runner import run_async

    _with_python_server(lambda name: run_async(_measure_roundtrip(name)))


def bench_cancel_latency() -> None:
    """协作式取消延迟：从发出取消到子进程被终止、工具返回的耗时。

    端到端由两段组成，分开记录才能看出服务端的真实响应粒度：
      - `cancel_task` 这一次 HTTP 往返（与往返开销同量级，与轮询无关）；
      - 服务端检测到 Event 并终止进程树的等待（受 POLL_INTERVAL 支配）。
    """
    from src.asyncio_runner import run_async

    _with_python_server(lambda name: run_async(_measure_cancel(name)))


def _payload_chars(messages: list[Any]) -> int:
    """消息序列化成 JSON 的字符数，近似实际请求体积。"""
    total = 0
    for message in messages:
        content = message.content
        if isinstance(content, str):
            total += len(content)
        else:
            total += len(json.dumps(content, ensure_ascii=False))
    return total


def _real_tokens(messages: list[Any]) -> int:
    """文本按真实分词器计数、图像块按 IMAGE_TOKENS_PER_BLOCK 固定计的总 token。

    为什么不用 count_tokens_approximately 一把梭：它按 ~4 字符/token 估，中文一个字
    常占 1.5~2 token，整体低估约 4 倍。而在「内联图 vs 引用图」的对比里，被低估的是
    **两边共有**的中文文本 —— 分母被压小、图像占比被抬高，省下的百分比会虚高。
    """
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for message in messages:
        content = message.content
        blocks = (
            [{"type": "text", "text": content}]
            if isinstance(content, str)
            else content
        )
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image_url":
                total += IMAGE_TOKENS_PER_BLOCK
            elif block.get("type") == "text":
                total += len(enc.encode(block.get("text") or ""))
    return total


def _image_info(name: str, read_bytes) -> tuple[int, int, str]:
    """返回 (原始字节数, base64 字符数, 尺寸描述)。"""
    from PIL import Image

    data = read_bytes(name) or b""
    b64_len = len(base64.b64encode(data).decode())
    with Image.open(io.BytesIO(data)) as im:
        size = f"{im.size[0]}x{im.size[1]}"
    return len(data), b64_len, size


def bench_image_ref() -> None:
    """量化「图片只存引用、历史轮降级为占位」省下的上下文体积。

    直接调用项目真实的 _resolve_images（本轮内联 base64、更早的降级成占位），
    对照组则把所有轮次的引用都内联，模拟「不做引用降级」的写法。
    """
    from langchain.messages import AIMessage, HumanMessage
    from langchain_core.messages.utils import count_tokens_approximately

    from src.agents.graph.main_graph import _load_image_block, _resolve_images
    from src.agents.mcps.upload_store import IMAGE_EXTS, UPLOAD_DIR, read_upload_bytes

    names = sorted(
        (p.name for p in UPLOAD_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS),
        key=lambda n: (UPLOAD_DIR / n).stat().st_size,
    )
    if not names:
        print("[图片引用] uploads 目录下没有图片，跳过")
        return

    messages: list[Any] = []
    for i, name in enumerate(names):
        messages.append(HumanMessage(content=[
            {"type": "image_ref", "name": name},
            {"type": "text", "text": f"第 {i + 1} 轮：这张截图里显示了什么？"},
        ]))
        messages.append(AIMessage(content=f"第 {i + 1} 轮回答：图中是一段界面截图，包含……（略）"))

    actual = _resolve_images(messages)
    all_inline = [
        message.model_copy(update={"content": [
            _load_image_block(b["name"]) if isinstance(b, dict) and b.get("type") == "image_ref"
            else b
            for b in message.content
        ]})
        if isinstance(message.content, list) else message
        for message in messages
    ]

    inline_chars, actual_chars = _payload_chars(all_inline), _payload_chars(actual)
    rounds = len(names)
    # 两种口径并列：真实分词器反映实际占用，approx 是库默认口径
    inline_real, actual_real = _real_tokens(all_inline), _real_tokens(actual)
    inline_approx = count_tokens_approximately(all_inline)
    actual_approx = count_tokens_approximately(actual)

    print(f"[图片引用] {rounds} 轮对话 / 每轮 1 张图（共 {rounds} 张）")
    for name in names:
        size, b64_len, size_wh = _image_info(name, read_upload_bytes)
        print(f"    {name}: 原图 {size:,} 字节（{size_wh}）→ base64 {b64_len:,} 字符")

    print(f"    全部内联（每轮都发原图）: {inline_chars:,} 字符")
    print(f"    引用式（仅本轮内联）    : {actual_chars:,} 字符")
    print(f"  [体积] 下降 {(1 - actual_chars / inline_chars) * 100:.1f}%（存储/传输口径）")
    print(
        f"  [token] 下降 {(1 - actual_real / inline_real) * 100:.1f}%"
        f"（{inline_real:,} → {actual_real:,}）"
    )
    print(
        f"    └ 对照：count_tokens_approximately 口径会报 "
        f"{(1 - actual_approx / inline_approx) * 100:.1f}%"
        f"（{inline_approx:,} → {actual_approx:,}），因中文被低估约 4 倍而虚高"
    )
    print(
        f"  [口径] 文本用 tiktoken cl100k_base 真实分词，图像按 {IMAGE_TOKENS_PER_BLOCK}"
        " token/张固定计（对齐 OpenAI 低分辨率开销）；图像那块与图片大小无关，"
        "收益随轮数增加，但被两边共有的文本稀释。"
    )


def bench_compression() -> None:
    """量化 langmem 摘要压缩前后的 token 下降比例（用真实摘要模型）。

    用项目真实的 build_summarize_node + count_tokens_approximately（与 langmem
    内部阈值判定同源），历史素材取自知识库文档。
    """
    from langchain.messages import AIMessage, HumanMessage
    from langchain_core.messages.utils import count_tokens_approximately

    from src.asyncio_runner import run_async
    from src.agents.graph import main_graph

    node = main_graph.build_summarize_node()
    if node is None:
        print("[上下文压缩] HISTORY_SUMMARY=0，压缩未启用，跳过")
        return

    kb = PROJECT_ROOT / "src" / "rag" / "knowledge_base"
    texts = [p.read_text(encoding="utf-8-sig") for p in sorted(kb.glob("*.txt"))]

    # 一直堆到超过触发阈值，避免换阈值后测不出压缩
    threshold = main_graph.HISTORY_SUMMARY_BEFORE_TOKENS
    messages: list[Any] = []
    i = 0
    while count_tokens_approximately(messages) <= threshold * 1.2:
        text = texts[i % len(texts)]
        messages.append(
            HumanMessage(content=f"第 {i + 1} 轮提问：这份制度的要点是什么？\n{text}", id=f"h{i}")
        )
        messages.append(AIMessage(content=f"第 {i + 1} 轮回答：{text}", id=f"a{i}"))
        i += 1
        if i >= 200:
            break

    before = count_tokens_approximately(messages)
    print(
        f"[上下文压缩] 输入 {len(messages)} 条消息 / {before:,} token"
        f"（触发阈值 max_tokens_before_summary={threshold}，"
        f"压缩后预算 max_tokens={main_graph.HISTORY_SUMMARY_MAX_TOKENS}，"
        f"摘要上限 max_summary_tokens={main_graph.HISTORY_SUMMARY_MAX_SUMMARY_TOKENS}）"
    )

    update = run_async(node({"messages": messages}))
    if not update:
        print("[上下文压缩] 未触发压缩（历史未达阈值，或摘要模型返回空），无法统计")
        return

    summarized = update["summarized_messages"]
    after = count_tokens_approximately(summarized)
    print(f"    压缩后 {len(summarized)} 条消息 / {after:,} token")
    print(
        f"  [结论] token 下降 {(1 - after / before) * 100:.1f}%"
        f"（{before:,} → {after:,}）"
    )


# 标注 query -> 期望命中文档；库里只有 7 篇，Hit@3 随机基线 43%，Hit@1 14%
RETRIEVAL_CASES: list[tuple[str, str]] = [
    ("公司哪些信息属于保密范围？", "knowledge_base/保密与信息安全制度.txt"),
    ("保密范围包括经营信息吗？", "knowledge_base/保密与信息安全制度.txt"),
    ("保密与信息安全制度对员工有什么要求？", "knowledge_base/保密与信息安全制度.txt"),
    ("入职需要准备哪些资料？", "knowledge_base/入职须知.txt"),
    ("新员工入职要提交身份证吗？", "knowledge_base/入职须知.txt"),
    ("员工入职须知里有哪些内容？", "knowledge_base/入职须知.txt"),
    ("公司有哪些培训类别？", "knowledge_base/培训管理制度.txt"),
    ("新员工入职培训什么时候完成？", "knowledge_base/培训管理制度.txt"),
    ("岗位技能培训由谁组织？", "knowledge_base/培训管理制度.txt"),
    ("上下班工作时间是怎么规定的？", "knowledge_base/考勤制度.txt"),
    ("公司实行几天工作制？", "knowledge_base/考勤制度.txt"),
    ("上午的工作时间是几点到几点？", "knowledge_base/考勤制度.txt"),
    ("员工薪酬由哪几部分组成？", "knowledge_base/薪酬管理制度.txt"),
    ("绩效工资占薪酬的比例是多少？", "knowledge_base/薪酬管理制度.txt"),
    ("基本工资是怎么确定的？", "knowledge_base/薪酬管理制度.txt"),
    ("请假有哪些类型？", "knowledge_base/请假管理制度.txt"),
    ("事假全年最多可以请多少天？", "knowledge_base/请假管理制度.txt"),
    ("年假有多少天？", "knowledge_base/请假管理制度.txt"),
    ("公司可以报销哪些费用？", "knowledge_base/费用报销流程.txt"),
    ("差旅费包括哪些项目？", "knowledge_base/费用报销流程.txt"),
    ("业务招待费可以报销吗？", "knowledge_base/费用报销流程.txt"),
]

SOURCE_RE = re.compile(r"来源：(.+?)）")


def bench_retrieval() -> None:
    """检索命中率（Hit@1 / Hit@3 / MRR），直接调用项目真实的 search_knowledge。

    判定依据是工具返回文本里的「来源：xxx」——即模型实际看到的内容，而不是绕过
    阈值过滤与重排、直接看向量召回结果。
    """
    from src.agents.mcps import rag_mcp_server as rag

    search = getattr(rag.search_knowledge, "fn", rag.search_knowledge)  # 兼容工具包装
    rerank_ok = rag.reranker.rerank("测试", ["测试文档"]) is not None

    hit1 = hit3 = 0
    reciprocal: list[float] = []
    empty = 0
    misses: list[str] = []

    for query, expected in RETRIEVAL_CASES:
        sources = SOURCE_RE.findall(search(query, top_k=3))
        if not sources:
            empty += 1
        rank = sources.index(expected) + 1 if expected in sources else 0
        if rank == 1:
            hit1 += 1
        if rank:
            hit3 += 1
        reciprocal.append(1 / rank if rank else 0.0)
        if not rank:
            misses.append(f"{query} → 期望 {expected.rsplit('/', 1)[-1]}，实得 {sources or '（无，未达阈值）'}")

    n = len(RETRIEVAL_CASES)
    print(f"[检索命中率] {n} 条标注 query / top_k=3")
    print(f"    重排：{'启用' if rerank_ok else '不可用（已退回余弦相似度）'}")
    print(f"    Hit@1 {hit1}/{n} = {hit1 / n * 100:.1f}%（随机基线 14.3%）")
    print(f"    Hit@3 {hit3}/{n} = {hit3 / n * 100:.1f}%（随机基线 42.9%）")
    print(f"    MRR  {statistics.fmean(reciprocal):.3f}")
    print(f"    阈值过滤后返回空：{empty}/{n}")
    for line in misses:
        print(f"    未命中：{line}")


# 端到端走知识库问答：LLM 决策 -> MCP 工具 -> LLM 汇总
E2E_QUESTIONS = [
    "公司的请假流程是怎样的？",
    "年假有多少天？",
    "差旅费包括哪些项目？",
]
E2E_WARMUP = 1  # 首次要建 MCP 连接与检查点，先跑一轮预热不计入


def bench_e2e_latency() -> None:
    """端到端延迟：ainvoke 一次完整问答的墙钟耗时。

    需要 PG（检查点/向量库）+ 5 个 MCP 常驻服务（本函数自己拉起并关闭）+ LLM API。
    注意 init_agent 会遍历全部服务拉工具，任一连不上就整体失败，因此无法只起部分服务。
    """
    from langchain.messages import HumanMessage

    from src.asyncio_runner import run_async
    from src.agents.checkpoint import close_async_persistence, init_async_persistence
    from src.agents.graph.main_graph import get_main_agent, init_agent

    async def measure() -> list[float]:
        await init_async_persistence()
        await init_agent()
        agent = get_main_agent()

        def state(question: str) -> dict[str, Any]:
            return {
                "messages": [HumanMessage(content=question)],
                "summarized_messages": [],
                "context": {},
                "task_completed": False,
                "pending_question": "",
                "pending_tool_call_id": "",
            }

        try:
            for _ in range(E2E_WARMUP):
                # 预热：首轮要建 MCP 连接、跑 setup
                await agent.ainvoke(
                    state(E2E_QUESTIONS[0]),
                    config={"configurable": {"thread_id": f"bench-warmup-{uuid.uuid4().hex}"}},
                )

            timings: list[float] = []
            for question in E2E_QUESTIONS:
                start = time.perf_counter()
                result = await agent.ainvoke(
                    state(question),
                    config={"configurable": {"thread_id": f"bench-{uuid.uuid4().hex}"}},
                )
                elapsed = (time.perf_counter() - start) * 1000
                timings.append(elapsed)
                used_tools = [
                    call["name"]
                    for message in result["messages"]
                    if getattr(message, "tool_calls", None)
                    for call in message.tool_calls
                ]
                print(f"    {elapsed:,.0f} ms | 触发工具 {used_tools or '（无）'} | {question}")
            return timings
        finally:
            await close_async_persistence()

    from src.agents.mcps import registry

    procs: list[subprocess.Popen] = []
    print("[端到端延迟] 拉起 MCP 服务（gui_server 要加载 torch，可能较慢）")
    try:
        procs = _start_servers(list(registry.SERVERS))
        timings = run_async(measure())
    finally:
        _stop_servers(procs)

    print(
        f"  [结论] 端到端中位数 {statistics.median(timings):,.0f} ms / "
        f"最小 {min(timings):,.0f} ms / 最大 {max(timings):,.0f} ms"
        f"（{len(timings)} 次，已剔除预热轮）"
    )


BENCHES = {
    "roundtrip": bench_mcp_roundtrip,
    "cancel": bench_cancel_latency,
    "image": bench_image_ref,
    "compress": bench_compression,
    "retrieval": bench_retrieval,
    "e2e": bench_e2e_latency,
}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    todo = (
        list(BENCHES) if which == "all"
        else [name.strip() for name in which.split(",") if name.strip()]
    )

    for name in todo:
        bench = BENCHES.get(name)
        if bench is None:
            print(f"[跳过] 未知指标：{name}（可选 {', '.join(BENCHES)}）")
            continue
        try:
            bench()
        except Exception as exc:  # noqa: BLE001 - 基准脚本，直接暴露原因
            print(f"[{name}] 测量失败：{type(exc).__name__}: {exc}")
        print()


if __name__ == "__main__":
    main()
