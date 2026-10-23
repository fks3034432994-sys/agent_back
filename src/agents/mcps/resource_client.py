"""MCP resource 的客户端工具。

resources 是 MCP 里 application-driven 的原语：由应用代码主动读取，模型不参与
选择。用户上传的附件经 ``upload://<name>`` 暴露，这里负责把它读成文本注入本轮
上下文。

对照 ``search_knowledge``（tool，model-driven）：工具是模型自己决定查什么，
resource 是用户明确指定要看哪份。

异步说明：全链路异步后不再需要"同步包装"——本模块只在对话流程（事件循环内）
被调用，直接 await 即可。
"""

from __future__ import annotations

import logging

from langchain_mcp_adapters.resources import load_mcp_resources

from src.agents.graph.main_graph import mcp_client
from src.agents.mcps.registry import FILE_SERVER
from src.agents.mcps.upload_store import name_from_uri

logger = logging.getLogger(__name__)


async def read_reference_text(uris: list[str]) -> str:
    """按 URI 读取附件内容，并标注来源。

    标注来源是必要的：资料是外部内容，要让模型能区分「外部资料」与「用户原话」。
    """
    async with mcp_client.session(FILE_SERVER) as session:
        blobs = await load_mcp_resources(session, uris=uris)

    parts: list[str] = []
    for blob in blobs:
        # 来源用可读文件名：URI 是百分号编码，写进上下文只是噪声
        uri = (blob.metadata or {}).get("uri", "")
        name = name_from_uri(uri)
        parts.append(f"【参考资料｜来源：{name or uri}】\n{blob.as_string()}")
    return "\n\n".join(parts)


async def load_reference_text(uris: list[str] | None) -> str:
    """读取选中的附件并拼成参考资料文本；失败只记日志，不打断本轮对话。"""
    if not uris:
        return ""
    try:
        return await read_reference_text(uris)
    except Exception:
        logger.exception("读取 MCP 参考资料失败")
        return ""
