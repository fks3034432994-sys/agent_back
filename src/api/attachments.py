"""对话附件上传接口。

上传的文件落盘到 MCP file server 的 uploads 目录：

- 文本附件随后以 MCP resource（``upload://<name>``）的形式被对话流程读取；
- 图片是二进制，无法走 resource 的文本链路，改由对话流程直接读字节并编码成
  多模态消息（见 src/api/chat.py），本模块只负责按原名把图片吐回前端展示。

职责划分：API 层只负责"把文件放进来 / 把图片原样取出来"，"把内容注入上下文"
由对话流程负责。
"""

import logging
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from src.agents.mcps.upload_store import (
    MAX_ATTACHMENT_BYTES,
    MAX_IMAGE_BYTES,
    guess_media_type,
    is_image,
    read_upload_bytes,
    save_upload,
    save_upload_bytes,
    safe_name,
    unique_name,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/attachments", tags=["attachments"])

# 按文本文件处理，超出这个范围的（pdf/docx 等）需要另外的解析器，暂不支持
ALLOWED_EXTS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".log",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".rb",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".php", ".kt", ".swift",
    ".sql", ".yaml", ".yml", ".toml", ".ini", ".conf", ".properties",
    ".xml", ".html", ".htm", ".css", ".scss", ".vue", ".sh", ".bat", ".ps1",
}


def _to_uri(name: str) -> str:
    """把文件名转成 resource URI。

    必须百分号编码（safe="" 让 "/" 也被编码），才能塞进单段模板参数；
    模板匹配时 FastMCP 会自动 unquote 还原。
    """
    return f"upload://{quote(name, safe='')}"


def _save_image(name: str, file: UploadFile) -> dict:
    """保存图片附件：按字节原样落盘，不做文本解码。"""
    raw = file.file.read(MAX_IMAGE_BYTES + 1)
    if len(raw) > MAX_IMAGE_BYTES:
        limit_mb = MAX_IMAGE_BYTES // 1024 // 1024
        raise HTTPException(status_code=400, detail=f"图片超过 {limit_mb}MB 限制")
    if not raw:
        raise HTTPException(status_code=400, detail="文件内容为空")

    save_upload_bytes(name, raw)
    return {
        "name": name,
        "uri": _to_uri(name),
        "size": len(raw),
        "chars": 0,
        "kind": "image",
    }


def _save_text(name: str, file: UploadFile) -> dict:
    """保存文本附件：校验扩展名并按文本解码。"""
    if Path(name).suffix.lower() not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"仅支持文本附件（如 {', '.join(sorted(ALLOWED_EXTS)[:8])} …）或图片",
        )

    raw = file.file.read(MAX_ATTACHMENT_BYTES + 1)
    if len(raw) > MAX_ATTACHMENT_BYTES:
        limit_mb = MAX_ATTACHMENT_BYTES // 1024 // 1024
        raise HTTPException(status_code=400, detail=f"附件超过 {limit_mb}MB 限制")

    # utf-8 优先，兼容 Windows 记事本的 GBK 编码
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            content = raw.decode("gbk")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="无法识别文件编码，请使用 UTF-8 或 GBK 编码的文本文件",
            )
    if not content.strip():
        raise HTTPException(status_code=400, detail="文件内容为空")

    save_upload(name, content)
    return {
        "name": name,
        "uri": _to_uri(name),
        "size": len(raw),
        "chars": len(content),
        "kind": "text",
    }


@router.post("")
def upload_attachment(file: UploadFile = File(...)):
    """上传一个附件（文本或图片），返回其 MCP resource URI。

    定义为同步函数：读取与落盘都是阻塞操作，FastAPI 会放入线程池执行。
    """
    name = safe_name(file.filename or "")
    if name is None:
        raise HTTPException(status_code=400, detail="文件名非法")

    # 保留原始文件名，重名时才追加序号
    name = unique_name(name)

    if is_image(name):
        return _save_image(name, file)
    return _save_text(name, file)


@router.get("/{name}")
def get_attachment(name: str):
    """按文件名读取附件原始内容（前端展示已上传的图片用）。

    历史消息里只保存文件名，图片实体由本接口按需提供，避免把 base64 塞进
    历史接口的响应体。
    """
    data = read_upload_bytes(name)
    if data is None:
        raise HTTPException(status_code=404, detail="附件不存在")
    return Response(content=data, media_type=guess_media_type(name))
