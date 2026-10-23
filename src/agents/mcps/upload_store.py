"""上传附件的落盘与读取。

支持两类附件，读取路径不同：

- **文本**：由 MCP file server 以 resource（``upload://<name>``）读出，走"取资料"
  链路，让内容获取与资源来源解耦。
- **图片**：二进制，当作文本读会被 utf-8 解码破坏，因此不进 MCP resource 链路，
  改由对话流程直接读字节并编码成多模态消息（见 src/api/chat.py）。

两类共用这里的目录与文件名校验，避免各写一份路径逻辑。
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from urllib.parse import unquote

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(__file__).parent / "uploads"

MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024  # 文本单文件上限 5MB

# 图片以 base64 进检查点且每轮送模型，上限必须比文本严格得多
MAX_IMAGE_BYTES = 4 * 1024 * 1024

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

# Windows 文件名非法字符
_INVALID_CHARS = '<>:"|?*'


def safe_name(name: str) -> str | None:
    """取纯文件名并校验，防止 ``../`` 目录穿越；非法时返回 None。"""
    pure = Path(name or "").name
    if not pure or pure in {".", ".."}:
        return None
    for ch in _INVALID_CHARS:
        if ch in pure:
            return None
    return pure


def _resolve(name: str) -> Path | None:
    """把文件名解析为上传目录内的绝对路径，越界或不存在合法名字时返回 None。"""
    safe = safe_name(name)
    if safe is None:
        return None
    path = (UPLOAD_DIR / safe).resolve()
    if path.parent != UPLOAD_DIR.resolve():
        return None
    return path


def is_image(name: str) -> bool:
    """按扩展名判断是否图片附件。"""
    return Path(name).suffix.lower() in IMAGE_EXTS


def name_from_uri(uri: str) -> str:
    """从 ``upload://<编码后的文件名>`` 还原文件名。

    编码方式见 attachments._to_uri。uri 由本系统生成：文件名入库前已过 safe_name
    （Path(...).name + 非法字符校验），不含路径分隔符，所以解码并去掉前缀即可；
    即使拿到带路径的外部 uri，下游 _resolve 也会再归一化一次。
    """
    return unquote(uri).removeprefix("upload://")


def guess_media_type(name: str) -> str:
    """按扩展名推断 MIME 类型，未知时退回二进制流。"""
    return mimetypes.guess_type(name)[0] or "application/octet-stream"


def unique_name(name: str) -> str:
    """同名附件已存在时自动加序号，避免覆盖已有文件。

    历史消息里记的是文件名（见 chat.py 写入的 image_ref），一旦覆盖，旧消息重放时
    就会指向错误的内容，所以附件只追加、不覆盖。
    """
    if not (UPLOAD_DIR / name).exists():
        return name
    dot = name.rfind(".")
    if dot > 0:
        base = name[:dot]
        ext = name[dot:]
    else:
        base = name
        ext = ""
    index = 1
    while (UPLOAD_DIR / f"{base}-{index}{ext}").exists():
        index += 1
    return f"{base}-{index}{ext}"


def save_upload(name: str, content: str) -> Path:
    """把文本内容写入上传目录（调用方需先经 unique_name 保证不覆盖已有附件）。"""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / name
    path.write_text(content, encoding="utf-8")
    return path


def save_upload_bytes(name: str, data: bytes) -> Path:
    """把二进制内容写入上传目录（同名覆盖），用于图片等非文本附件。

    必须按字节写入：图片经文本编码会损坏（且 write_text 会在写入失败时先截断文件）。
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / name
    path.write_bytes(data)
    return path


def read_upload(name: str) -> str:
    """读取文本附件内容；名字非法或文件不存在时返回提示文本。

    返回提示文本而不是抛异常：本函数在 MCP resource 里被调用，resource 读取失败
    会让整个 read_resource 请求报错，而"附件的另一种失败"用文本表达更友好。
    """
    path = _resolve(name)
    if path is None:
        return f"附件名非法：{name}"
    if not path.is_file():
        return f"附件不存在：{name}"
    return path.read_text(encoding="utf-8", errors="replace")


def read_upload_bytes(name: str) -> bytes | None:
    """读取附件的原始字节；名字非法或文件不存在时返回 None。

    与 read_upload 的"返回提示文本"不同：本函数的调用方要直接返回二进制响应或做
    base64 编码，失败必须用 None 表达，好让上层转成 HTTP 404。
    """
    path = _resolve(name)
    if path is None or not path.is_file():
        return None
    return path.read_bytes()
