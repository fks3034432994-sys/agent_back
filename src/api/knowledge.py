"""知识库向量化数据可视化接口。

读取 langchain_pg_embedding / langchain_pg_collection 表，将高维 embedding
用 PCA 投影到二维平面，连同来源、内容预览一起返回，供前端散点图展示。
"""

import logging
import os
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from src.rag.indexer import (
    COLLECTION_NAME,
    KNOWLEDGE_BASE_DIR,
    delete_by_id,
    delete_by_source,
    ingest_text,
)

logger = logging.getLogger(__name__)

load_dotenv()

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])

PREVIEW_LEN = 200  # document 字段返回给前端的截断长度（完整内容不入库展示，控制响应体积）
ALLOWED_EXTS = {".txt", ".md"}  # 允许上传的文本文件扩展名
MAX_UPLOAD_SIZE = 2 * 1024 * 1024  # 上传大小限制 2MB


def _parse_vector(text: str) -> list[float]:
    """解析 pgvector 的文本表示 '[0.1,0.2,...]' 为浮点列表。"""
    return [float(x) for x in text.strip()[1:-1].split(",") if x.strip()]


def _pca_2d(vectors: np.ndarray) -> np.ndarray:
    """将 (n, d) 向量矩阵投影到前两个主成分。

    返回 (n, 2)：以 0.5 为中心、等比缩放（X/Y 用同一比例尺，保持簇的几何形状
    不被拉伸变形），坐标严格落在 [0, 1]，最外围的点也不会越出画布。
    """
    n, d = vectors.shape
    if n < 2 or d < 2:
        return np.full((n, 2), 0.5)
    centered = vectors - vectors.mean(axis=0)
    # SVD 取前两个主成分，数值上比协方差特征分解稳定
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:2].T
    lo, hi = float(proj.min()), float(proj.max())
    span = hi - lo  # 全局跨度，统一缩放保持纵横比
    if span == 0:
        return np.full((n, 2), 0.5)
    # 用全局包围盒中点平移：按轴均值居中会让某轴溢出 [0,1]
    return (proj - (lo + hi) / 2) / span + 0.5


@router.get("/embeddings")
def get_embeddings(
    collection: str = Query("knowledge_base", min_length=1),
    limit: int = Query(2000, ge=1, le=5000),
):
    """返回指定 collection 的全部向量片段（含 PCA 二维坐标），供前端可视化。"""
    db_uri = os.getenv("DB_URI")
    if not db_uri:
        raise HTTPException(status_code=500, detail="未配置 DB_URI")

    try:
        with psycopg.connect(
            db_uri, row_factory=psycopg.rows.dict_row
        ) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT e.id,
                       e.document,
                       e.cmetadata ->> 'source' AS source,
                       e.embedding::text        AS embedding
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON c.uuid = e.collection_id
                WHERE c.name = %s
                LIMIT %s
                """,
                (collection, limit),
            )
            rows = cur.fetchall()
    except psycopg.OperationalError:
        logger.exception("连接向量库失败")
        raise HTTPException(status_code=500, detail="连接向量库失败，请检查 DB_URI 与数据库状态")

    points = []
    source_counts: dict[str, int] = {}
    dim = 0
    if rows:
        vectors = np.array([_parse_vector(r["embedding"]) for r in rows])
        dim = int(vectors.shape[1])
        coords = _pca_2d(vectors)
        for r, (x, y) in zip(rows, coords):
            source = r["source"] or "未知来源"
            source_counts[source] = source_counts.get(source, 0) + 1
            points.append(
                {
                    "id": r["id"],
                    "x": round(float(x), 4),
                    "y": round(float(y), 4),
                    "source": source,
                    "document": (r["document"] or "")[:PREVIEW_LEN],
                }
            )

    return {
        "collection": collection,
        "total": len(points),
        "dim": dim,
        "sources": [
            {"name": name, "count": count}
            for name, count in sorted(source_counts.items(), key=lambda kv: -kv[1])
        ],
        "points": points,
    }


@router.get("/embeddings/{point_id}")
def get_embedding_detail(
    point_id: str,
    collection: str = Query("knowledge_base", min_length=1),
):
    """返回单个片段的完整内容。

    列表接口的 document 只带 PREVIEW_LEN 字预览（控制响应体积），
    抽屉展示时再按 id 拉取完整文本，避免长片段被截断。
    """
    db_uri = os.getenv("DB_URI")
    if not db_uri:
        raise HTTPException(status_code=500, detail="未配置 DB_URI")

    try:
        with psycopg.connect(
            db_uri, row_factory=psycopg.rows.dict_row
        ) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT e.id,
                       e.document,
                       e.cmetadata ->> 'source' AS source
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON c.uuid = e.collection_id
                WHERE c.name = %s AND e.id = %s
                """,
                (collection, point_id),
            )
            row = cur.fetchone()
    except psycopg.OperationalError:
        logger.exception("连接向量库失败")
        raise HTTPException(status_code=500, detail="连接向量库失败，请检查 DB_URI 与数据库状态")

    if row is None:
        raise HTTPException(status_code=404, detail="片段不存在")

    return {
        "id": row["id"],
        "source": row["source"] or "未知来源",
        "document": row["document"] or "",
    }


def _remove_local_file(source: str) -> bool:
    """删除知识库目录下与 source 对应的本地文件，返回是否真的删除了文件。

    source 形如 "knowledge_base/xxx.txt"。只取纯文件名（Path.name）并校验最终
    路径确实落在 KNOWLEDGE_BASE_DIR 内，防止 source 被构造成 ../ 越权删除。
    不做删除时全量重建会把已删文档重新灌回库里，因此默认一并清理。
    """
    prefix = f"{COLLECTION_NAME}/"
    if not source.startswith(prefix):
        return False
    name = Path(source[len(prefix) :]).name
    if not name:
        return False
    root = KNOWLEDGE_BASE_DIR.resolve()
    path = (KNOWLEDGE_BASE_DIR / name).resolve()
    if path.parent != root or not path.is_file():
        return False
    path.unlink()
    return True


@router.delete("/embeddings/{point_id}")
def delete_embedding(
    point_id: str,
    collection: str = Query("knowledge_base", min_length=1),
):
    """删除单个向量片段（按 id）。"""
    try:
        deleted = delete_by_id(point_id, collection)
    except psycopg.Error:
        logger.exception("删除向量片段失败: %s", point_id)
        raise HTTPException(status_code=500, detail="删除失败，请检查数据库状态")

    if deleted == 0:
        raise HTTPException(status_code=404, detail="片段不存在")

    return {"deleted": point_id, "count": deleted}


@router.delete("/sources")
def delete_source(
    source: str = Query(..., min_length=1, description="片段来源，形如 knowledge_base/xxx.txt"),
    collection: str = Query("knowledge_base", min_length=1),
    remove_file: bool = Query(True, description="是否同时删除知识库目录下的本地文件"),
):
    """按来源删除该文档的全部片段（清空某个文件的向量化数据）。

    delete 语义幂等：先删库再删盘，库里没有对应数据时返回 404。
    """
    try:
        deleted = delete_by_source(source, collection)
    except psycopg.Error:
        logger.exception("按来源删除向量失败: %s", source)
        raise HTTPException(status_code=500, detail="删除失败，请检查数据库状态")

    if deleted == 0:
        raise HTTPException(status_code=404, detail="该来源没有向量化数据")

    # 只处理属于当前 collection 的 source，避免误删其它库的文档
    removed_file = _remove_local_file(source) if remove_file and collection == COLLECTION_NAME else False

    return {"source": source, "deleted": deleted, "removed_file": removed_file}


@router.post("/documents")
def upload_document(file: UploadFile = File(...)):
    """上传文本文档并向量化入库（与 indexer.py 相同的分块与存储逻辑）。

    同名文件重复上传时自动替换旧片段（先删后加）。
    定义为同步函数：嵌入调用是阻塞 HTTP 请求，由 FastAPI 放入线程池执行。
    """
    # 取 Path.name 去掉路径部分，防目录穿越
    name = Path(file.filename or "").name
    if not name:
        raise HTTPException(status_code=400, detail="文件名为空")
    if Path(name).suffix.lower() not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"仅支持 {', '.join(sorted(ALLOWED_EXTS))} 文本文件",
        )

    raw = file.file.read(MAX_UPLOAD_SIZE + 1)
    if len(raw) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail="文件超过 2MB 限制")
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            content = raw.decode("gbk")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="无法识别文件编码，请使用 UTF-8 或 GBK 编码的文本文件")
    if not content.strip():
        raise HTTPException(status_code=400, detail="文件内容为空")

    dest = KNOWLEDGE_BASE_DIR / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")

    try:
        chunks, replaced = ingest_text(name, content)
    except Exception:
        logger.exception("文档向量化失败: %s", name)
        raise HTTPException(status_code=500, detail="文档向量化失败，请检查嵌入服务配置与数据库状态")

    return {
        "filename": name,
        "chunks": chunks,
        "replaced_chunks": replaced,
        "source": f"{COLLECTION_NAME}/{name}",
    }
