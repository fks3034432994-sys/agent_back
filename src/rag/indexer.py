"""知识库向量化入库。

提供两种入口：
- build_knowledge_base()：扫描 knowledge_base/ 目录全量重建（保留原有脚本用法）；
- ingest_text()：单文档导入（上传接口使用），同名 source 自动替换旧片段。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.agents.models import SiliconFlowEmbeddings

load_dotenv()

# 知识库目录用绝对路径，避免受 CWD 影响
KNOWLEDGE_BASE_DIR = Path(__file__).parent / "knowledge_base"
COLLECTION_NAME = "knowledge_base"


def normalized_source(raw: str) -> str:
    """把任意来源写法规整成统一的 ``knowledge_base/<文件名>``。

    同一张表里出现过三种写法：``ingest_text`` 写正斜杠的相对路径；
    ``DirectoryLoader`` 在 Windows 上给反斜杠、甚至绝对路径（早期用相对目录建库，
    所以库里留下的是 ``knowledge_base\\xxx.txt``）。三者在数据库里无法互相匹配，
    后果是"按 source 删除/查重"只对其中一部分数据生效——表现为有些文档删不掉。
    """
    name = str(raw or "").replace("\\", "/").rsplit("/", 1)[-1]
    return f"{COLLECTION_NAME}/{name}"


def normalize_doc_sources(docs: list[Document]) -> list[Document]:
    """就地规整 loader 给出的 source（Windows 反斜杠/绝对路径 -> knowledge_base/<名>）。"""
    for doc in docs:
        doc.metadata["source"] = normalized_source(str(doc.metadata.get("source", "")))
    return docs


def normalize_source_paths(collection: str = COLLECTION_NAME) -> int:
    """一次性修正历史数据里 source 的路径写法，返回被修正的条数。

    幂等：已规整的行不在更新范围内，可以重复执行。命令行入口见文件末尾。
    """
    import psycopg

    updates: list[tuple[str, str]] = []
    with psycopg.connect(os.getenv("DB_URI")) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.id, e.cmetadata ->> 'source'
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON c.uuid = e.collection_id
            WHERE c.name = %s
            """,
            (collection,),
        )
        updates = [
            (normalized_source(raw), row_id)
            for row_id, raw in cur.fetchall()
            if raw and normalized_source(raw) != raw
        ]
        if updates:
            # to_jsonb 是多态函数，参数必须显式转 text；占位符替换在客户端做
            cur.executemany(
                """
                UPDATE langchain_pg_embedding
                SET cmetadata = jsonb_set(cmetadata, '{source}', to_jsonb(%s::text))
                WHERE id = %s
                """,
                updates,
            )
            conn.commit()
    return len(updates)

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
)

_embeddings: SiliconFlowEmbeddings | None = None
_vectorstore: PGVector | None = None


def get_embeddings() -> SiliconFlowEmbeddings:
    """懒加载嵌入模型（进程内复用，避免每次导入重新创建）。"""
    global _embeddings
    if _embeddings is None:
        _embeddings = SiliconFlowEmbeddings()
    return _embeddings


def get_vectorstore() -> PGVector:
    """懒加载向量存储实例（进程内复用）。"""
    global _vectorstore
    if _vectorstore is None:
        connection_string = os.getenv("DB_URI").replace("postgresql://", "postgresql+psycopg://")
        _vectorstore = PGVector(
            connection=connection_string,
            embeddings=get_embeddings(),
            collection_name=COLLECTION_NAME,
        )
    return _vectorstore


def delete_by_source(source: str, collection: str = COLLECTION_NAME) -> int:
    """删除指定 source 的旧片段（重复导入时先删后加，实现替换），返回删除条数。

    当前安装的 langchain_postgres 版本 delete() 仅支持按 id 删除，
    不支持按 metadata 过滤，因此直接用 SQL 按 cmetadata->>'source' 清理。
    """
    import psycopg

    with psycopg.connect(os.getenv("DB_URI")) as conn, conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM langchain_pg_embedding e
            USING langchain_pg_collection c
            WHERE c.uuid = e.collection_id
              AND c.name = %s
              AND e.cmetadata ->> 'source' = %s
            """,
            (collection, source),
        )
        deleted = cur.rowcount
        conn.commit()
    return deleted


def delete_by_id(point_id: str, collection: str = COLLECTION_NAME) -> int:
    """删除指定 id 的单个向量片段，返回删除条数。"""
    import psycopg

    with psycopg.connect(os.getenv("DB_URI")) as conn, conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM langchain_pg_embedding e
            USING langchain_pg_collection c
            WHERE c.uuid = e.collection_id
              AND c.name = %s
              AND e.id = %s
            """,
            (collection, point_id),
        )
        deleted = cur.rowcount
        conn.commit()
    return deleted


def ingest_text(name: str, content: str) -> tuple[int, int]:
    """导入单个文本文档：分块、嵌入、入库。

    返回 (生成的 chunk 数, 被替换的旧 chunk 数)。
    """
    # source 统一成 knowledge_base/ 前缀 + 正斜杠
    source = normalized_source(name)
    deleted = delete_by_source(source)

    doc = Document(page_content=content, metadata={"source": source})
    chunks = SPLITTER.split_documents([doc])
    if chunks:
        vs = get_vectorstore()
        vs.create_collection()  # get_or_create，幂等；collection 不存在时自动建
        vs.add_documents(chunks)
    return len(chunks), deleted


def build_knowledge_base() -> tuple[int, int]:
    """全量重建：扫描 knowledge_base/ 目录下所有文本文件并向量化入库。"""
    loader = DirectoryLoader(
        str(KNOWLEDGE_BASE_DIR),
        glob="**/*.*",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8-sig"},
        show_progress=True,
    )
    docs = loader.load()
    # DirectoryLoader 写的是平台相关路径，统一规整否则删不掉
    normalize_doc_sources(docs)
    print(f"加载了 {len(docs)} 个文档片段")

    chunks = SPLITTER.split_documents(docs)
    print(f"分块后共 {len(chunks)} 个片段")

    if chunks:
        get_vectorstore().add_documents(chunks)
    print("知识库构建完成！")
    return len(docs), len(chunks)


if __name__ == "__main__":
    # 用法：
    #   python -m src.rag.indexer                     # 全量重建知识库
    #   python -m src.rag.indexer --normalize-sources # 只修正历史 source 的路径写法
    import sys

    if "--normalize-sources" in sys.argv[1:]:
        print(f"已修正 {normalize_source_paths()} 条 source")
    else:
        build_knowledge_base()
