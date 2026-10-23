"""知识库入库的 source 路径规整（src/rag/indexer.py）。

同一张表里 source 有三种历史写法（纯文件名 / 正斜杠相对路径 / Windows 反斜杠或绝对路径），
必须收敛到同一个规范形式，否则"按 source 删除、按 source 查重"只对一部分数据生效——
表现就是有些文档删不掉。这里不连数据库：迁移函数用假 psycopg 驱动。
"""

import sys

import pytest
from langchain_core.documents import Document

from src.rag import indexer


@pytest.mark.parametrize(
    "raw",
    [
        "入职须知.txt",                                        # 纯文件名
        "knowledge_base/入职须知.txt",                         # ingest_text 的规范写法
        "knowledge_base\\入职须知.txt",                        # 早期 Windows 建库留下的
        "c:\\Users\\a\\src\\rag\\knowledge_base\\入职须知.txt",  # DirectoryLoader 的绝对路径
        "/home/a/src/rag/knowledge_base/入职须知.txt",
    ],
)
def test_normalized_source_converges_to_one_form(raw):
    assert indexer.normalized_source(raw) == "knowledge_base/入职须知.txt"


@pytest.mark.parametrize("raw", ["", None])
def test_normalized_source_handles_empty(raw):
    assert indexer.normalized_source(raw) == "knowledge_base/"


def test_normalize_doc_sources_rewrites_in_place():
    docs = [
        Document(page_content="a", metadata={"source": "knowledge_base\\考勤制度.txt"}),
        Document(page_content="b", metadata={"source": "c:/kb/knowledge_base/请假管理制度.txt"}),
        Document(page_content="c", metadata={}),
    ]

    out = indexer.normalize_doc_sources(docs)

    assert out is docs  # 就地修改，调用方拿到的是同一个列表
    assert [doc.metadata["source"] for doc in docs] == [
        "knowledge_base/考勤制度.txt",
        "knowledge_base/请假管理制度.txt",
        "knowledge_base/",
    ]


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.queries: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.queries.append(("execute", sql, params))

    def fetchall(self):
        return self._rows

    def executemany(self, sql, params):
        self.queries.append(("executemany", sql, list(params)))


class FakeConnection:
    def __init__(self, rows):
        self.cursor_obj = FakeCursor(rows)
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed = True


class FakePsycopg:
    def __init__(self, connection):
        self.connection = connection
        self.uri = None

    def connect(self, uri):
        self.uri = uri
        return self.connection


@pytest.fixture()
def fake_db(monkeypatch):
    def _install(rows) -> FakeConnection:
        connection = FakeConnection(rows)
        monkeypatch.setitem(sys.modules, "psycopg", FakePsycopg(connection))
        return connection

    return _install


def test_normalize_source_paths_updates_only_non_canonical_rows(fake_db):
    conn = fake_db(
        [
            ("id-1", "knowledge_base\\考勤制度.txt"),       # 需要修正
            ("id-2", "knowledge_base/请假管理制度.txt"),    # 已规范
            ("id-3", "c:/kb/knowledge_base/费用报销流程.txt"),  # 需要修正
            ("id-4", None),                                # 没有 source，跳过
        ]
    )

    fixed = indexer.normalize_source_paths("knowledge_base")

    assert fixed == 2
    assert conn.committed is True
    updates = [q for q in conn.cursor_obj.queries if q[0] == "executemany"]
    assert updates[0][2] == [
        ("knowledge_base/考勤制度.txt", "id-1"),
        ("knowledge_base/费用报销流程.txt", "id-3"),
    ]
    # 假 cursor 校验不了 SQL 语义，只钉住一个坑：to_jsonb 是多态函数，参数要显式转 text
    assert "to_jsonb(%s::text)" in updates[0][1]


def test_normalize_source_paths_is_idempotent(fake_db):
    """第二次执行不应该再产生任何更新。"""
    conn = fake_db([("id-1", "knowledge_base/考勤制度.txt")])

    assert indexer.normalize_source_paths("knowledge_base") == 0
    assert conn.committed is False
    assert [q for q in conn.cursor_obj.queries if q[0] == "executemany"] == []


def test_normalize_source_paths_scopes_to_collection(fake_db):
    """只动指定 collection，避免误改其它库的数据。"""
    conn = fake_db([])

    indexer.normalize_source_paths("other_kb")

    select = conn.cursor_obj.queries[0]
    assert select[0] == "execute"
    assert select[2] == ("other_kb",)
