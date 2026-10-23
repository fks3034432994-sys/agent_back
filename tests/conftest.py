"""测试环境准备。

整个测试套件**不需要数据库、不需要启动 MCP 服务、不联网、不调用大模型**，
所以 clone 下来直接 `uv run pytest` 就能跑。

三件与"能不能跑起来"有关的事：

1. 在导入任何 src.* 之前把模型相关环境变量补成占位值。本项目在 import 期就会构造
   模型客户端（agents/models/models.py），没有 .env 时 import 直接失败。用
   os.environ.setdefault 而不是硬覆盖，所以本地有 .env 时仍然走真实配置
   （load_dotenv 默认也不覆盖已存在的变量）。
2. 异步测试统一通过 src.asyncio_runner.run_async 执行。Windows 上生产环境必须跑在
   SelectorEventLoop（异步 psycopg 的前提），直接 asyncio.run() 会拿到
   ProactorEventLoop——测试与生产的事件循环保持一致，才不会出现"测试过了线上挂"。
3. 附件相关的测试把 uploads 目录指到 tmp_path，既不污染真实目录，也不依赖它。
"""

import os

import pytest

_PLACEHOLDER_ENV = {
    "MAIN_MODEL_ID": "test-main",
    "MAIN_MODEL_URL": "http://127.0.0.1:1/v1",
    "MAIN_MODEL_API_KEY": "test-key",
    "TARS_MODEL_ID": "test-vision",
    "TARS_MODEL_API_KEY": "test-key",
    "TASK_PLANNING_MODEL_ID": "test-planning",
    "SILI_CONFLOW_URL": "http://127.0.0.1:1/v1",
    "SILI_CONFLOW_API_KEY": "test-key",
    "EMBEDDING_MODEL_URL": "http://127.0.0.1:1",
    "EMBEDDING_MODEL_API_KEY": "test-key",
    "EMBEDDING_MODEL_ID": "test-embedding",
}

for _key, _value in _PLACEHOLDER_ENV.items():
    os.environ.setdefault(_key, _value)


@pytest.fixture()
def upload_dir(tmp_path, monkeypatch):
    """把附件目录指向临时目录，返回该目录。"""
    from src.agents.mcps import upload_store

    target = tmp_path / "uploads"
    monkeypatch.setattr(upload_store, "UPLOAD_DIR", target)
    return target
