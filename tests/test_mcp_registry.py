"""MCP 服务地址表：主进程客户端与服务端都靠它拼接 URL，改错一个端口就是连不上。"""

import pytest

from src.agents.mcps import registry as reg


def test_url_uses_default_host_and_path():
    assert reg.url_of(reg.FILE_SERVER) == "http://127.0.0.1:8761/mcp"
    assert reg.url_of(reg.GUI_SERVER) == "http://127.0.0.1:8765/mcp"


def test_port_can_be_overridden_by_env(monkeypatch):
    # 同一台机器跑多套环境时按服务名改端口
    monkeypatch.setenv("MCP_PORT_FILE_SERVER", "9999")
    assert reg.port_of(reg.FILE_SERVER) == 9999
    assert reg.url_of(reg.FILE_SERVER).endswith(":9999/mcp")


def test_server_table_is_complete_and_consistent():
    assert set(reg.SERVERS) == {
        reg.FILE_SERVER,
        reg.PYTHON_SERVER,
        reg.RAG_SERVER,
        reg.STORE_SERVER,
        reg.GUI_SERVER,
    }
    ports = [spec.port for spec in reg.SERVERS.values()]
    assert len(set(ports)) == len(ports), "端口不能重复"
    for spec in reg.SERVERS.values():
        assert spec.module.startswith("src.agents.mcps.")


def test_serve_rejects_unknown_server_name():
    with pytest.raises(ValueError):
        reg.serve(object(), "not-a-server")
