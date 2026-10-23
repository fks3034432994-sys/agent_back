"""附件落盘/读取与文件名安全校验。

这里除了功能，还专门覆盖"目录穿越"这类安全问题：附件名来自 HTTP 上传，
是不可信输入。
"""

from src.agents.mcps import upload_store as us
from src.api.attachments import _to_uri


def test_safe_name_strips_directory():
    # Windows / POSIX 两种写法都会被压成纯文件名
    assert us.safe_name(r"C:\tmp\a.txt") == "a.txt"
    assert us.safe_name("dir/sub/a.txt") == "a.txt"
    assert us.safe_name("../../etc/passwd") == "passwd"


def test_safe_name_rejects_illegal_input():
    for bad in ("", None, ".", "..", "a<b.txt", "a>b.txt", 'a"b.txt', "a|b.txt", "a?b.txt", "a*b.txt"):
        assert us.safe_name(bad) is None, bad


def test_safe_name_always_returns_a_safe_single_segment():
    """断言的是"结果一定安全"这个不变量，而不是具体字符串。

    原因：冒号在平台间归属不同——Windows 把 "a:b.txt" 的 "a:" 当盘符吃掉（得到
    "b.txt"），POSIX 则把它当普通字符从而被 _INVALID_CHARS 拒绝。两种结果都安全。
    """
    for raw in ("a:b.txt", r"C:\tmp\a.txt", "dir/sub/a.txt", "../../x.txt"):
        safe = us.safe_name(raw)
        assert safe is not None, raw
        assert "/" not in safe and "\\" not in safe
        for ch in '<>:"|?*':
            assert ch not in safe, (raw, safe)


def test_path_traversal_cannot_escape_upload_dir(upload_dir, tmp_path):
    upload_dir.mkdir(parents=True)
    # 把"机密文件"放在 uploads 的上一级：真有目录穿越就会被读到
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")

    assert us.read_upload_bytes("../secret.txt") is None
    assert us.read_upload_bytes("../../secret.txt") is None
    # 绝对路径也只取最后一段文件名
    assert us.read_upload_bytes(str(secret)) is None


def test_directory_part_of_a_name_is_ignored(upload_dir):
    upload_dir.mkdir(parents=True)
    (upload_dir / "note.txt").write_text("hi", encoding="utf-8")

    assert us.read_upload_bytes("any/dir/note.txt") == b"hi"


def test_is_image_is_case_insensitive():
    assert us.is_image("a.PNG")
    assert us.is_image("a.jpeg")
    assert not us.is_image("a.txt")
    assert not us.is_image("noext")


def test_guess_media_type_falls_back_to_binary():
    assert us.guess_media_type("a.png") == "image/png"
    assert us.guess_media_type("a.unknown-ext") == "application/octet-stream"


def test_unique_name_appends_increasing_index(upload_dir):
    upload_dir.mkdir(parents=True)
    (upload_dir / "a.txt").write_text("x", encoding="utf-8")
    assert us.unique_name("a.txt") == "a-1.txt"

    (upload_dir / "a-1.txt").write_text("x", encoding="utf-8")
    assert us.unique_name("a.txt") == "a-2.txt"


def test_unique_name_without_extension_and_when_free(upload_dir):
    upload_dir.mkdir(parents=True)
    assert us.unique_name("free.md") == "free.md"

    (upload_dir / "Makefile").write_text("x", encoding="utf-8")
    assert us.unique_name("Makefile") == "Makefile-1"


def test_binary_roundtrip_keeps_bytes_exact(upload_dir):
    # 图片必须以字节写读：走文本编码会把非 UTF-8 字节写坏
    raw = b"\x89PNG\r\n\x1a\n\xff\xfe\x00binary"
    us.save_upload_bytes("probe.png", raw)
    assert us.read_upload_bytes("probe.png") == raw


def test_read_upload_returns_message_for_missing_file(upload_dir):
    assert us.read_upload("nope.txt") == "附件不存在：nope.txt"


def test_read_upload_rejects_illegal_name(upload_dir):
    assert us.read_upload("..") == "附件名非法：.."


def test_uri_roundtrip_between_api_and_store():
    """api/attachments 生成 URI、upload_store 解析 URI：跨模块契约不能错位。"""
    for name in ("a b.txt", "附件-1.png", "a#b?c.txt", "带百分号%的.txt"):
        assert us.name_from_uri(_to_uri(name)) == name
