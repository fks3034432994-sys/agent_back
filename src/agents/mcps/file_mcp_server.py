import os
import json
import sys
from pathlib import Path

# 先补 sys.path：直接运行本文件时 sys.path[0] 是 mcps/，否则找不到 src 包
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from fastmcp import FastMCP
from src.agents.mcps.upload_store import read_upload

mcp = FastMCP("File")


@mcp.tool()
def read_file(path: str) -> str:
    """
    读取整个文件的内容。

    参数:
        path: 文件路径（绝对路径或相对路径）

    返回:
        成功时返回文件内容字符串；失败时返回错误信息字符串
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"错误：文件 '{path}' 不存在"
    except PermissionError:
        return f"错误：没有权限读取文件 '{path}'"
    except Exception as e:
        return f"读取文件失败：{e}"


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """
    将内容写入文件（覆盖模式）。

    参数:
        path: 文件路径
        content: 要写入的文本内容

    返回:
        成功时返回确认信息；失败时返回错误信息字符串
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"文件已成功写入: {path}"
    except Exception as e:
        return f"写入文件失败：{e}"


@mcp.tool()
def append_file(path: str, content: str) -> str:
    """
    将内容追加到文件末尾。

    参数:
        path: 文件路径
        content: 要追加的文本内容

    返回:
        成功时返回确认信息；失败时返回错误信息字符串
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(content)
        return f"内容已成功追加到: {path}"
    except Exception as e:
        return f"追加内容失败：{e}"



@mcp.tool()
def read_lines(path: str, start_line: int = 1, end_line: int = None) -> str:
    """
    读取文件的指定行范围（行号从1开始）。
    如果不指定 end_line，则读取从 start_line 到文件末尾。

    参数:
        path: 文件路径
        start_line: 起始行号（从1开始，默认为1）
        end_line: 结束行号（包含该行，默认为文件末尾）

    返回:
        成功时返回指定行范围的文本内容（含行号信息）；失败时返回错误信息字符串
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        total = len(lines)
        if start_line < 1:
            start_line = 1
        if end_line is None or end_line > total:
            end_line = total
        if start_line > total:
            return f"起始行 {start_line} 超出文件总行数 {total}"
        selected = lines[start_line - 1:end_line]
        result = "".join(selected)
        return f"（显示第 {start_line} 行至第 {end_line} 行，共 {len(selected)} 行）\n" + result
    except FileNotFoundError:
        return f"错误：文件 '{path}' 不存在"
    except Exception as e:
        return f"读取文件失败：{e}"


@mcp.tool()
def read_first_lines(path: str, n: int = 10) -> str:
    """
    读取文件的前 n 行。

    参数:
        path: 文件路径
        n: 要读取的行数（默认为10）

    返回:
        成功时返回前 n 行内容；文件为空时返回提示；失败时返回错误信息字符串
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = []
            for i in range(n):
                line = f.readline()
                if not line:
                    break
                lines.append(line)
        if not lines:
            return "文件为空或不存在"
        return "".join(lines)
    except FileNotFoundError:
        return f"错误：文件 '{path}' 不存在"
    except Exception as e:
        return f"读取文件失败：{e}"


@mcp.tool()
def read_last_lines(path: str, n: int = 10) -> str:
    """
    读取文件的最后 n 行。

    参数:
        path: 文件路径
        n: 要读取的行数（默认为10）

    返回:
        成功时返回最后 n 行内容；文件为空时返回提示；失败时返回错误信息字符串
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if not lines:
            return "文件为空"
        tail = lines[-n:]
        return "".join(tail)
    except FileNotFoundError:
        return f"错误：文件 '{path}' 不存在"
    except Exception as e:
        return f"读取文件失败：{e}"


@mcp.tool()
def write_lines(path: str, lines: list[str], mode: str = "overwrite") -> str:
    """
    将多行文本写入文件。

    参数:
        path: 文件路径
        lines: 要写入的行列表（包含换行符或由工具添加）
        mode: 写入模式，"overwrite"（覆盖，默认）或 "append"（追加）

    返回:
        成功时返回确认信息（含行数）；失败时返回错误信息字符串
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if mode == "overwrite":
            with open(path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return f"成功写入 {len(lines)} 行到文件: {path} (覆盖)"
        elif mode == "append":
            with open(path, 'a', encoding='utf-8') as f:
                f.writelines(lines)
            return f"成功追加 {len(lines)} 行到文件: {path} (追加)"
        else:
            return f"错误：未知模式 '{mode}'，请使用 'overwrite' 或 'append'"
    except Exception as e:
        return f"写入文件失败：{e}"



@mcp.tool()
def list_directory(path: str = ".") -> str:
    """
    列出目录中的文件和子目录。

    参数:
        path: 目录路径（默认为当前目录 "."）

    返回:
        成功时返回 JSON 格式的文件/目录列表（目录以 '/' 结尾）；失败时返回错误信息字符串
    """
    try:
        items = os.listdir(path)
        result = []
        for item in items:
            full = os.path.join(path, item)
            if os.path.isdir(full):
                result.append(f"{item}/")
            else:
                result.append(item)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except FileNotFoundError:
        return f"错误：目录 '{path}' 不存在"
    except PermissionError:
        return f"错误：没有权限访问目录 '{path}'"
    except Exception as e:
        return f"列出目录失败：{e}"


@mcp.tool()
def create_directory(path: str) -> str:
    """
    创建目录（包括所有必要的父目录）。

    参数:
        path: 要创建的目录路径

    返回:
        成功时返回确认信息；失败时返回错误信息字符串
    """
    try:
        os.makedirs(path, exist_ok=True)
        return f"目录已创建: {path}"
    except Exception as e:
        return f"创建目录失败：{e}"


@mcp.tool()
def file_info(path: str) -> str:
    """
    获取文件或目录的元信息（大小、修改时间等）。

    参数:
        path: 文件或目录路径

    返回:
        成功时返回 JSON 格式的元信息（包含路径、大小、类型、修改时间、创建时间、权限）；
        失败时返回错误信息字符串
    """
    try:
        stat = os.stat(path)
        info = {
            "path": path,
            "size": stat.st_size,
            "is_dir": os.path.isdir(path),
            "is_file": os.path.isfile(path),
            "modified": stat.st_mtime,
            "created": stat.st_ctime,
            "permissions": oct(stat.st_mode)[-3:]
        }
        return json.dumps(info, ensure_ascii=False, indent=2)
    except FileNotFoundError:
        return f"错误：路径 '{path}' 不存在"
    except Exception as e:
        return f"获取信息失败：{e}"


@mcp.tool()
def delete_file(path: str) -> str:
    """
    删除文件或空目录（谨慎使用）。

    参数:
        path: 要删除的文件或目录路径

    返回:
        成功时返回确认信息；失败时返回错误信息字符串。
        注意：仅删除空目录，非空目录需先删除内部文件。
    """
    try:
        if os.path.isdir(path):
            os.rmdir(path)  # 仅删除空目录
            return f"目录已删除: {path}"
        else:
            os.remove(path)
            return f"文件已删除: {path}"
    except FileNotFoundError:
        return f"错误：路径 '{path}' 不存在"
    except OSError as e:
        return f"删除失败，可能目录非空或权限不足：{e}"
    except Exception as e:
        return f"删除失败：{e}"



@mcp.tool()
def replace_line(path: str, line_number: int, new_content: str) -> str:
    """
    将文件中的指定行替换为新内容（行号从1开始）。

    参数:
        path: 文件路径
        line_number: 要替换的行号（从1开始）
        new_content: 新的行内容（无需包含换行符，工具会自动添加）

    返回:
        成功时返回确认信息；失败时返回错误信息字符串
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        total = len(lines)
        if line_number < 1 or line_number > total:
            return f"错误：行号 {line_number} 超出范围（文件共 {total} 行）"

        if not new_content.endswith('\n'):
            new_content += '\n'

        old_content = lines[line_number - 1]
        lines[line_number - 1] = new_content

        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        return f"第 {line_number} 行已替换\n原内容: {old_content.rstrip()}\n新内容: {new_content.rstrip()}"

    except FileNotFoundError:
        return f"错误：文件 '{path}' 不存在"
    except PermissionError:
        return f"错误：没有权限写入文件 '{path}'"
    except Exception as e:
        return f"替换行失败：{e}"


@mcp.tool()
def replace_lines_range(path: str, start_line: int, end_line: int, new_lines: list[str]) -> str:
    """
    将文件中指定行范围替换为多行新内容（行号从1开始）。

    参数:
        path: 文件路径
        start_line: 起始行号（从1开始）
        end_line: 结束行号（包含该行）
        new_lines: 新的多行内容列表（每行无需包含换行符，工具会自动添加）

    返回:
        成功时返回确认信息；失败时返回错误信息字符串
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        total = len(lines)
        if start_line < 1 or start_line > total:
            return f"错误：起始行号 {start_line} 超出范围（文件共 {total} 行）"
        if end_line < start_line or end_line > total:
            return f"错误：结束行号 {end_line} 超出范围（文件共 {total} 行）"

        old_count = end_line - start_line + 1
        old_content = "".join(lines[start_line - 1:end_line])

        formatted_new_lines = [f"{line}\n" if not line.endswith('\n') else line for line in new_lines]

        lines[start_line - 1:end_line] = formatted_new_lines

        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        return (f"第 {start_line}-{end_line} 行已替换\n"
                f"原 {old_count} 行替换为 {len(new_lines)} 行\n"
                f"原内容: {old_content.rstrip()}")

    except FileNotFoundError:
        return f"错误：文件 '{path}' 不存在"
    except PermissionError:
        return f"错误：没有权限写入文件 '{path}'"
    except Exception as e:
        return f"替换行范围失败：{e}"


@mcp.tool()
def replace_lines_by_content(path: str, old_text: str, new_text: str, exact_match: bool = True) -> str:
    """
    按内容搜索并替换行。

    参数:
        path: 文件路径
        old_text: 要搜索的文本
        new_text: 替换后的文本
        exact_match: True 表示整行完全匹配才替换；False 表示包含该文本的行即替换（默认为 True）

    返回:
        成功时返回替换的统计信息；失败时返回错误信息字符串
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        replaced = 0
        new_lines = []
        for line in lines:
            if exact_match:
                # 整行匹配时去掉换行符再比较
                if line.rstrip('\n') == old_text:
                    new_line = f"{new_text}\n" if not new_text.endswith('\n') else new_text
                    new_lines.append(new_line)
                    replaced += 1
                else:
                    new_lines.append(line)
            else:
                if old_text in line:
                    new_line = line.replace(old_text, new_text)
                    new_lines.append(new_line)
                    replaced += 1
                else:
                    new_lines.append(line)

        if replaced == 0:
            return f"未找到匹配的内容: '{old_text}'"

        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

        return f"成功替换 {replaced} 行"

    except FileNotFoundError:
        return f"错误：文件 '{path}' 不存在"
    except PermissionError:
        return f"错误：没有权限写入文件 '{path}'"
    except Exception as e:
        return f"替换失败：{e}"


@mcp.tool()
def insert_line(path: str, line_number: int, content: str) -> str:
    """
    在文件指定位置插入新行（行号从1开始）。

    参数:
        path: 文件路径
        line_number: 插入位置的行号（从1开始）
        content: 要插入的行内容（无需包含换行符，工具会自动添加）

    返回:
        成功时返回确认信息；失败时返回错误信息字符串
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        total = len(lines)
        if line_number < 1:
            line_number = 1
        if line_number > total + 1:
            return f"错误：行号 {line_number} 超出范围（文件共 {total} 行，最多可插入第 {total + 1} 行）"

        if not content.endswith('\n'):
            content += '\n'

        lines.insert(line_number - 1, content)

        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        return f"已在第 {line_number} 行前插入新行"

    except FileNotFoundError:
        return f"错误：文件 '{path}' 不存在"
    except PermissionError:
        return f"错误：没有权限写入文件 '{path}'"
    except Exception as e:
        return f"插入行失败：{e}"


# 上传附件以只读 resource 暴露：API 层写入 uploads，这里读取
@mcp.resource(
    "upload://{name}",
    description="读取用户上传的对话附件",
    mime_type="text/plain",
)
def read_attachment(name: str) -> str:
    """读取上传目录中的附件内容（name 已由框架 unquote）。"""
    return read_upload(name)


if __name__ == "__main__":
    from src.agents.mcps.registry import FILE_SERVER, serve

    serve(mcp, FILE_SERVER)
