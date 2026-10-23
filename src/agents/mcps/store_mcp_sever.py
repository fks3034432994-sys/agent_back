import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# 先补 sys.path：直接运行本文件时 sys.path[0] 是 mcps/，否则找不到 src 包
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

load_dotenv()

from fastmcp import FastMCP
from src.agents.checkpoint import (
    CONSTRAINT_NAMESPACE,
    MATERIAL_NAMESPACE,
    get_store,
)

mcp = FastMCP("store")

# 独立进程里拿不到 langgraph 注入的 ToolRuntime.store，直接用同一 DB_URI 上的同步 store
# 主进程用异步版（见 checkpoint.postgresql_saver）
# 工具写成同步函数，FastMCP 会派发到工作线程，阻塞调用不会卡住事件循环
# namespace 第二级区分用途：constraint 常驻注入不检索，material 按需语义检索
_NAMESPACE_BY_SCOPE = {
    "constraint": CONSTRAINT_NAMESPACE,
    "material": MATERIAL_NAMESPACE,
}


@mcp.tool()
def set_store(
    key: str,
    value: str,
    scope: Literal["constraint", "material"] = "constraint",
) -> bool | str:
    """记住并持久化一条关于用户的长期信息（偏好、习惯、身份、约束等）。

    scope 选择（决定这条记忆以后怎么被使用）：
    - `constraint`（默认）：要求 agent **今后怎么做**的规则，如回答语言、称呼方式、
      回复风格、输出格式、禁忌。这类会被常驻注入系统提示词，每一轮都要遵守。
    - `material`：**明确**是关于用户或世界的事实描述，如姓名、职位、项目背景、经历、
      以后可能用得到的资料。这类不会被注入，只在被问到或被需要时按语义检索。
    - **分不清是事实还是行为要求时，一律用 `constraint`**（默认值）。

    使用场景：
    - 用户明确表达"记住/以后都/我习惯/我喜欢/不要再/请始终"等长期性意图时调用。
    - 用户透露稳定的个人信息时调用：影响回复方式的（称呼、回答语言、风格、忌口）
      用 `constraint`；纯事实的（姓名、职位、技术栈、项目背景）用 `material`。
    - 仅在信息具备长期价值时调用；一次性、任务内、临时性的内容不要写入。

    key 命名要求：
    - 使用简短、稳定、语义唯一的英文小写标识，单词间用下划线，如 `nickname`、`preferred_language`、`reply_style`、`dietary_restriction`。
    - 同一类信息必须复用同一个 key，避免 `语言` / `language` / `preferred_language` 并存。
    - key 表示"主题"，value 表示"结论"，不要把整句话塞进 key。

    value 写法要求：
    - 用一句简洁完整的中文陈述结论，如"称呼用户为『老王』"、"回答一律使用简体中文"。
    - 不要写"用户说……"这类过程描述，直接写最终要遵守的规则。

    返回值与冲突处理：
    - 返回 True：写入成功，或新旧内容一致（幂等，无需重复告知用户）。
    - 返回字符串：检测到与已有偏好冲突，含义是本次**未写入**。此时必须：
      1. 向用户原样转述返回的冲突内容（原有偏好 / 新偏好），并询问是否用新偏好覆盖旧偏好，使用ask_user工具进行提问；
      2. 不要断言已更新，也不要用相同 key 再次调用本工具（只会再次返回冲突）；
      3. 仅在用户明确同意覆盖后，改用 `set_store_cover` 工具，以**相同的 key 和 value** 完成覆盖；
         用户未明确表态或选择保留原偏好时，保持原值不变，不要调用任何写入工具。
    """
    store = get_store()
    namespace = _NAMESPACE_BY_SCOPE[scope]
    existing = store.get(namespace, key)
    if existing:
        old_value = existing.value["value"]
        if old_value != value:
            return (
                f"检测到偏好冲突：\n"
                f"- 原有偏好：{old_value}（记录于 {existing.value.get('updated_at', '未知')}）\n"
                f"- 新偏好：{value}\n"
                f"请确认：是否用新偏好覆盖旧偏好？"
            )
        else:
            return True
    else:
        store.put(namespace, key, {
            "value": value,
            "updated_at": datetime.now().isoformat()
        })
        return True


@mcp.tool()
def set_store_cover(
    key: str,
    value: str,
    scope: Literal["constraint", "material"] = "constraint",
) -> bool:
    """强制覆盖已存在的用户长期偏好，专用于 `set_store` 报冲突且用户明确同意覆盖之后的写入。

    调用前提（必须全部满足，否则禁止调用）：
    - 刚刚调用 `set_store` 得到的是字符串返回的冲突提示，即该次写入**未生效**；
    - 已把冲突内容（原有偏好 / 新偏好）原样转述给用户，并拿到用户对"覆盖"的明确确认
      （如"可以/覆盖/改成这个"）。
    - 用户未明确表态、或表示保留原偏好时，保持原值不变，不要调用本工具。

    key、value 与 scope 要求：
    - key 与 scope 必须与产生冲突的那次 `set_store` 调用完全一致，否则会写到另一条/另一类
      记录上，旧的冲突值依然留在原处（如 `nickname`、`preferred_language`）。
    - value 使用一句简洁完整的中文陈述结论，如"称呼用户为『老王』"、"回答一律使用简体中文"；
      不要写"用户说……"这类过程描述。确认过程中用户另有说明的，以用户最终表述为准。

    返回值：
    - 恒返回 True：旧偏好已被新偏好覆盖，此时可明确告知用户"已更新为……"。
    """
    get_store().put(_NAMESPACE_BY_SCOPE[scope], key, {
        "value": value,
        "updated_at": datetime.now().isoformat()
    })
    return True


@mcp.tool()
def get_store_by_key(
    key: str,
    scope: Literal["constraint", "material"] = "constraint",
) -> str:
    """
    查询此前保存的关于用户的长期信息（偏好、习惯、身份、约束等）。

    使用场景：
    - 回答前需要确认是否记录过某类用户偏好（称呼、语言、回复风格、忌口等）时调用。
    - 用户询问"你还记得我……吗""我的偏好是什么"时调用。
    - 不要凭印象猜测用户偏好，不确定就先调用本工具查询。

    scope 要求：
    - 必须与写入时使用的 scope 一致：约束类用 `constraint`（默认），资料类用 `material`。
    - 注意：`constraint` 类内容在会话开始时已整体注入系统提示词，一般无需再查；带
      `constraint` 主要用在需要向用户复述确切措辞、或确认某条是否已记录时。

    key 要求：
    - 必须与 `set_store` 写入时使用的 key 完全一致（如 `nickname`、`preferred_language`）。
    - 只支持精确匹配，不支持模糊或语义检索；同一主题的 key 命名需保持稳定复用。

    返回值与后续动作：
    - 返回一句中文陈述：该 key 对应的偏好内容，可直接作为事实依据使用。
    - 返回 `{key}无数据`：表示从未记录过该类信息。此时**不要**编造内容，
      应改为直接向用户询问，或在用户明确表达长期偏好后调用 `set_store` 写入。
    """
    existing = get_store().get(_NAMESPACE_BY_SCOPE[scope], key)
    if not existing:
        return f"{key}无数据"
    res = existing.value.get("value", None)
    if not res:
        return f"{key}无数据"
    return res


@mcp.tool()
def get_store_by_embedding(query: str) -> str:
    """按语义相似度检索**资料类**长期记忆（用户身份、背景、经历、项目信息等）。

    检索范围只包含 `material` 类：约束类（要求 agent 怎么做的规则，如回答语言、称呼、
    风格）不在这里——它们在会话开始时已整体注入系统提示词，不需要也不必靠本工具去找。
    用户问"我的信息""你还记得我的什么"这类问题时，用本工具查。

    与 `get_store_by_key` 的区别：
    - `get_store_by_key` 需要精确的 key 且要指定 scope；本工具只给一段自然语言描述，
      只在资料类（`material`）里按相似度找最相近的若干条。
    - 当你不确定当初用什么 key 记录、或只想知道"有没有存过相关内容"时，用本工具。

    使用场景：
    - 回答前需要核实某类资料是否已记录，但 key 未知或不确定时调用。
    - 用户提问涉及多个可能相关的主题（如"我之前跟你提过的项目信息有哪些"）时调用。

    query 要求：
    - 用一句自然语言描述你要找的信息主题，如"用户的项目背景"、"之前提到的服务地址"。
    - 不要传单个英文 key，也不要传与主题无关的闲聊内容。

    返回值与后续动作：
    - 最多返回 3 条最相近记录，格式为 `相似度: 0.xx, 内容: ...`，相似度越高越相关。
    - 空字符串表示没有任何匹配记录，此时不要编造内容，应改为询问用户或按需调用 `set_store` 记录。
    - 结果只是"相近"而非精确命中，相似度偏低的条目需谨慎采信。
    - 本工具只读，不会修改任何已保存的记忆。
    """
    results = get_store().search(
        MATERIAL_NAMESPACE,
        query=query,
        limit=3
    )
    res_str = ""
    for r in results:
        content = (r.value or {}).get("value", "")
        score = f"{r.score:.2f}" if r.score is not None else "N/A"
        res_str = res_str + f"相似度: {score}, 内容: {content}\n"
    return res_str


if __name__ == "__main__":
    # 入口缺失会让子进程导入后立即退出，报 Connection closed
    from src.agents.mcps.registry import STORE_SERVER, serve

    serve(mcp, STORE_SERVER)
