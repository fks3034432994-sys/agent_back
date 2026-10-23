"""GUI 自动化 MCP 服务（把 touch_graph 子图包装成一个工具）。

GUI 子图保持**同步**：pyautogui 点击、PIL 截图、YOLO 推理都是阻塞调用，且必须在
独立进程里跑（torch 首次加载数秒~数十秒），放这里正好与主进程的事件循环隔离。

与旧版的三点区别：
1. HTTP 常驻服务：不再是随主进程生灭的 stdio 子进程；
2. 取消改为**协作式**：以前客户端断开就杀掉整个 GUI MCP 进程，现在服务常驻、不能整体
   杀掉（会连带杀死其它会话的任务），改为把取消信号（Event）注入子图 state，
   由节点在步骤边界检查后主动收尾；
3. 并发保护：GUI 是天然独占资源（同一套鼠标键盘 + 固定的截图文件名），因此同时只允许
   一个任务执行；第二个直接返回"繁忙"而不是排队——排队会让取消信号迟到很久才被处理。
"""

import sys
import threading
from pathlib import Path

# 先补 sys.path：直接运行本文件时 sys.path[0] 是 mcps/，否则找不到 src 包
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from fastmcp import FastMCP

from src.agents.graph.touch_graph import TouchMessagesState, touch_agent
from src.agents.mcps.cancel_registry import CancelRegistry, attach_cancel_tool

mcp = FastMCP("GUI")

REGISTRY = CancelRegistry()
attach_cancel_tool(mcp, REGISTRY)

# GUI 任务独占鼠标键盘和截图路径，同一时刻只允许一个
_GUI_BUSY = threading.Lock()


def _initial_state(task_desc: str, cancel) -> TouchMessagesState:
    return {
        "messages": [],
        "user_input_task": task_desc,
        "plan": [],
        "current_step_index": 0,
        "current_images": {},
        "element_map": {},
        "action_history": [],
        "task_completed": False,
        "step_count": 0,
        "max_steps": 200,
        "image_width": 0,
        "image_height": 0,
        "last_action": None,
        "goto": "check",
        # cancel_event 不可序列化：本图没有 checkpointer，state 只在内存流转
        "cancel_event": cancel,
        "aborted": False,
    }


@mcp.tool()
def execute_gui_task(task_desc: str, task_id: str = "") -> str:
    """
    执行 GUI 自动化任务。

    参数:
        task_desc: 用户任务描述，如 "打开微信并发送消息给张三"
        task_id: 内部参数，由主进程注入用于取消，模型无需填写

    返回:
        任务执行结果的描述。
    """
    if not _GUI_BUSY.acquire(blocking=False):
        return "已有 GUI 任务正在执行，请等它结束后再试（GUI 操作独占鼠标键盘）"

    cancel = REGISTRY.register(task_id)
    try:
        result = touch_agent.invoke(_initial_state(task_desc, cancel))
        history = " -> ".join(result.get("action_history", []))
        if result.get("aborted"):
            return f"GUI 任务已被用户中断：{task_desc}\n已执行: {history}"
        if result.get("task_completed", False):
            return f"GUI 任务完成：{task_desc}\n操作历史: {history}"
        return f"GUI 任务未完成：{task_desc}\n当前状态: {result.get('last_action', '未知')}"
    except Exception as e:
        return f"GUI 任务执行失败：{e}"
    finally:
        REGISTRY.unregister(task_id)
        _GUI_BUSY.release()


if __name__ == "__main__":
    from src.agents.mcps.registry import GUI_SERVER, serve

    serve(mcp, GUI_SERVER)
