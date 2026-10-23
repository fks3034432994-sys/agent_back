import base64
import pyautogui
from pathlib import Path
from typing import Tuple, Dict, Optional
from src.agents.utils import click_at_image_coords
from langchain.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from typing_extensions import TypedDict, Annotated
from src.agents.utils import screen_shot
# yolo_draw 从子模块直接导入：utils 包不导出它，避免 import 包就加载 torch
from src.agents.utils.yolo.test_yolo import yolo_draw
from src.agents.models import check_model, click_model, task_planning_model
import operator
import time
import pyperclip
import sys


# 用绝对路径，相对路径会随进程 CWD 失效
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DRAW_PATH = str(PROJECT_ROOT / "src" / "agents" / "utils" / "screen" / "image" / "full_screenshot.png")
YOLO_PATH = str(PROJECT_ROOT / "src" / "agents" / "utils" / "yolo" / "image" / "annotated_clean.png")


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class ElementInfo(TypedDict):
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    conf: float


class TouchMessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_input_task: str
    plan: list[str]
    current_step_index: int
    current_images: Dict[str, str]
    element_map: Dict[int, ElementInfo]
    action_history: list[str]
    task_completed: bool
    step_count: int
    max_steps: int
    image_width: int
    image_height: int
    last_action: Optional[dict]
    goto: str
    # cancel_event 由 gui_mcp_server 注入；本图没有 checkpointer，state 可以带不可序列化对象
    # aborted 区分被取消与正常完成
    cancel_event: Optional[object]
    aborted: bool


def _is_cancelled(state: TouchMessagesState) -> bool:
    """是否收到取消信号。

    只在**步骤边界**检查：单个动作（点击、截图、YOLO 推理、LLM 调用）无法中途打断，
    但每一步之间都能停下来，实际取消延迟在秒级。
    """
    event = state.get("cancel_event")
    return bool(event is not None and getattr(event, "is_set", None) and event.is_set())


def _aborted_update(state: TouchMessagesState) -> dict:
    """取消时的统一返回：结束任务并留下可读的收尾消息。"""
    print("收到取消信号，停止 GUI 任务", file=sys.stderr)
    return {
        "task_completed": True,
        "aborted": True,
        "messages": [AIMessage(content="任务已被用户中断")],
    }


def user_input(state: TouchMessagesState):
    if not state["user_input_task"]:
        task = input("输入任务：")
    else:
        task = state["user_input_task"]

    screen_shot()
    draw_path = DRAW_PATH
    element_map, img_w, img_h = yolo_draw(draw_path)
    yolo_path = YOLO_PATH

    return {
        "messages": [HumanMessage(content=task)],
        "user_input_task": task,
        "current_images": {
            "yolo": image_to_base64(yolo_path),
            "original": image_to_base64(draw_path)
        },
        "element_map": element_map,
        "image_width": img_w,
        "image_height": img_h,
        "step_count": state.get("step_count", 0) + 1,
        "task_completed": False,
        "max_steps": state.get("max_steps", 20),
        "last_action": None,
        "plan": [],
        "current_step_index": 0,
        "action_history": [],
        "goto": "check"
    }


def task_planning(state: TouchMessagesState):
    if state.get("task_completed", False):
        return {"plan": []}
    plan_prompt = f"""
    你是一个 GUI 自动化助手。
    用户任务：{state['user_input_task']}
    当前屏幕截图如下。
    请根据截图和窗口信息，判断为了推进任务，**下一步**最应该执行什么操作。
    如果任务已经完成，输出 "DONE"。
    如果当前界面缺少必要元素，可以输出探索性操作，如：
      - "显示桌面"   （按 Win+D）
      - "打开开始菜单"
      - "最小化所有窗口"
    如果需要在输入框中输入文字，请输出 "输入: 要输入的文字"。
    输出格式：一个字符串，例如："点击任务栏微信图标" 或 "显示桌面" 或 "输入: deepseek"。
    只输出操作描述，不要输出其他文字。
    """
    current_images = state.get("current_images", {})
    image_content = [
        {"type": "image", "source_type": "base64", "data": current_images.get("original", ""), "mime_type": "image/png"}
    ]
    image_message = HumanMessage(content=image_content)
    response = task_planning_model.invoke([SystemMessage(content=plan_prompt), image_message])
    next_step = response.content.strip()
    print(f"plan: {next_step}", file=sys.stderr)
    if "DONE" in next_step.upper():
        return {"task_completed": True, "plan": []}

    return {
        "plan": [next_step],
        "current_step_index": 0,
        "action_history": state.get("action_history", [])
    }


def llm_call(state: TouchMessagesState):
    plan = state.get("plan", [])
    if not plan:
        return {"last_action": None, "goto": "check"}

    current_intent = plan[0]
    action_history_str = " -> ".join(state.get("action_history", [])) or "无"

    system_prompt = f"""
    你是一个 GUI 自动化助手。
    用户最终任务：{state['user_input_task']}
    当前操作意图：{current_intent}
    操作历史：{action_history_str}
    请根据当前界面截图，选择最合适的操作以实现该意图。
    如果当前界面已经满足该意图，输出 action_type: "skip"（表示无需操作）。

    - 如果意图是点击某个元素（特别是桌面图标），输出 action_type: "click" 并指定 element_id。
      如果是双击（例如打开桌面快捷方式），请将 clicks 设置为 2（默认是1）。
    - 如果意图是输入文字，输出 action_type: "type" 并提供 text 字段。
    - 如果意图是按键（如回车），输出 action_type: "key" 并提供 key 字段。
    - 如果意图是快捷键（如 Win+D），输出 action_type: "hotkey" 并提供 key 字段（如 "win+d"）。

    桌面图标通常需要双击才能打开，请根据实际情况设置 clicks 值。
    """

    current_images = state.get("current_images", {})
    image_content = [
        {"type": "text", "text": "第一张是标注了编号的界面，第二张是原图："},
        {"type": "image", "source_type": "base64", "data": current_images.get("yolo", ""), "mime_type": "image/png"},
        {"type": "image", "source_type": "base64", "data": current_images.get("original", ""),
         "mime_type": "image/png"},
    ]
    image_message = HumanMessage(content=image_content)

    try:
        action_param = click_model.invoke([SystemMessage(content=system_prompt), image_message])
        action_dict = action_param.model_dump()
        print(f"{current_intent} -> {action_dict}", file=sys.stderr)
        if action_param.action_type == "click" and action_param.element_id is None:
            action_param.element_id = -1
    except Exception as e:
        print(f"结构化输出失败，兜底点击: {e}", file=sys.stderr)
        action_dict = {"action_type": "click", "element_id": -1, "button": "left", "clicks": 1, "interval": 0.0}

    return {
        "messages": [AIMessage(content=f"AI 生成动作: {action_dict}")],
        "current_images": {},
        "last_action": action_dict,
        "step_count": state.get("step_count", 0) + 1,
        "goto": "check"
    }


def execute_exploration(intent: str) -> bool:
    intent_lower = intent.lower()
    if "显示桌面" in intent_lower:
        pyautogui.hotkey('win', 'd')
        print("显示桌面 (Win+D)", file=sys.stderr)
        return True
    elif "打开开始菜单" in intent_lower or "开始菜单" in intent:
        pyautogui.hotkey('win')
        print("打开开始菜单 (Win)", file=sys.stderr)
        return True
    elif "最小化" in intent_lower:
        pyautogui.hotkey('win', 'd')
        print("最小化所有窗口 (Win+D)", file=sys.stderr)
        return True
    else:
        return False


def action_node(state: TouchMessagesState):
    if _is_cancelled(state):
        # 取消后不能再产生任何鼠标/键盘动作
        return {**_aborted_update(state), "goto": "check"}

    last_action = state.get("last_action")
    if not last_action:
        print("没有动作参数", file=sys.stderr)
        return {"goto": "check", "step_count": state.get("step_count", 0) + 1}

    action_type = last_action.get("action_type")
    print(f"动作: {action_type}", file=sys.stderr)

    if action_type in ("type", "key", "hotkey"):
        if action_type == "type":
            text = last_action.get("text", "")
            if any(ord(c) > 127 for c in text):
                pyperclip.copy(text)  # 复制到系统剪贴板
                pyautogui.hotkey('ctrl', 'v')  # 模拟 Ctrl+V 粘贴
                print(f"粘贴输入: {text}", file=sys.stderr)
            else:
                pyautogui.typewrite(text)  # 纯 ASCII 直接按键
                print(f"按键输入: {text}", file=sys.stderr)
            new_history = state.get("action_history", []) + [f"输入: {text}"]

        elif action_type == "key":
            key = last_action.get("key", "")
            pyautogui.press(key)
            print(f"按键: {key}", file=sys.stderr)
            new_history = state.get("action_history", []) + [f"按键: {key}"]

        elif action_type == "hotkey":
            key_str = last_action.get("key", "")
            keys = key_str.split('+')
            pyautogui.hotkey(*keys)
            print(f"快捷键: {key_str}", file=sys.stderr)
            new_history = state.get("action_history", []) + [f"快捷键: {key_str}"]
        else:
            new_history = state.get("action_history", []) + ["未知键盘操作"]

        return {
            "action_history": new_history,
            "step_count": state.get("step_count", 0) + 1,
            "goto": "check"
        }

    if action_type == "click":
        element_id = last_action.get("element_id")
        if element_id == 0:
            print("skip，界面已满足意图", file=sys.stderr)
            new_history = state.get("action_history", []) + ["步骤已满足"]
            return {
                "action_history": new_history,
                "step_count": state.get("step_count", 0) + 1,
                "goto": "check"
            }
        if element_id is None or element_id == -1:
            print("点击不可用，改走探索性操作", file=sys.stderr)
            plan = state.get("plan", [])
            if plan and execute_exploration(plan[0]):
                new_history = state.get("action_history", []) + [f"探索: {plan[0]}"]
                return {
                    "action_history": new_history,
                    "step_count": state.get("step_count", 0) + 1,
                    "goto": "check"
                }
            else:
                new_history = state.get("action_history", []) + ["无法执行，返回 user_input"]
                return {
                    "action_history": new_history,
                    "step_count": state.get("step_count", 0) + 1,
                    "goto": "user_input"
                }

        element = state["element_map"].get(element_id)
        if not element:
            print(f"编号 {element_id} 不在当前截图", file=sys.stderr)
            new_history = state.get("action_history", []) + [f"编号 {element_id} 不存在"]
            return {
                "action_history": new_history,
                "step_count": state.get("step_count", 0) + 1,
                "goto": "check"
            }

        cx, cy = element["center"]
        img_w = state["image_width"]
        img_h = state["image_height"]

        clicks = last_action.get("clicks", 1)
        click_at_image_coords(
            img_x=cx,
            img_y=cy,
            img_width=img_w,
            img_height=img_h,
            button=last_action.get("button", "left"),
            clicks=clicks,
            interval=last_action.get("interval", 0.0)
        )

        action_desc = f"{clicks}击编号 {element_id}（{last_action.get('button', 'left')}键）"
        new_history = state.get("action_history", []) + [action_desc]
        print(action_desc, file=sys.stderr)
        time.sleep(0.3)
        return {
            "action_history": new_history,
            "step_count": state.get("step_count", 0) + 1,
            "goto": "check"
        }

    return {"goto": "check", "step_count": state.get("step_count", 0) + 1}


def check_node(state: TouchMessagesState):
    if _is_cancelled(state):
        return _aborted_update(state)
    if state.get("step_count", 0) >= state.get("max_steps", 20):
        print(f"达到最大步数 {state['max_steps']}，结束", file=sys.stderr)
        ai_message = AIMessage(content=f"达到最大步数 {state['max_steps']}，强制结束")
        return {"task_completed": True, "messages": [ai_message]}
    if state.get("task_completed", False):
        return {"task_completed": True}
    time.sleep(0.5)
    screen_shot()
    draw_path = DRAW_PATH
    check_prompt = f"""
    当前任务：{state['user_input_task']}
    操作历史：{' -> '.join(state.get('action_history', []))}
    请根据最新截图判断任务是否已经完成。
    如果已完成，只回复 "已完成"。
    如果未完成，回复 "未完成"。
    """
    image_content = [
        {"type": "image", "source_type": "base64", "data": image_to_base64(draw_path), "mime_type": "image/png"},
    ]
    check_message = HumanMessage(content=image_content)
    response = check_model.invoke([SystemMessage(content=check_prompt), check_message])
    reply = response.content.strip()
    print(f"check: {reply}", file=sys.stderr)

    if "已完成" in reply:
        print("任务完成", file=sys.stderr)
        return {"task_completed": True}

    element_map, img_w, img_h = yolo_draw(draw_path)
    yolo_path = YOLO_PATH
    return {
        "task_completed": False,
        "current_images": {
            "yolo": image_to_base64(yolo_path),
            "original": image_to_base64(draw_path)
        },
        "element_map": element_map,
        "image_width": img_w,
        "image_height": img_h,
        "last_action": None,
        "step_count": state.get("step_count", 0) + 1,
        "goto": "check"
    }


def goto_or_continue(state: TouchMessagesState) -> str:
    if state.get("goto") == "user_input":
        return "user_input"
    return "continue"


def should_continue(state: TouchMessagesState) -> str:
    if state.get("task_completed", False):
        return "end"
    return "continue"


agent_builder = StateGraph(TouchMessagesState)

agent_builder.add_node("user_input", user_input)
agent_builder.add_node("task_planning", task_planning)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("action", action_node)
agent_builder.add_node("check", check_node)

agent_builder.add_edge(START, "user_input")
agent_builder.add_edge("user_input", "task_planning")
agent_builder.add_edge("task_planning", "llm_call")
agent_builder.add_edge("llm_call", "action")

agent_builder.add_conditional_edges(
    "action",
    goto_or_continue,
    {
        "continue": "check",
        "user_input": "user_input"
    }
)

agent_builder.add_conditional_edges(
    "check",
    should_continue,
    {
        "continue": "task_planning",
        "end": END
    }
)

touch_agent = agent_builder.compile()

if __name__ == "__main__":
    try:
        graph_png = touch_agent.get_graph().draw_mermaid_png()
        with open("langgraph_hello.png", "wb") as f:
            f.write(graph_png)
        print("Graph visualization saved as 'langgraph_hello.png'", file=sys.stderr)
    except Exception as e:
        print(f"绘图失败: {e}", file=sys.stderr)

    initial_state: TouchMessagesState = {
        "messages": [],
        "user_input_task": "",
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
        "cancel_event": None,
        "aborted": False
    }

    result = touch_agent.invoke(initial_state)
    print("\n=== 最终结果 ===", file=sys.stderr)
    print("任务完成状态:", result["task_completed"], file=sys.stderr)
    print("操作历史:", " -> ".join(result.get("action_history", [])), file=sys.stderr)
    if result["messages"]:
        print("最后消息:", result["messages"][-1].content, file=sys.stderr)
