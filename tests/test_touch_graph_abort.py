"""GUI 子图（touch_graph）的取消收尾与动作分发。

取消是**协作式**的：客户端断开时主进程调服务内的 cancel_task 置位 Event，
子图在步骤边界检查后收尾。这里不连模型、不碰真实鼠标键盘——所有会落到
pyautogui / 截图 / YOLO 的调用都被替换成记录器。

注意：导入本模块会连带 import ultralytics/torch（GUI 子图的真实依赖），
所以收集阶段会比其他测试慢一点；缺 torch 时整个模块跳过。
"""

import threading

import pytest

pytest.importorskip("torch", reason="GUI 子图依赖 ultralytics/torch")

from langchain.messages import AIMessage  # noqa: E402

from src.agents.graph import touch_graph as tg  # noqa: E402


def _state(**overrides) -> dict:
    """一份完整的初始状态（字段与 TouchMessagesState 对齐）。"""
    base = {
        "messages": [],
        "user_input_task": "打开记事本",
        "plan": [],
        "current_step_index": 0,
        "current_images": {},
        "element_map": {},
        "action_history": [],
        "task_completed": False,
        "step_count": 0,
        "max_steps": 200,
        "image_width": 100,
        "image_height": 100,
        "last_action": None,
        "goto": "check",
        "cancel_event": None,
        "aborted": False,
    }
    base.update(overrides)
    return base



def test_is_cancelled_requires_a_settable_event():
    assert tg._is_cancelled(_state()) is False

    event = threading.Event()
    assert tg._is_cancelled(_state(cancel_event=event)) is False

    event.set()
    assert tg._is_cancelled(_state(cancel_event=event)) is True


def test_is_cancelled_tolerates_foreign_objects():
    assert tg._is_cancelled(_state(cancel_event=object())) is False


def test_aborted_update_marks_task_done_and_leaves_a_message():
    update = tg._aborted_update(_state())

    assert update["task_completed"] is True
    assert update["aborted"] is True
    assert isinstance(update["messages"][0], AIMessage)
    assert "中断" in update["messages"][0].content



def test_action_node_does_not_click_after_cancel(monkeypatch):
    """点了停止还在继续点，是最不能接受的行为。"""
    clicks = []
    monkeypatch.setattr(tg, "click_at_image_coords", lambda **kwargs: clicks.append(kwargs))
    monkeypatch.setattr(tg.time, "sleep", lambda _seconds: None)

    event = threading.Event()
    event.set()
    update = tg.action_node(_state(
        cancel_event=event,
        last_action={"action_type": "click", "element_id": 1},
        element_map={1: {"center": (10, 20), "bbox": (0, 0, 20, 40), "conf": 0.9}},
    ))

    assert clicks == []
    assert update["aborted"] is True
    assert update["goto"] == "check"


def test_action_node_does_not_type_after_cancel(monkeypatch):
    typed = []
    monkeypatch.setattr(tg.pyautogui, "typewrite", lambda text: typed.append(text))
    monkeypatch.setattr(tg.pyautogui, "hotkey", lambda *keys: typed.append(keys))

    event = threading.Event()
    event.set()
    update = tg.action_node(_state(
        cancel_event=event,
        last_action={"action_type": "type", "text": "hello"},
    ))

    assert typed == []
    assert update["aborted"] is True


def test_check_node_short_circuits_on_cancel(monkeypatch):
    """取消后连截图都不该做（截图 + YOLO 是每步最贵的部分）。"""
    shots = []
    monkeypatch.setattr(tg, "screen_shot", lambda: shots.append(1))

    event = threading.Event()
    event.set()
    update = tg.check_node(_state(cancel_event=event, step_count=3))

    assert shots == []
    assert update["aborted"] is True
    assert update["task_completed"] is True



def test_action_node_clicks_when_not_cancelled(monkeypatch):
    clicks = []
    monkeypatch.setattr(tg, "click_at_image_coords", lambda **kwargs: clicks.append(kwargs))
    monkeypatch.setattr(tg.time, "sleep", lambda _seconds: None)

    update = tg.action_node(_state(
        last_action={"action_type": "click", "element_id": 1, "clicks": 2},
        element_map={1: {"center": (10, 20), "bbox": (0, 0, 20, 40), "conf": 0.9}},
    ))

    assert len(clicks) == 1
    assert clicks[0]["img_x"] == 10
    assert clicks[0]["img_y"] == 20
    assert clicks[0]["clicks"] == 2
    assert update["goto"] == "check"
    # 正常的动作分支不碰 aborted，这里断言它没被置位
    assert "aborted" not in update


def test_action_node_element_id_zero_means_noop(monkeypatch):
    """element_id == 0 是"该步骤已满足"的哨兵值（element_map 从 1 开始编号）。"""
    clicks = []
    monkeypatch.setattr(tg, "click_at_image_coords", lambda **kwargs: clicks.append(kwargs))

    update = tg.action_node(_state(last_action={"action_type": "click", "element_id": 0}))

    assert clicks == []
    assert "步骤已满足" in update["action_history"][-1]


def test_non_ascii_text_goes_through_clipboard(monkeypatch):
    """中文等非 ASCII 必须走剪贴板粘贴：pyautogui.typewrite 打不出 CJK。"""
    copied: list[str] = []
    hotkeys: list[tuple] = []
    monkeypatch.setattr(tg.pyperclip, "copy", lambda text: copied.append(text))
    monkeypatch.setattr(tg.pyautogui, "hotkey", lambda *keys: hotkeys.append(keys))
    monkeypatch.setattr(
        tg.pyautogui, "typewrite", lambda text: pytest.fail("非 ASCII 不应走按键输入")
    )

    update = tg.action_node(_state(last_action={"action_type": "type", "text": "你好"}))

    assert copied == ["你好"]
    assert hotkeys == [("ctrl", "v")]
    assert "输入: 你好" in update["action_history"][-1]


def test_ascii_text_uses_typewrite(monkeypatch):
    typed: list[str] = []
    monkeypatch.setattr(tg.pyautogui, "typewrite", lambda text: typed.append(text))
    monkeypatch.setattr(
        tg.pyperclip, "copy", lambda text: pytest.fail("纯 ASCII 不该动剪贴板")
    )

    tg.action_node(_state(last_action={"action_type": "type", "text": "hello"}))

    assert typed == ["hello"]



def test_routing_helpers():
    assert tg.goto_or_continue(_state(goto="user_input")) == "user_input"
    assert tg.goto_or_continue(_state(goto="check")) == "continue"
    assert tg.should_continue(_state(task_completed=True)) == "end"
    assert tg.should_continue(_state()) == "continue"
