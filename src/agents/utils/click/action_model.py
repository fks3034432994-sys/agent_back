from pydantic import BaseModel, Field
from typing import Literal,Optional # 用于类型约束（Python 3.8+）

class ClickParams(BaseModel):
    x: int                     # 屏幕 X 坐标
    y: int                     # 屏幕 Y 坐标
    button: Literal['left', 'right', 'middle'] = 'left'  # 限定取值
    clicks: int = Field(1, ge=1)               # 点击次数，≥1
    interval: float = Field(0.0, ge=0.0)       # 间隔时间，≥0

class AiClickParam(BaseModel):
    """AI 输出的点击参数（不包含坐标，坐标由 element_map 转换）"""
    button_num: int = Field(..., description="界面元素的编号（对应 element_map 中的编号）")
    button: Literal['left', 'right', 'middle'] = Field('left', description="鼠标按钮类型")
    clicks: int = Field(1, ge=1, description="点击次数，1=单击，2=双击")
    interval: float = Field(0.0, ge=0.0, description="多次点击之间的间隔时间（秒）")

class ActionParam(BaseModel):
    """Agent 可以执行的动作参数"""
    action_type: Literal['click', 'type', 'key', 'hotkey'] = Field(..., description="动作类型")
    element_id: Optional[int] = Field(None, description="点击时使用的元素编号")
    text: Optional[str] = Field(None, description="type 时输入的文本")
    key: Optional[str] = Field(None, description="key/hotkey 时按键名称，如 'enter', 'tab', 'win+d'")
    button: Literal['left', 'right', 'middle'] = Field('left', description="鼠标按键")
    clicks: int = Field(1, ge=1, description="点击次数")
    interval: float = Field(0.0, ge=0.0, description="点击间隔")