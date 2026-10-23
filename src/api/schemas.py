from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息；resume=True 时为对上一轮提问的回答")
    thread_id: str | None = Field(None, description="会话 ID，用于多轮记忆；缺省由服务端生成")
    system_prompt: str | None = Field(None, description="可选的自定义系统提示词（仅首次会话生效）")
    resume: bool = Field(
        False,
        description="是否为回答上一轮 ask_user 提问（中断续答）；此时 message 作为 resume 值交回图",
    )
    resource_uris: list[str] | None = Field(
        None,
        description="本轮作为参考资料附带的 MCP resource URI 列表。"
        "由应用/用户选择，模型不参与选择，内容以文本形式注入本轮输入。",
    )
