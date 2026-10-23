import uuid

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.agents.checkpoint import close_async_persistence, init_async_persistence
from src.agents import models
from src.agents.graph import main_graph
from src.agents.graph.main_graph import MainMessagesState, init_agent
from src.asyncio_runner import run_async
from agentevals.trajectory.llm import create_async_trajectory_llm_as_judge


ANSWER_QUALITY_PROMPT = """你是公司内部助手的质量审核员。请判断下面这一次回答是否合格。

<审核标准>
只审"回答本身对不对、有没有依据、有没有帮到用户"，看这四条：

1. 直接回答了用户的问题，没有跑题或答非所问；
2. 涉及公司制度、政策、数字的具体说法，都能在工具返回的内容里找到依据，没有编造；
3. 知识库里查不到的内容，明确说明"未查到／建议向人力资源部确认"，而不是硬猜一个数；
4. 没有自相矛盾的说法。

以下情况**不算不合格**，不要因此扣分：
- 工具调用的次数或顺序与参考轨迹不同（多查一次、少查一次、换个顺序都一样）；
- 措辞、排版、详略程度与参考轨迹不同；
- 补充了参考轨迹里没写、但确实有依据的额外说明。

结构一致性由另一项独立的确定性检查负责，这里不重复评判。
</审核标准>

<参考轨迹>
下面这次轨迹只用来帮你理解"预期行为大致是什么样"，它是一个样本，不是必须逐字对齐的规格：

{reference_outputs}
</参考轨迹>

<待审轨迹>
{outputs}
</待审轨迹>
"""


async def main():
    await init_async_persistence()          # 1. 建 pg 连接池 + checkpointer/store 并 setup
    await init_agent()                      # 2. 连 MCP + bind_tools + compile
    try:
        agent = main_graph.get_main_agent()
        state: MainMessagesState = {
            "messages": [HumanMessage(content="产假有多少天？")],
            "summarized_messages": [],
            "context": {},
            "task_completed": False,
            "pending_question": "",
            "pending_tool_call_id": ""
        }
        result = await agent.ainvoke(
            state, # type: ignore[arg-type]
            config={"configurable": {"thread_id": f"trajectory-match-{uuid.uuid4().hex[:8]}"}},
        )
        judge = models.main_model.model_copy(update={
            "max_tokens": 9999999999,
            "streaming": False,
            "extra_body": {"thinking": {"type": "disabled"}},
        })
        reference_trajectory = [
            HumanMessage(content=""),
            AIMessage(content="", tool_calls=[
                {"id": "call_1", "name": "search_knowledge", "args": {"query": "产假有多少天"}},
            ]),
            ToolMessage(content="", tool_call_id="call_1"),
            AIMessage(content=""),
        ]
        evaluator = create_async_trajectory_llm_as_judge(
            judge=judge,
            prompt=ANSWER_QUALITY_PROMPT,
            # feedback_key 改成对应的名字，报告里才不会误导
            feedback_key="answer_quality",
        )
        evaluation = await evaluator(
            outputs=result["messages"],
            reference_outputs=reference_trajectory,
        )
        calls = [
            tc["name"]
            for message in result["messages"]
            for tc in (getattr(message, "tool_calls", None) or [])
        ]
        print("回答：", result["messages"][-1].content)
        print("工具：", calls)
        # 成功时也打印 reasoning，便于确认评委不是瞎给
        print(
            "评委：",
            "未评" if evaluation["score"] is None else evaluation["score"],
            "—",
            evaluation["comment"],
        )
        assert evaluation["score"] is True, evaluation.get("comment")
    finally:
        await close_async_persistence()


if __name__ == "__main__":
    run_async(main())
