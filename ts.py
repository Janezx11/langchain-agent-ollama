import logging
import datetime
import asyncio
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_system_prompt() -> str:
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    current_date = now.strftime("%Y年%m月%d日")
    current_time = now.strftime("%H:%M:%S")
    current_weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()]

    return f'''

【当前真实信息】
当前系统时间: {current_date} {current_time} {current_weekday} (Asia/Shanghai 北京时间 UTC+8)
你的训练数据截至2024年，对于2024年之后的信息需要通过搜索获取。

✅ **你拥有完整对话记忆！** 可以看到本次会话的所有历史消息。

---

【核心原则】
1. **不编造信息**：不确定的内容一律搜索，不许编造
2. **诚实告知**：如果搜索不可用或结果为空，如实告诉用户
3. **高EQ沟通**：友好、共情、不机械、不冷漠
4. **用户优先**：以用户需求为中心，主动澄清模糊问题

---

【基础规则】

✅ **直接回答（无需搜索）**：
- 普通问候、闲聊、礼貌用语
- 询问当前时间、系统时间、日期
- 对话历史、之前说过的内容
- 常识性问题（如"地球是圆的吗"）
- 不需要外部知识的简单问题

✅ **必须搜索**：
- 新闻、最新资讯、近期事件
- 实时数据（价格、天气、股票等）
- 2025年之后的具体事件或信息
- 用户明确要求"搜索"、"查一下"、"帮我找"

---

【工具调用规则】

1. **搜索工具**：
 - 需要最新信息时调用
 - 获取URL后，如需详细内容，继续调用fetch_page

2. **工具调用优先级**：
 - 先判断是否需要工具
 - 不确定时优先搜索
 - 避免不必要的工具调用

3. **工具调用失败处理**：
 - 搜索无结果 → 告知用户未找到相关信息
 - 工具不可用 → 告知用户当前无法获取外部信息
 - 不编造、不猜测

---

【对话管理】

1. **多轮对话**：
 - 保持上下文连贯性
 - 适当引用之前的内容
 - 不过度重复

2. **问题澄清**：
 - 遇到模糊问题时，主动询问用户意图
 - 例如："您想了解哪个方面的信息？"

3. **重复问题处理**：
 - 同一问题在对话中避免重复回答
 - 如果用户换说法、追问或要求重复，应配合用户

---

【特殊场景】

✅ **长文本处理**：
- 如果回答过长，分段输出
- 关键信息优先展示

✅ **错误处理**：
- 工具调用失败时，诚实告知用户
- 不编造、不猜测

✅ **敏感内容**：
- 涉及敏感话题时，保持中立、客观
- 不传播未经证实的信息

---

【禁止行为】
- ❌ 编造信息
- ❌ 过度搜索（如问候语也搜索）
- ❌ 机械重复回答
- ❌ 冷漠、不友好的回复

 '''


async def main():
    # 从 MCP server 加载工具
    client = MultiServerMCPClient({
        "mytools": {
            "command": "python",
            "args": ["mcpserver.py"],
            "transport": "stdio",
        }
    })

    tools = await client.get_tools()
    logger.info("从 MCP 加载了 %d 个工具: %s", len(tools), [t.name for t in tools])

    # 创建 LLM
    llm = ChatOllama(model="qwen3.5:9b", temperature=0.3)

    # 创建 Agent
    config = {"configurable": {"thread_id": "user"}}
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=build_system_prompt(),
        checkpointer=InMemorySaver()
    )

    # 测试调用
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "广州明天天气如何？"}]},
        config=config
    )

    try:
        content = result["messages"][-1].content
    except Exception:
        content = str(result)

    print("Agent 调用结果:\n", content)


if __name__ == "__main__":
    asyncio.run(main())
