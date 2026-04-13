import logging
import datetime
import asyncio
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient

# 👇 RAG 相关
from rag1 import create_rag_chain  
from langchain.tools import tool

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_NAME = "qwen3.5:9b"

# ==========================================
# RAG Tool
# ==========================================

rag_chain = create_rag_chain()

@tool
def valorant_rag(query: str) -> str:
    """
    ⚠️ 必须使用此工具回答所有 Valorant 相关问题。

    包括：
    - 英雄（Iso, Jett, Reyna, Astra）
    - 技能 / 特点 / abilities
    - 地图（Bind, Haven, Ascent）

    不允许直接回答，必须调用此工具
    """

    print("\n🔥 [RAG TOOL CALLED]:", query)

    result = rag_chain.invoke(query)

    if "NOT_FOUND" in result or "NO_CONTEXT" in result:
        return "知识库中没有相关信息"

    print("📄 [RAG RESULT]:", result[:200], "\n")

    return result


# ==========================================
# System Prompt（强化版）
# ==========================================

def build_system_prompt() -> str:
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    current_date = now.strftime("%Y年%m月%d日")
    current_time = now.strftime("%H:%M:%S")
    current_weekday = ["周一","周二","周三","周四","周五","周六","周日"][now.weekday()]

    return f"""
【当前信息】
时间: {current_date} {current_time} {current_weekday}

---

【核心规则】

1. 不允许编造信息
2. 不确定必须使用工具
3. 优先使用已有工具
4. 保持自然对话

---

【工具使用规则】

你拥有一个重要工具：

👉 valorant_rag（Valorant知识库）

使用规则：

1. 所有 Valorant 相关问题：
   - 英雄（Iso, Jett, Reyna）
   - 技能 / 特点
   - 地图

👉 必须调用 valorant_rag

2. 不允许直接回答 Valorant 知识

3. 即使你知道答案，也必须调用

4. 如果没有调用该工具 → 属于错误行为

---

【其他工具】

- 搜索 / 天气 / 实时信息 → 使用 MCP 工具

---

【对话要求】

- 自然表达
- 不机械
- 有逻辑
"""


# ==========================================
# Router（强制RAG，防偷懒🔥）
# ==========================================

def is_valorant_query(q: str):
    q = q.lower()
    return any(x in q for x in [
        "valorant", "iso", "jett", "reyna", "astra",
        "技能", "英雄", "地图"
    ])


# ==========================================
# 主程序
# ==========================================

async def main():
    print("🚀 Agent + RAG 启动")
    print("=" * 50)

    # MCP 工具
    client = MultiServerMCPClient({
        "mytools": {
            "command": "python",
            "args": ["mcpserver.py"],
            "transport": "stdio",
        }
    })

    mcp_tools = await client.get_tools()
    logger.info("MCP工具: %s", [t.name for t in mcp_tools])

    # 👇 加入 RAG Tool
    tools = mcp_tools + [valorant_rag]

    # LLM（统一模型）
    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.3
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=build_system_prompt(),
        checkpointer=InMemorySaver()
    )

    config = {"configurable": {"thread_id": "user"}}

    print("\n✅ 已就绪，输入 exit 退出\n")

    while True:
        query = input("❓ 问题: ")

        if query.lower() in ["exit", "quit"]:
            break

        # 🔥 Router（关键！）
        if is_valorant_query(query):
            print("\n⚡ 直接走 RAG（强制）")
            result = valorant_rag.invoke(query)
        else:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": query}]},
                config=config
            )

            try:
                result = result["messages"][-1].content
            except Exception:
                result = str(result)

        print("\n💡 回答:")
        print(result)
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())