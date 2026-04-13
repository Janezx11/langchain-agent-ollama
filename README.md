# 🤖 AI Agent 智能问答系统

> 基于 LangChain + RAG + MCP 的多工具 AI Agent，支持本地大模型部署，具备知识库检索、网页搜索、天气查询等能力。

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────┐
│                    用户输入                            │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │   Intent Router │ ◄── 关键词分类 / 意图识别
              └───┬────────┬───┘
                  │        │
         RAG 问题 │        │ 通用问题
                  ▼        ▼
    ┌──────────────┐  ┌──────────────────┐
    │  RAG Chain   │  │  Agent (LLM)     │
    │              │  │                  │
    │  Query       │  │  System Prompt   │
    │  Rewrite     │  │  + Tool Dispatch │
    │      ↓       │  │                  │
    │  ChromaDB    │  │  ┌────────────┐  │
    │  Vector      │  │  │ MCP Tools  │  │
    │  Search      │  │  │            │  │
    │      ↓       │  │  │ • web_search│  │
    │  LLM 生成    │  │  │ • fetch_page│  │
    └──────────────┘  │  │ • weather  │  │
                      │  └────────────┘  │
                      │  ┌────────────┐  │
                      │  │ RAG Tool   │  │
                      │  └────────────┘  │
                      └──────────────────┘
                               │
                               ▼
                      ┌────────────────┐
                      │    回复用户     │
                      └────────────────┘
```

## ✨ 核心特性

### 🧠 Agent 智能调度
- 基于 **LangGraph** 构建有状态的 Agent 工作流
- 支持多工具动态调度，LLM 自主决策调用时机
- InMemorySaver 实现对话上下文持久化

### 📚 RAG 检索增强生成
- **文档分块**：RecursiveCharacterTextSplitter（chunk_size=500, overlap=100）
- **向量化**：Ollama Embeddings（nomic-embed-text）
- **向量存储**：ChromaDB，支持增量更新
- **检索策略**：MMR（Maximal Marginal Relevance），兼顾相关性与多样性
- **Query Rewrite**：中英文混合查询语义增强

### 🔌 MCP 工具服务器
- 基于 **Model Context Protocol** 标准化工具接口
- 内置工具：网页搜索、页面抓取、天气查询
- Session 管理 + 自动重试，保证稳定性

### 🛡️ 双层防幻觉
- **系统级 Prompt 约束**：强制模型调用工具而非编造
- **Router 强制路由**：关键词分类，确保特定领域问题走 RAG

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| Agent 框架 | LangChain, LangGraph |
| 大模型 | Ollama (qwen3.5:9b) |
| 向量数据库 | ChromaDB |
| Embedding | nomic-embed-text (Ollama) |
| 工具协议 | MCP (Model Context Protocol) |
| 文档解析 | UnstructuredMarkdownLoader |
| 网页解析 | BeautifulSoup4 |
| 异步框架 | Python asyncio |

## 📁 项目结构

```
.
├── main.py              # Agent 主程序（入口）
├── rag1.py              # RAG 检索链
├── mcpserver.py         # MCP 工具服务器
├── data/                # 知识库文档（Markdown）
├── vectordb/            # ChromaDB 向量数据库（自动生成）
├── .env                 # 环境变量配置
└── requirements.txt     # 依赖清单
```

## 🚀 快速开始

### 1. 环境准备

```bash
# Python 3.10+
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 安装 Ollama 并拉取模型

```bash
# 安装 Ollama（参考 https://ollama.com）
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
```

### 3. 准备知识库

将 Markdown 格式的知识文档放入 `data/` 目录：

```bash
mkdir -p data
# 放入你的 .md 文件
```

### 4. 构建向量数据库

首次运行需构建向量库。在 `rag1.py` 中设置：

```python
FORCE_REBUILD = True
```

运行一次后改回 `False`。

### 5. 启动

```bash
python main.py
```

启动后即可输入问题：

```
🚀 Agent + RAG 启动
==================================================

✅ 已就绪，输入 exit 退出

❓ 问题: Jett 的技能是什么？
⚡ 直接走 RAG（强制）
🔥 [RAG TOOL CALLED]: Jett 的技能是什么？
...
```

## 💡 使用示例

```
❓ 问题: Jett 有什么技能？
→ 自动走 RAG，从知识库检索英雄技能信息

❓ 问题: 今天广州天气怎么样？
→ Agent 调用 MCP weather 工具，返回实时天气

❓ 问题: 帮我搜一下最新的 AI 新闻
→ Agent 调用 MCP web_search 工具，返回搜索结果
```

## ⚙️ 配置说明

| 参数 | 位置 | 说明 |
|------|------|------|
| `MODEL_NAME` | main.py / rag1.py | Ollama 模型名称 |
| `EMBEDDING_MODEL` | rag1.py | Embedding 模型名称 |
| `DATA_PATH` | rag1.py | 知识库文档目录 |
| `DB_PATH` | rag1.py | 向量数据库存储路径 |
| `FORCE_REBUILD` | rag1.py | 是否强制重建向量库 |
| `chunk_size` | rag1.py | 文本分块大小 |
| `chunk_overlap` | rag1.py | 分块重叠长度 |

## 🔮 后续规划

- [ ] 支持更多文档格式（PDF、Word）
- [ ] Query Rewrite 改用 LLM 动态生成
- [ ] 接入更多 MCP 工具（日历、邮件等）
- [ ] 添加 Streamlit / Gradio Web UI
- [ ] 支持多用户会话管理
- [ ] Docker 一键部署

## 📄 License

MIT

---

> 💻 独立开发 | 持续迭代中
