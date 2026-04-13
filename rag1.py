from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

import os
import shutil

# ==========================================
# 配置
# ==========================================

DATA_PATH = "./data"
DB_PATH = "./vectordb"

MODEL_NAME = "qwen3.5:9b"
EMBEDDING_MODEL = "nomic-embed-text"

FORCE_REBUILD = False  # 第一次必须 True

# ==========================================
# 构建向量数据库（最终稳定版🔥）
# ==========================================

def build_vector_db():
    print("🔨 正在构建向量数据库...")

    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        recursive=True
    )

    documents = loader.load()
    print("📄 文档数:", len(documents))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    splits = []

    for doc in documents:
        chunks = splitter.create_documents([doc.page_content])

        for chunk in chunks:
            text = chunk.page_content

            # ❗核心：过滤脏数据（解决 NaN）
            if not text:
                continue

            text = text.strip()

            if text == "" or text.lower() == "nan":
                continue

            if len(text) < 20:
                continue

            chunk.metadata.update(doc.metadata)

            # ✅ 增强语义（中英都能命中）
            chunk.page_content = f"""
Valorant Agent Knowledge

{text}
"""

            splits.append(chunk)

    print("✂️ 分块数:", len(splits))

    # 再过滤一层（保险）
    safe_splits = []
    for s in splits:
        try:
            if s.page_content and len(s.page_content.strip()) > 10:
                safe_splits.append(s)
        except:
            continue

    print("✅ 有效 chunk 数:", len(safe_splits))

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vectordb = Chroma.from_documents(
        documents=safe_splits,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print("✅ 向量数据库构建完成")

    return vectordb


def load_vector_db():
    if FORCE_REBUILD and os.path.exists(DB_PATH):
        print("⚠️ 删除旧数据库...")
        shutil.rmtree(DB_PATH)

    if not os.path.exists(DB_PATH):
        return build_vector_db()

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    print("📦 已加载数据库")
    print("📊 向量数量:", db._collection.count())

    return db

# ==========================================
# Query Rewrite（解决中文查询🔥）
# ==========================================

def rewrite_query(q: str):
    q_lower = q.lower()

    if "jett" in q_lower:
        if "技能" in q_lower:
            return "Jett abilities skills"
        if "特点" in q_lower:
            return "Jett strengths weaknesses abilities"

    if "iso" in q_lower:
        return "Iso abilities characteristics"

    return q

# ==========================================
# RAG Chain
# ==========================================

def create_rag_chain():
    vectordb = load_vector_db()

    prompt = PromptTemplate.from_template("""
你是一个【Valorant知识库查询工具】。

规则：
- 只能使用提供内容
- 没有答案输出：NOT_FOUND
- 回答简洁

内容：
{context}

问题：
{question}

答案：
""")

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0
    )

    def retrieve_and_format(query):
        query = rewrite_query(query)

        print("\n🔍 查询:", query)

        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 6,
                "fetch_k": 20,
                "lambda_mult": 0.6,
            }
        )

        docs = retriever.invoke(query)

        print("📄 命中文档:", len(docs))

        for i, d in enumerate(docs):
            print(f"\n--- 文档{i+1} ---")
            print(d.page_content[:200])

        if not docs:
            return "NO_CONTEXT"

        return "\n\n".join(d.page_content for d in docs)

    rag_chain = (
        {
            "context": RunnableLambda(retrieve_and_format),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain