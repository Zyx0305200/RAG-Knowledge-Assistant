
import os
import sys
from pathlib import Path

import streamlit as st

# 保证从不同工作目录启动时，也能找到项目模块
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from rag import create_vector_store
from llm import generate_answer


DATA_DIRECTORY = "data"
SCORE_THRESHOLD = 1.0
TOP_K = 3


st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Knowledge Assistant")
st.caption("基于 BGE + Chroma + Qwen 的本地知识库问答系统")


@st.cache_resource
def load_vector_store():
    return create_vector_store(DATA_DIRECTORY)


try:
    vector_store = load_vector_store()
except Exception as error:
    st.error(f"知识库加载失败：{error}")
    st.stop()


# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []


# 侧边栏
with st.sidebar:
    st.header("系统设置")

    st.write("Embedding：BAAI/bge-small-zh-v1.5")
    st.write("Vector DB：Chroma")
    st.write("LLM：Qwen Plus")

    st.divider()

    st.write(f"Top-K：{TOP_K}")
    st.write(f"Score Threshold：{SCORE_THRESHOLD}")

    if st.button("清空聊天记录"):
        st.session_state.messages = []
        st.rerun()

    st.info(
        "回答基于 data/ 中的知识库文件。"
        "当检索不到足够相关的资料时，系统会拒答。"
    )


# 显示历史聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("查看引用来源"):
                for source in message["sources"]:
                    st.write(source)


# 接收用户输入
question = st.chat_input("请输入你的问题……")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("正在检索知识库……"):
            try:
                search_results = (
                    vector_store.similarity_search_with_score(
                        question,
                        k=TOP_K
                    )
                )

                filtered_results = [
                    (document, score)
                    for document, score in search_results
                    if score < SCORE_THRESHOLD
                ]

                if not filtered_results:
                    answer = "根据现有资料无法回答。"
                    sources = []

                else:
                    context = "\n\n".join(
                        document.page_content
                        for document, score in filtered_results
                    )

                    prompt = f"""
你是一个知识库问答助手。

请严格根据以下参考资料回答问题。
不要编造参考资料中没有的信息。
如果参考资料无法回答问题，
请回答“根据现有资料无法回答”。

参考资料：
{context}

用户问题：
{question}
"""

                    with st.spinner("正在生成回答……"):
                        answer = generate_answer(prompt)

                    sources = []

                    for document, score in filtered_results:
                        source_path = document.metadata.get(
                            "source", "未知文件"
                        )
                        filename = os.path.basename(source_path)

                        page = document.metadata.get("page")

                        if isinstance(page, int):
                            location = f"{filename} · 第 {page + 1} 页"
                        else:
                            location = filename

                        sources.append(
                            f"{location} · 距离分数：{score:.4f}"
                        )

            except Exception as error:
                answer = f"处理问题时发生错误：{error}"
                sources = []

        st.markdown(answer)

        if sources:
            with st.expander("查看引用来源"):
                for source in sources:
                    st.write(source)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
