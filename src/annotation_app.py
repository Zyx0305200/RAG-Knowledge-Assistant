
import os
import json
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_DIRECTORY = "data"
ANNOTATION_FILE = "evaluation_annotations.json"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

QUESTIONS = [
    "什么是差序格局？",
    "《史记》最突出的成就是什么？",
    "《红楼梦》的核心是什么？",
    "为什么说数学的核心是证明而不是计算？",
    "什么是黑洞？",
    "Transformer和RNN有什么区别？",
    "RAG是什么？",
    "实验室安全承诺书要求学习哪些安全知识？",
    "秦始皇是哪一年统一六国的？",
    "Python的装饰器是什么？",
]


@st.cache_data
def load_chunks():
    documents = []

    for filename in sorted(os.listdir(DATA_DIRECTORY)):
        file_path = os.path.join(DATA_DIRECTORY, filename)

        if not os.path.isfile(file_path):
            continue

        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.lower().endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            continue

        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return splitter.split_documents(documents)


def load_annotations():
    if os.path.exists(ANNOTATION_FILE):
        with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_annotations(annotations):
    with open(ANNOTATION_FILE, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=4)


st.set_page_config(
    page_title="RAG 检索评测标注",
    layout="wide"
)

st.title("RAG 检索评测 · 人工标注工具")

chunks = load_chunks()
annotations = load_annotations()

st.caption(f"当前知识库：{len(chunks)} 个 Chunk")

question_index = st.selectbox(
    "选择评测问题",
    range(len(QUESTIONS)),
    format_func=lambda i: f"Q{i + 1}：{QUESTIONS[i]}"
)

question = QUESTIONS[question_index]
question_id = f"Q{question_index + 1}"

keyword = st.text_input(
    "搜索 Chunk（关键词，可留空）",
    value=""
)

st.info(
    "相关性等级：0 = 无关；1 = 部分相关；"
    "2 = 高度相关、足以支撑回答"
)

current_labels = annotations.get(question_id, {})

matched = []

for index, chunk in enumerate(chunks):
    if keyword.strip() and keyword.lower() not in chunk.page_content.lower():
        continue
    matched.append((index, chunk))

st.write(f"当前显示 {len(matched)} 个 Chunk")

for index, chunk in matched:

    source = chunk.metadata.get("source", "未知")
    page = chunk.metadata.get("page")

    page_text = f"第 {page + 1} 页" if isinstance(page, int) else "无页码"

    with st.expander(
        f"Chunk {index} · {os.path.basename(source)} · {page_text}"
    ):
        st.write(chunk.page_content)

        key = str(index)

        grade = st.radio(
            "相关性",
            options=[0, 1, 2],
            index=int(current_labels.get(key, 0)),
            horizontal=True,
            key=f"{question_id}_chunk_{index}"
        )

        if grade > 0:
            current_labels[key] = grade
        else:
            current_labels.pop(key, None)

annotations[question_id] = current_labels

if st.button("保存当前问题的标注", type="primary"):
    save_annotations(annotations)
    st.success("标注已保存到 evaluation_annotations.json")
