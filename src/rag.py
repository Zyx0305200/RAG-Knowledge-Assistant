from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import os
import hashlib
import json
import shutil


# =========================
# 项目配置
# =========================

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


def calculate_file_hash(file_path):
    with open(file_path, "rb") as file:
        file_content = file.read()

    return hashlib.sha256(file_content).hexdigest()


def get_data_hashes(data_directory):
    file_hashes = {}

    for filename in os.listdir(data_directory):
        file_path = os.path.join(data_directory, filename)

        if (
            os.path.isfile(file_path)
            and filename.lower().endswith((".pdf", ".txt"))
        ):
            file_hashes[filename] = calculate_file_hash(file_path)

    return file_hashes


def create_vector_store(data_directory):

    persist_directory = "chroma_db"

    # 原来的 file_hashes.json
    # 现在不仅保存 Hash，还保存 Embedding 模型
    state_file = "knowledge_base_state.json"

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # =========================
    # 获取当前知识库状态
    # =========================

    current_state = {
        "embedding_model": EMBEDDING_MODEL,
        "file_hashes": get_data_hashes(data_directory)
    }

    # =========================
    # 读取上一次知识库状态
    # =========================

    if os.path.exists(state_file):
        with open(
            state_file,
            "r",
            encoding="utf-8"
        ) as file:
            old_state = json.load(file)
    else:
        old_state = {}

    # =========================
    # 判断是否可以直接加载数据库
    # =========================

    if (
        current_state == old_state
        and os.path.exists(persist_directory)
    ):
        print(
            "知识库状态没有变化，"
            "直接加载已有 Chroma 数据库"
        )

        vector_store = Chroma(
            collection_name="rag_knowledge_base",
            embedding_function=embeddings,
            persist_directory=persist_directory
        )

        return vector_store

    # =========================
    # 知识库发生变化
    # =========================

    print(
        "检测到知识库文件或 Embedding 模型发生变化，"
        "需要重新构建向量数据库"
    )

    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print("旧的 Chroma 数据库已删除")

    # =========================
    # 加载知识文件
    # =========================

    documents = []

    for filename in os.listdir(data_directory):
        file_path = os.path.join(
            data_directory,
            filename
        )

        if filename.lower().endswith(".pdf"):
            print("正在加载 PDF：", file_path)

            loader = PyPDFLoader(file_path)
            file_documents = loader.load()

            documents.extend(file_documents)

        elif filename.lower().endswith(".txt"):
            print("正在加载 TXT：", file_path)

            loader = TextLoader(
                file_path,
                encoding="utf-8"
            )

            file_documents = loader.load()

            documents.extend(file_documents)

    print(
        "文件加载完成，共加载",
        len(documents),
        "个 Document"
    )

    if len(documents) == 0:
        raise ValueError(
            "data 文件夹中没有找到 PDF 或 TXT 文件"
        )

    # =========================
    # 文档切块
    # =========================

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(
        documents
    )

    print(
        "文档切块完成，共生成",
        len(chunks),
        "个 Chunk"
    )

    # =========================
    # 创建 Chroma 数据库
    # =========================

    vector_store = Chroma(
        collection_name="rag_knowledge_base",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    vector_store.add_documents(chunks)

    print("新的 Chroma 向量数据库构建完成")

    # =========================
    # 保存当前知识库状态
    # =========================

    with open(
        state_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            current_state,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("新的知识库状态已保存")

    return vector_store