
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

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

COLLECTION_NAME = "rag_knowledge_base"
PERSIST_DIRECTORY = "chroma_db"
STATE_FILE = "knowledge_base_state.json"


def calculate_file_hash(file_path):
    with open(file_path, "rb") as file:
        file_content = file.read()

    return hashlib.sha256(file_content).hexdigest()


def get_data_hashes(data_directory):
    file_hashes = {}

    for filename in sorted(os.listdir(data_directory)):
        file_path = os.path.join(data_directory, filename)

        if (
            os.path.isfile(file_path)
            and filename.lower().endswith((".pdf", ".txt"))
        ):
            file_hashes[filename] = calculate_file_hash(file_path)

    return file_hashes


def create_vector_store(data_directory):

    # =========================
    # 获取当前知识库状态
    # =========================

    current_state = {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "file_hashes": get_data_hashes(data_directory)
    }

    # =========================
    # 读取上一次知识库状态
    # =========================

    if os.path.exists(STATE_FILE):
        with open(
            STATE_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            old_state = json.load(file)
    else:
        old_state = {}

    # =========================
    # 判断是否可以直接加载数据库
    # =========================

    database_exists = (
        os.path.isdir(PERSIST_DIRECTORY)
        and os.path.exists(
            os.path.join(PERSIST_DIRECTORY, "chroma.sqlite3")
        )
    )

    # 兼容旧版状态文件：
    # 旧版没有保存切块参数，但实际使用的是 500 / 100。
    # 当配置仍为旧版默认值时，不必因此重建数据库。

    if (
        "chunk_size" not in old_state
        and "chunk_overlap" not in old_state
        and old_state.get("embedding_model") == EMBEDDING_MODEL
        and old_state.get("file_hashes") == current_state["file_hashes"]
        and CHUNK_SIZE == 500
        and CHUNK_OVERLAP == 100
    ):
        old_state = current_state.copy()

        with open(
            STATE_FILE,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                old_state,
                file,
                ensure_ascii=False,
                indent=4
            )

        print("已升级旧版知识库状态文件")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    if current_state == old_state and database_exists:
        print(
            "知识库状态没有变化，"
            "直接加载已有 Chroma 数据库"
        )

        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )

    # =========================
    # 知识库配置或文件发生变化
    # =========================

    print(
        "检测到知识库文件、Embedding 模型"
        "或切块参数发生变化，需要重新构建数据库"
    )

    # =========================
    # 加载知识文件
    # =========================

    documents = []

    for filename in sorted(os.listdir(data_directory)):
        file_path = os.path.join(
            data_directory,
            filename
        )

        if not os.path.isfile(file_path):
            continue

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
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = text_splitter.split_documents(documents)

    print(
        "文档切块完成，共生成",
        len(chunks),
        "个 Chunk"
    )

    # =========================
    # 重建 Chroma 数据库
    # =========================

    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
        print("旧的 Chroma 数据库已删除")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    vector_store.add_documents(chunks)

    print("新的 Chroma 向量数据库构建完成")

    # =========================
    # 保存当前知识库状态
    # =========================

    with open(
        STATE_FILE,
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
