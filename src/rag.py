from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import os
import hashlib
import json
import shutil


# =========================
# 计算单个文件的 Hash
# =========================
def calculate_file_hash(file_path):
    with open(file_path, "rb") as file:
        file_content = file.read()

    return hashlib.sha256(file_content).hexdigest()


# =========================
# 计算知识库文件的 Hash
# 目前只监控 PDF 和 TXT
# =========================
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


# =========================
# 创建 / 加载向量数据库
# =========================
def create_vector_store(data_directory):

    persist_directory = "chroma_db"
    hash_file = "file_hashes.json"

    # 创建 Embedding 模型
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # -------------------------
    # 1. 计算当前知识文件的 Hash
    # -------------------------
    current_hashes = get_data_hashes(data_directory)

    # -------------------------
    # 2. 读取上一次保存的 Hash
    # -------------------------
    if os.path.exists(hash_file):
        with open(hash_file, "r", encoding="utf-8") as file:
            old_hashes = json.load(file)
    else:
        old_hashes = {}

    # -------------------------
    # 3. 文件没有变化：
    #    直接加载已有 Chroma
    # -------------------------
    if (
        current_hashes == old_hashes
        and os.path.exists(persist_directory)
    ):
        print("知识库文件没有变化，直接加载已有 Chroma 数据库")

        vector_store = Chroma(
            collection_name="rag_knowledge_base",
            embedding_function=embeddings,
            persist_directory=persist_directory
        )

        return vector_store

    # -------------------------
    # 4. 文件发生变化：
    #    重新构建向量数据库
    # -------------------------
    print("检测到知识库发生变化，需要重新构建向量数据库")

    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print("旧的 Chroma 数据库已删除")

    # -------------------------
    # 5. 加载所有 PDF + TXT
    # -------------------------
    documents = []

    for filename in os.listdir(data_directory):
        file_path = os.path.join(data_directory, filename)

        # 加载 PDF
        if filename.lower().endswith(".pdf"):
            print("正在加载 PDF：", file_path)

            loader = PyPDFLoader(file_path)
            file_documents = loader.load()

            documents.extend(file_documents)

        # 加载 TXT
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

    # 没有找到支持的知识文件
    if len(documents) == 0:
        raise ValueError(
            "data 文件夹中没有找到 PDF 或 TXT 文件"
        )

    # -------------------------
    # 6. 文档切块
    # -------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    print(
        "文档切块完成，共生成",
        len(chunks),
        "个 Chunk"
    )

    # -------------------------
    # 7. 创建新的 Chroma
    # -------------------------
    vector_store = Chroma(
        collection_name="rag_knowledge_base",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    # -------------------------
    # 8. Chunk 写入 Chroma
    # -------------------------
    vector_store.add_documents(chunks)

    print("新的 Chroma 向量数据库构建完成")

    # -------------------------
    # 9. 构建成功后保存最新 Hash
    # -------------------------
    with open(
        hash_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            current_hashes,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("新的文件 Hash 已保存")

    return vector_store