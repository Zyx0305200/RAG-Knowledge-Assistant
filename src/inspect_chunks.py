from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

import os


# =========================
# 项目配置
# =========================

DATA_DIRECTORY = "data"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# =========================
# 加载知识库文件
# =========================

documents = []

for filename in os.listdir(DATA_DIRECTORY):

    file_path = os.path.join(
        DATA_DIRECTORY,
        filename
    )

    if filename.lower().endswith(".pdf"):

        loader = PyPDFLoader(file_path)
        file_documents = loader.load()

        documents.extend(file_documents)

    elif filename.lower().endswith(".txt"):

        loader = TextLoader(
            file_path,
            encoding="utf-8"
        )

        file_documents = loader.load()

        documents.extend(file_documents)


# =========================
# 文档切块
# =========================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

chunks = text_splitter.split_documents(
    documents
)

print(f"\n知识库共生成 {len(chunks)} 个 Chunk")


# =========================
# 搜索 Chunk
# =========================

while True:

    keyword = input(
        "\n请输入要查找的内容（输入 exit 退出）："
    )

    if keyword.lower() == "exit":
        print("程序已退出")
        break

    matched_chunks = []

    for index, chunk in enumerate(chunks):

        if keyword.lower() in chunk.page_content.lower():

            matched_chunks.append(
                (index, chunk)
            )

    # =========================
    # 显示搜索结果
    # =========================

    print(
        f"\n找到 {len(matched_chunks)} 个包含"
        f"“{keyword}”的 Chunk"
    )

    for index, chunk in matched_chunks:

        source = chunk.metadata.get(
            "source",
            "未知"
        )

        page = chunk.metadata.get(
            "page"
        )

        print("\n" + "=" * 60)

        print(f"Chunk ID：{index}")
        print(f"Source：{source}")

        if isinstance(page, int):
            print(f"Page：{page + 1}")
        else:
            print("Page：无")

        print(
            f"字符数：{len(chunk.page_content)}"
        )

        print("\nContent：")
        print(chunk.page_content)