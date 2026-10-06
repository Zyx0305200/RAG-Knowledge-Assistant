from rag import create_vector_store
from llm import generate_answer


# =========================
# 1. 创建 / 加载知识库
# =========================
vector_store = create_vector_store("data")


# =========================
# 2. 连续问答
# =========================
while True:

    question = input(
        "\n请输入你的问题（输入 exit 退出）："
    )

    if question.lower() == "exit":
        print("程序已退出")
        break

    # =========================
    # 3. 检索相关文档
    # =========================
    results = vector_store.similarity_search_with_score(
        question,
        k=3
    )

    # =========================
    # 4. 显示检索结果和分数
    # =========================
    print("\n===== 检索结果 =====")

    for i, (document, score) in enumerate(
        results,
        start=1
    ):
        print(f"\n结果 {i}")
        print(f"Score：{score:.4f}")
        print(
            "Source：",
            document.metadata.get(
                "source",
                "未知来源"
            )
        )
        print(
            "Page：",
            document.metadata.get(
                "page",
                "未知"
            )
        )

        print(
            "Content：",
            document.page_content[:200]
        )

    # =========================
    # 5. 提取 Document
    # =========================
    documents = [
        document
        for document, score in results
    ]

    # =========================
    # 6. 构造 Context
    # =========================
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # =========================
    # 7. 构造 Prompt
    # =========================
    prompt = f"""
请根据下面提供的参考资料回答问题。

参考资料：
{context}

问题：
{question}

要求：
1. 只根据参考资料回答。
2. 如果参考资料中没有答案，请回答“根据现有资料无法回答”。
3. 回答简洁、准确。
"""

    # =========================
    # 8. 调用 Qwen
    # =========================
    answer = generate_answer(prompt)

    # =========================
    # 9. 输出最终答案
    # =========================
    print("\n===== 最终回答 =====")
    print(answer)

    # =========================
    # 10. 输出来源
    # =========================
    print("\n===== 参考来源 =====")

    for document in documents:

        source = document.metadata.get(
            "source",
            "未知来源"
        )

        page = document.metadata.get(
            "page",
            "未知"
        )

        if isinstance(page, int):
            page = page + 1

        print(
            f"- {source}，第 {page} 页"
        )