from rag import create_vector_store
from llm import generate_answer


# =========================
# 项目配置
# =========================

SCORE_THRESHOLD = 1.0


# =========================
# 创建 / 加载向量数据库
# =========================

vector_store = create_vector_store("data")


# =========================
# 交互式问答
# =========================

while True:

    question = input(
        "\n请输入你的问题（输入 exit 退出）："
    )

    if question.lower() == "exit":
        print("程序已退出")
        break

    # =========================
    # 第一步：从 Chroma 检索 Top-3
    # =========================

    results = vector_store.similarity_search_with_score(
        question,
        k=3
    )

    print("\n===== 原始检索结果 =====")

    for i, (document, score) in enumerate(
        results,
        start=1
    ):

        source = document.metadata.get(
            "source",
            "未知"
        )

        page = document.metadata.get(
            "page",
            "未知"
        )

        print(f"\n结果 {i}")
        print(f"Score：{score:.4f}")
        print(f"Source：{source}")
        print(f"Page：{page}")
        print(
            "Content：",
            document.page_content[:200]
        )

    # =========================
    # 第二步：Score 阈值过滤
    # =========================

    filtered_results = [
        (document, score)
        for document, score in results
        if score < SCORE_THRESHOLD
    ]

    print(
        f"\n通过 Score < {SCORE_THRESHOLD} "
        f"过滤后剩余 {len(filtered_results)} 个结果"
    )

    # =========================
    # 第三步：没有相关资料时直接拒答
    # =========================

    if len(filtered_results) == 0:
        print("\n===== 最终回答 =====")
        print("根据现有资料无法回答。")
        continue

    # =========================
    # 第四步：提取过滤后的 Document
    # =========================

    documents = [
        document
        for document, score in filtered_results
    ]

    # =========================
    # 第五步：构造 Context
    # =========================

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # =========================
    # 第六步：构造 Prompt
    # =========================

    prompt = f"""
你是一个知识库问答助手。

请严格根据下面提供的参考资料回答问题。

如果参考资料中没有足够的信息回答问题，
请回答“根据现有资料无法回答”。

参考资料：
{context}

用户问题：
{question}
"""

    # =========================
    # 第七步：调用 Qwen
    # =========================

    answer = generate_answer(prompt)

    print("\n===== 最终回答 =====")
    print(answer)

    # =========================
    # 第八步：显示来源
    # =========================

    print("\n===== 参考来源 =====")

    for document in documents:

        source = document.metadata.get(
            "source",
            "未知"
        )

        page = document.metadata.get(
            "page"
        )

        if isinstance(page, int):
            print(
                f"- {source}，第 {page + 1} 页"
            )
        else:
            print(
                f"- {source}"
            )