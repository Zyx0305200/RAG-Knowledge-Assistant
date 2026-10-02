from llm import generate_answer

from rag import create_vector_store

vector_store = create_vector_store("data")

while True:
    question = input("\n请输入你的问题（输入 exit 退出）：")

    if question.lower() == "exit": #lower 转小写
        break


    # 5. 用户输入问题
    # question = input("请输入你的问题：")


    # 6. 从 Vector Store 检索相关 Chunks
    results = vector_store.similarity_search(
        question,
        k=3
    )


    # 7. 构造 Context
    context = "\n\n".join(
        document.page_content for document in results
    )


    # 8. 构造 Prompt
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


    # 9. 调用 Qwen
    answer = generate_answer(prompt)


    # 10. 输出答案
    print("\n===== 回答 =====")
    print(answer)


    # 11. 输出资料来源
    print("\n===== 参考来源 =====")

    for document in results:
        print(
            f"{document.metadata['source']} "
            f"第 {document.metadata['page_label']} 页"
        )