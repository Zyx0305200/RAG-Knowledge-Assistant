# with open("data/knowledge.txt", "r", encoding="utf-8") as file:
#     text = file.read()

# chunks = text.split("\n\n")

# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i + 1}:")
#     print(chunk)
#     print("-----")

# from sentence_transformers import SentenceTransformer

# # 加载 Embedding 模型
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # 准备一句话
# text = "Transformer和RNN有什么区别？"

# # 把文字转换成向量
# embedding = model.encode(text)

# print("原始文本：")
# print(text)

# print("\nEmbedding：")
# print(embedding)

# print("\n向量维度：")
# print(len(embedding))


# #自己手动实现
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# # 1. 加载 Embedding 模型
# model = SentenceTransformer(
#     "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# )

# # 2. 读取知识库
# with open("data/knowledge.txt", "r", encoding="utf-8") as file:
#     text = file.read()

# # 3. 切成多个 Chunk
# chunks = text.split("\n\n")

# # 4. 把所有 Chunk 转换成向量
# chunk_embeddings = model.encode(chunks)

# # 5. 用户的问题
# question = "Transformer和RNN有什么区别？"

# # 6. 把问题也转换成向量
# question_embedding = model.encode([question])

# # 7. 计算问题与每个 Chunk 的余弦相似度
# similarities = cosine_similarity(
#     question_embedding,
#     chunk_embeddings
# )[0]

# # 8. 打印结果
# for i, similarity in enumerate(similarities):
#     print(f"Chunk {i + 1} 相似度: {similarity:.4f}")
#     print(chunks[i])
#     print("-----")

# # 9. 找到相似度最高的 Chunk
# top_k = 2

# top_indices = similarities.argsort()[::-1][:top_k]

# print("\nTop-2 检索结果：")

# for index in top_indices:
#     print(f"\n相似度：{similarities[index]:.4f}")
#     print(chunks[index])


from langchain_text_splitters import RecursiveCharacterTextSplitter

#使用Langchain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


# 1. 创建 Embedding 对象
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# 2. 读取知识库
with open("data/knowledge.txt", "r", encoding="utf-8") as file:
    text = file.read()


# 3. 切成 Chunk
# chunks = text.split("\n\n")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = text_splitter.split_text(text)


# 4. 创建 Vector Store
vector_store = InMemoryVectorStore(embeddings)


# 5. 把 Chunk 加入 Vector Store
# vector_store.add_texts(chunks)
metadatas = []

for i in range(len(chunks)):
    metadatas.append({
        "source": "knowledge.txt",
        "chunk_id": i + 1
    })

    
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}:")
    print(chunk)

vector_store.add_texts(
    texts=chunks,
    metadatas=metadatas
)


print("成功存入 Vector Store！")

question = "Transformer和RNN有什么区别？"

results = vector_store.similarity_search(
    question,
    k=2
)

print("\n===== 检索结果 =====")

for i, document in enumerate(results):
    print(f"\n结果 {i + 1}：")
    print("正文：", document.page_content)
    print("来源：", document.metadata)

# for i, document in enumerate(results):
#     print(f"\n结果 {i + 1}：")
#     print(document.page_content)
