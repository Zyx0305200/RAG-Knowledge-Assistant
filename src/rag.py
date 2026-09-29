from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


def create_vector_store(pdf_path):
    # 读取 PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 文档切块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    # Embedding Model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Vector Store
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(chunks)

    return vector_store