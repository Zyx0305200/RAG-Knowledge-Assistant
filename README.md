# RAG Knowledge Assistant

一个基于 **RAG（Retrieval-Augmented Generation，检索增强生成）** 的本地知识库问答系统。

项目支持读取 PDF 和 TXT 文档，通过 Embedding 模型将文档转换为向量并存储到 Chroma 向量数据库中。当用户提出问题时，系统会检索相关文档片段，并将检索结果作为上下文交给 Qwen 大语言模型生成回答。

## Features

- 支持 PDF 文档加载
- 支持 TXT 文档加载
- 支持多文档知识库
- 文档自动切块（Chunking）
- HuggingFace Embedding 向量化
- Chroma 向量数据库持久化
- 基于文件 Hash 检测知识库变化
- 知识文件变化后自动重建向量数据库
- 基于语义相似度进行 Top-K 检索
- 接入 Qwen 大语言模型生成答案
- 支持连续多轮提问
- 显示检索结果来源信息

## RAG Workflow

项目的核心流程：

```text
PDF / TXT
    ↓
Document Loader
    ↓
Text Splitter
    ↓
Chunks
    ↓
Embedding Model
    ↓
Chroma Vector Database
    ↓
User Question
    ↓
Query Embedding
    ↓
Similarity Search
    ↓
Top-K Relevant Chunks
    ↓
Prompt
    ↓
Qwen LLM
    ↓
Answer
```

其中：

- **Document Loader**：负责读取 PDF / TXT 文件
- **Text Splitter**：将长文档切分成多个 Chunk
- **Embedding Model**：将文本转换为向量表示
- **Chroma**：存储文档向量并进行相似度检索
- **Qwen**：根据检索到的资料生成最终回答

## Project Structure

```text
RAG_Project/
│
├── data/
│   └── knowledge.txt
│
├── src/
│   ├── main.py
│   ├── rag.py
│   └── llm.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

主要模块：

- `src/main.py`：程序入口，负责组织完整 RAG 问答流程
- `src/rag.py`：负责文档加载、切块、Embedding、Chroma 和知识库更新
- `src/llm.py`：负责调用 Qwen API 生成回答
- `data/`：存放本地知识库文件

## Installation

### 1. Clone repository

```bash
git clone https://github.com/Zyx0305200/RAG-Knowledge-Assistant.git
cd RAG-Knowledge-Assistant
```

### 2. Create virtual environment

Windows：

```bash
python -m venv .venv
```

激活虚拟环境：

```bash
.\.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## API Configuration

项目使用 Qwen API。

在项目根目录创建：

```text
.env
```

然后添加：

```text
DASHSCOPE_API_KEY=your_api_key_here
```

请不要将真实 API Key 上传到 GitHub。

`.env` 已加入 `.gitignore`。

## Usage

将自己的 PDF 或 TXT 文件放入：

```text
data/
```

然后运行：

```bash
python src/main.py
```

首次运行时，程序会：

```text
读取知识文件
    ↓
文档切块
    ↓
生成 Embedding
    ↓
创建 Chroma 向量数据库
    ↓
保存文件 Hash
```

之后如果知识文件没有发生变化，程序会直接加载已有的 Chroma 数据库，避免重复构建。

## Example

```text
请输入你的问题（输入 exit 退出）：乡土中国的作者是谁？

回答：
《乡土中国》的作者是费孝通。
```

输入：

```text
exit
```

即可退出程序。

## Knowledge Base Update

系统会计算知识文件的 SHA-256 Hash。

当 PDF 或 TXT 文件发生新增、删除或修改时：

```text
文件发生变化
    ↓
Hash 发生变化
    ↓
删除旧 Chroma 数据库
    ↓
重新加载知识文件
    ↓
重新生成向量
    ↓
建立新的 Chroma 数据库
```

如果文件没有变化，则直接加载已有数据库。

## Tech Stack

- Python
- LangChain
- HuggingFace Sentence Transformers
- Chroma
- Qwen
- OpenAI-compatible API
- PyPDF

## Current Status

目前已经完成：

- [x] PDF 文档加载
- [x] TXT 文档加载
- [x] 多文档知识库
- [x] 文档 Chunking
- [x] Embedding
- [x] Chroma 持久化
- [x] 文件 Hash 变化检测
- [x] 自动重建知识库
- [x] 语义检索
- [x] Qwen API 接入
- [x] 连续问答
- [x] Git / GitHub 版本管理

后续计划：

- [ ] 优化检索质量
- [ ] 增加检索分数与阈值过滤
- [ ] 优化来源展示
- [ ] 增加更多文档格式
- [ ] 增加 Web 界面
- [ ] 增加 FastAPI 后端接口

## License

This project is for learning and educational purposes.