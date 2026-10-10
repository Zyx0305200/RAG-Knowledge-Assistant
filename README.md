
# RAG Knowledge Assistant

基于 RAG（Retrieval-Augmented Generation，检索增强生成）的本地知识库智能问答系统。

项目支持 PDF、TXT 多文档知识库，使用 BGE 模型生成文本向量，通过 Chroma 进行语义检索，并调用 Qwen 大语言模型生成回答。

除了基础问答功能，项目还实现了知识库自动更新、检索阈值过滤、Streamlit 网页交互、人工相关性标注以及检索效果评测。

## 1. Features

### 知识库管理

- 支持 PDF / TXT 文档加载
- 支持多文档知识库
- 自动文档切块（Chunking）
- 使用 SHA-256 检测知识文件变化
- 检测 Embedding 模型及切块参数变化
- 知识库变化后自动重建 Chroma 数据库
- 数据库持久化，避免重复构建

### RAG 问答

- 使用 BGE 中文 Embedding 模型
- 基于 Chroma 的 Top-K 语义检索
- 使用 Score Threshold 过滤低相关性结果
- 接入 Qwen 大语言模型
- 根据检索资料生成回答
- 知识库资料不足时拒答
- 展示检索资料来源及距离分数

### Web 界面

- 基于 Streamlit 的聊天界面
- 支持连续输入问题
- 显示当前会话的聊天记录
- 展示回答引用的知识库文件
- 支持清空聊天记录

### 检索评测

- 人工标注 Chunk 相关性
- 支持 0 / 1 / 2 三级相关性标签
- 计算 Precision@K、Recall@K、nDCG@K
- 支持 Score Threshold 消融实验
- 评估知识库外问题的拒答效果
- 保存评测结果为 JSON

## 2. Tech Stack

| 模块 | 技术 |
|---|---|
| 开发语言 | Python |
| RAG 框架 | LangChain |
| Embedding | BAAI/bge-small-zh-v1.5 |
| 向量数据库 | Chroma |
| 大语言模型 | Qwen Plus |
| LLM API | OpenAI-compatible API |
| 文档解析 | PyPDFLoader、TextLoader |
| 文档切块 | RecursiveCharacterTextSplitter |
| Web 界面 | Streamlit |
| 评测与可视化 | Python、Pandas、Streamlit |

## 3. RAG Architecture

### 知识库构建流程

```text
PDF / TXT Documents
        |
        v
Document Loader
        |
        v
Text Splitter
        |
        v
Text Chunks
        |
        v
BGE Embedding Model
        |
        v
Chroma Vector Database
```

### 问答流程

```text
User Question
      |
      v
Query Embedding
      |
      v
Chroma Similarity Search
      |
      v
Top-K Chunks
      |
      v
Score Threshold Filtering
      |
      +---- No relevant chunks
      |           |
      |           v
      |       Refuse to answer
      |
      v
Build Context and Prompt
      |
      v
Qwen LLM
      |
      v
Generated Answer
```

系统首先从知识库中检索与用户问题相关的文本片段，再根据距离阈值筛选结果。

如果没有检索结果通过阈值，系统直接拒答；否则将保留的资料作为上下文交给 Qwen 生成回答。

## 4. Project Structure

```text
RAG-Knowledge-Assistant/
|
|-- data/
|   |-- knowledge.txt
|
|-- src/
|   |-- main.py
|   |-- rag.py
|   |-- llm.py
|   |-- web_app.py
|   |-- inspect_chunks.py
|   |-- annotation_app.py
|   |-- evaluate_retrieval.py
|
|-- .gitignore
|-- README.md
|-- requirements.txt
```

主要文件说明：

- `src/main.py`：命令行 RAG 问答入口
- `src/rag.py`：文档加载、切块、向量化及数据库更新
- `src/llm.py`：Qwen API 调用
- `src/web_app.py`：Streamlit 网页问答界面
- `src/inspect_chunks.py`：查看知识库 Chunk
- `src/annotation_app.py`：人工相关性标注工具
- `src/evaluate_retrieval.py`：检索评测与消融实验
- `data/`：本地知识库文档目录

本地运行时还会生成：

- `chroma_db/`：Chroma 向量数据库
- `knowledge_base_state.json`：知识库状态记录
- `evaluation_annotations.json`：人工标注数据
- `evaluation_results.json`：评测结果

这些本地数据文件不会提交到公开仓库。

## 5. Installation

### 5.1 Clone Repository

```bash
git clone https://github.com/Zyx0305200/RAG-Knowledge-Assistant.git
cd RAG-Knowledge-Assistant
```

### 5.2 Create Virtual Environment

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 5.3 Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

首次使用 Embedding 模型时，需要从 Hugging Face 下载模型文件。

### 5.4 Configure API Key

在项目根目录创建 `.env` 文件：

```env
DASHSCOPE_API_KEY=your_api_key_here
```

将 `your_api_key_here` 替换为自己的 DashScope API Key。

请勿将真实 API Key 上传到 GitHub。

## 6. Usage

### 6.1 Prepare Knowledge Base

将 PDF 或 TXT 文件放入 `data/` 目录。

项目提供示例文件：

```text
data/knowledge.txt
```

用户也可以自行添加知识库文件。

### 6.2 Command Line

运行：

```powershell
python src/main.py
```

程序会加载或构建向量数据库，然后进入问答流程。

### 6.3 Web Chat

运行：

```powershell
python -m streamlit run src/web_app.py
```

在浏览器中打开 Streamlit 提供的本地地址，通常为：

```text
http://localhost:8501
```

用户可以通过聊天界面提问、查看回答及检索来源。

### 6.4 Inspect Chunks

运行：

```powershell
python src/inspect_chunks.py
```

用于检查知识库中的文本片段。

### 6.5 Human Annotation

运行：

```powershell
python -m streamlit run src/annotation_app.py
```

人工检查检索语料并标注相关性。

相关性标签：

- 0：不相关
- 1：部分相关
- 2：高度相关

### 6.6 Retrieval Evaluation

完成标注后运行：

```powershell
python src/evaluate_retrieval.py
```

程序计算检索指标，并将结果保存到：

```text
evaluation_results.json
```

## 7. Knowledge Base Update

系统通过 SHA-256 文件哈希和配置状态检测知识库变化。

以下情况会触发数据库重建：

- 新增知识文件
- 删除知识文件
- 修改知识文件
- 修改 Embedding 模型
- 修改 Chunk Size
- 修改 Chunk Overlap

当前切块参数：

```python
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
```

当文件与配置没有变化时，程序直接加载已有的 Chroma 数据库。

注意：修改知识库或切块配置后，原有 Chunk 可能发生变化，需要重新检查人工标注与评测数据。

## 8. Retrieval Evaluation

项目构建了一个小规模人工评测集，用于验证检索效果。

评测集包含：

- 8 道知识库内问题
- 2 道知识库外问题

人工标注采用三级相关性标签。

主要评测指标：

**Precision@3**

衡量 Top-3 检索结果中相关 Chunk 的比例。

**Recall@3**

衡量知识库中已标注的相关 Chunk 有多少被成功检索。

**nDCG@3**

衡量检索排序质量，同时考虑相关性等级与排名位置。

**Retained Purity**

衡量阈值过滤后保留结果的相关比例。

**Rejection Rate**

衡量知识库外问题被系统拒答的比例。

## 9. Ablation Study

为了验证 Score Threshold 的实际作用，项目设计了对照实验。

- Baseline：保留原始 Top-3 检索结果
- Threshold：只保留距离分数小于 1.0 的 Chunk

其他检索条件保持一致。

### Experimental Results

| Metric | Baseline | Threshold |
|---|---:|---:|
| Precision@3 | 0.500 | 0.500 |
| Retained Purity | 0.500 | 0.812 |
| Recall@3 | 0.927 | 0.927 |
| nDCG@3 | 0.981 | 0.981 |
| Average Retained Chunks | 3.000 | 2.000 |
| Out-of-scope Rejection Rate | 0.000 | 1.000 |

### Analysis

在当前评测集上：

1. 阈值过滤将平均保留结果纯度从 50.0% 提高到 81.2%。
2. 平均保留 Chunk 数从 3 减少到 2。
3. Recall@3 保持不变，没有观察到相关 Chunk 召回损失。
4. 两道知识库外问题均被阈值机制拒答。

实验说明，在当前测试数据上，Score Threshold 能够减少部分无关检索结果。

需要注意：

- 评测集只有 10 道问题，规模较小。
- 拒答率仅基于 2 道知识库外问题。
- 结果不能直接推广到其他知识库。
- 当前实验评估的是检索阶段，并非最终生成答案的准确率。
- 阈值 1.0 是当前实验配置，不代表通用最优阈值。

## 10. Limitations

当前项目仍存在一些限制：

- 仅支持 PDF 和 TXT 文档。
- 目前主要使用向量相似度检索。
- 尚未实现 BM25 与向量检索的混合检索。
- 尚未实现 Reranker 重排序。
- 尚未实现基于对话历史的上下文检索。
- 当前网页能够保存会话消息，但每次提问仍独立检索。
- 评测集规模有限。
- 尚未提供 FastAPI 服务接口。

## 11. Future Work

后续可以继续优化：

- 增加 BM25 + Dense Retrieval 混合检索
- 引入 Reranker 提升检索排序质量
- 扩充人工标注评测集
- 比较不同 Chunk Size 和 Chunk Overlap
- 优化知识库外问题拒答策略
- 实现真正的多轮上下文问答
- 增加 FastAPI 后端接口
- 增加自动化测试与持续集成

## 12. Security

- API Key 通过 `.env` 管理。
- `.env` 不提交到 GitHub。
- 本地 PDF 文件默认不提交。
- Chroma 数据库不提交。
- 人工标注和评测结果默认保存在本地。
- 上传公开知识库前应检查版权及个人隐私信息。

## 13. License

This project is for learning and educational purposes.

目前未配置正式开源许可证。使用或分发代码时，应遵守适用的版权与许可要求。
