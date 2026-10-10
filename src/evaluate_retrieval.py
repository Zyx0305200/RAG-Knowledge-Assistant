
import os
import json
import math
import hashlib
from collections import Counter

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import create_vector_store


# ==============================
# 配置
# ==============================

DATA_DIRECTORY = "data"
ANNOTATION_FILE = "evaluation_annotations.json"
REVIEW_FILE = "evaluation_review_status.json"
SNAPSHOT_FILE = "evaluation_corpus_snapshot.json"
RESULT_FILE = "evaluation_results.json"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

TOP_K = 3
SCORE_THRESHOLD = 1.0

QUESTIONS = [
    "什么是差序格局？",
    "《史记》最突出的成就是什么？",
    "《红楼梦》的核心是什么？",
    "为什么说数学的核心是证明而不是计算？",
    "什么是黑洞？",
    "Transformer和RNN有什么区别？",
    "RAG是什么？",
    "实验室安全承诺书要求学习哪些安全知识？",
    "秦始皇是哪一年统一六国的？",
    "Python的装饰器是什么？",
]


# ==============================
# 文件读写
# ==============================

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4
        )


# ==============================
# Chunk 指纹
# ==============================

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def chunk_fingerprint(document):
    source = os.path.normpath(
        document.metadata.get("source", "")
    ).replace("\\", "/")

    identity = json.dumps(
        {
            "source": source,
            "page": document.metadata.get("page"),
            "content": document.page_content,
        },
        ensure_ascii=False,
        sort_keys=True
    )

    return sha256_bytes(identity.encode("utf-8"))


def get_file_hashes():
    hashes = {}

    for filename in sorted(os.listdir(DATA_DIRECTORY)):
        path = os.path.join(DATA_DIRECTORY, filename)

        if not os.path.isfile(path):
            continue

        if not filename.lower().endswith((".pdf", ".txt")):
            continue

        with open(path, "rb") as f:
            hashes[filename] = sha256_bytes(f.read())

    return hashes


def load_chunks():
    documents = []

    for filename in sorted(os.listdir(DATA_DIRECTORY)):
        path = os.path.join(DATA_DIRECTORY, filename)

        if not os.path.isfile(path):
            continue

        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)

        elif filename.lower().endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")

        else:
            continue

        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return splitter.split_documents(documents)


# ==============================
# 语料快照验证
# ==============================

def build_snapshot(chunks):
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "file_hashes": get_file_hashes(),
        "chunk_fingerprints": [
            chunk_fingerprint(chunk)
            for chunk in chunks
        ]
    }


def verify_snapshot(snapshot):
    if not os.path.exists(SNAPSHOT_FILE):
        raise RuntimeError(
            "找不到已有语料快照。"
            "请不要自动创建新快照，以免旧标签错位。"
        )

    saved_snapshot = read_json(SNAPSHOT_FILE)

    if saved_snapshot != snapshot:
        raise RuntimeError(
            "语料快照不一致！"
            "请检查文件、切块参数和标注数据。"
        )

    print("语料快照验证通过")


# ==============================
# 计算检索指标
# ==============================

def calculate_metrics(retrieved_ids, labels):
    relevant_ids = {
        chunk_id
        for chunk_id, grade in labels.items()
        if grade >= 1
    }

    relevant_count = sum(
        1
        for chunk_id in retrieved_ids
        if labels.get(chunk_id, 0) >= 1
    )

    precision_at_k = relevant_count / TOP_K

    retained_purity = (
        relevant_count / len(retrieved_ids)
        if retrieved_ids else None
    )

    recall_at_k = (
        relevant_count / len(relevant_ids)
        if relevant_ids else None
    )

    dcg = sum(
        (2 ** labels.get(chunk_id, 0) - 1)
        / math.log2(rank + 2)
        for rank, chunk_id in enumerate(retrieved_ids)
    )

    ideal_grades = sorted(
        labels.values(),
        reverse=True
    )[:TOP_K]

    idcg = sum(
        (2 ** grade - 1)
        / math.log2(rank + 2)
        for rank, grade in enumerate(ideal_grades)
    )

    ndcg_at_k = (
        dcg / idcg
        if idcg > 0 else None
    )

    return {
        "precision_at_3": precision_at_k,
        "retained_purity": retained_purity,
        "recall_at_3": recall_at_k,
        "ndcg_at_3": ndcg_at_k,
        "retained_count": len(retrieved_ids)
    }


def average_metric(results, key):
    values = [
        result[key]
        for result in results
        if result[key] is not None
    ]

    return (
        sum(values) / len(values)
        if values else None
    )


def display_number(value):
    if value is None:
        return "N/A"

    return f"{value:.3f}"


# ==============================
# 主程序
# ==============================

def main():
    annotations = read_json(ANNOTATION_FILE)
    review_status = read_json(REVIEW_FILE)

    reviewed = set(
        review_status["reviewed_questions"]
    )

    expected_questions = {
        f"Q{i + 1}"
        for i in range(len(QUESTIONS))
    }

    if reviewed != expected_questions:
        raise ValueError(
            "评测问题尚未全部审核。"
        )

    chunks = load_chunks()

    print(f"当前语料：{len(chunks)} 个 Chunk")

    snapshot = build_snapshot(chunks)
    verify_snapshot(snapshot)

    fingerprints = snapshot["chunk_fingerprints"]

    counts = Counter(fingerprints)

    if any(count > 1 for count in counts.values()):
        raise ValueError(
            "发现重复 Chunk 指纹，无法唯一匹配。"
        )

    fingerprint_to_id = {
        fingerprint: str(index)
        for index, fingerprint in enumerate(fingerprints)
    }

    valid_chunk_ids = {
        str(index)
        for index in range(len(chunks))
    }

    for question_id, labels in annotations.items():
        if question_id not in expected_questions:
            raise ValueError(
                f"未知问题：{question_id}"
            )

        for chunk_id, grade in labels.items():
            if chunk_id not in valid_chunk_ids:
                raise ValueError(
                    f"无效 Chunk ID：{chunk_id}"
                )

            if grade not in (1, 2):
                raise ValueError(
                    "相关性标签必须为 1 或 2"
                )

    vector_store = create_vector_store(DATA_DIRECTORY)

    baseline_metrics = []
    threshold_metrics = []

    baseline_out_of_scope = []
    threshold_out_of_scope = []

    question_results = []

    for index, question in enumerate(QUESTIONS):
        question_id = f"Q{index + 1}"
        labels = annotations.get(question_id, {})

        search_results = (
            vector_store.similarity_search_with_score(
                question,
                k=TOP_K
            )
        )

        raw_ids = []
        filtered_ids = []
        retrieval_details = []

        for document, score in search_results:
            fingerprint = chunk_fingerprint(document)

            if fingerprint not in fingerprint_to_id:
                raise RuntimeError(
                    f"{question_id} 检索到快照外 Chunk。"
                )

            chunk_id = fingerprint_to_id[fingerprint]

            raw_ids.append(chunk_id)

            passed = score < SCORE_THRESHOLD

            if passed:
                filtered_ids.append(chunk_id)

            retrieval_details.append({
                "chunk_id": chunk_id,
                "score": float(score),
                "grade": labels.get(chunk_id, 0),
                "passed_threshold": passed
            })

        print(f"\n{question_id}：{question}")
        print("原始检索：", raw_ids)
        print("阈值过滤：", filtered_ids)

        if labels:
            baseline = calculate_metrics(
                raw_ids,
                labels
            )

            threshold = calculate_metrics(
                filtered_ids,
                labels
            )

            baseline_metrics.append(baseline)
            threshold_metrics.append(threshold)

            print(
                "Baseline：",
                "P@3 =", display_number(
                    baseline["precision_at_3"]
                ),
                "Recall@3 =", display_number(
                    baseline["recall_at_3"]
                ),
                "nDCG@3 =", display_number(
                    baseline["ndcg_at_3"]
                )
            )

            print(
                "Threshold：",
                "P@3 =", display_number(
                    threshold["precision_at_3"]
                ),
                "Recall@3 =", display_number(
                    threshold["recall_at_3"]
                ),
                "nDCG@3 =", display_number(
                    threshold["ndcg_at_3"]
                )
            )

            record = {
                "question_id": question_id,
                "question": question,
                "type": "in_scope",
                "baseline": baseline,
                "threshold": threshold,
                "retrieval": retrieval_details
            }

        else:
            # 不过滤时，只要返回 Chunk，就视为接受
            baseline_rejected = len(raw_ids) == 0

            threshold_rejected = (
                len(filtered_ids) == 0
            )

            baseline_out_of_scope.append(
                baseline_rejected
            )

            threshold_out_of_scope.append(
                threshold_rejected
            )

            print(
                "Baseline 拒答：",
                baseline_rejected
            )
            print(
                "Threshold 拒答：",
                threshold_rejected
            )

            record = {
                "question_id": question_id,
                "question": question,
                "type": "out_of_scope",
                "baseline_rejected": baseline_rejected,
                "threshold_rejected": threshold_rejected,
                "retrieval": retrieval_details
            }

        question_results.append(record)

    metric_keys = [
        "precision_at_3",
        "retained_purity",
        "recall_at_3",
        "ndcg_at_3",
        "retained_count"
    ]

    baseline_summary = {
        key: average_metric(
            baseline_metrics,
            key
        )
        for key in metric_keys
    }

    threshold_summary = {
        key: average_metric(
            threshold_metrics,
            key
        )
        for key in metric_keys
    }

    baseline_summary["rejection_rate"] = (
        sum(baseline_out_of_scope)
        / len(baseline_out_of_scope)
    )

    threshold_summary["rejection_rate"] = (
        sum(threshold_out_of_scope)
        / len(threshold_out_of_scope)
    )

    print("\n========== 消融实验结果 ==========")

    print(
        f"{'指标':<24}"
        f"{'Baseline':>12}"
        f"{'Threshold':>12}"
    )

    for key in metric_keys + ["rejection_rate"]:
        print(
            f"{key:<24}"
            f"{display_number(baseline_summary[key]):>12}"
            f"{display_number(threshold_summary[key]):>12}"
        )

    write_json(
        RESULT_FILE,
        {
            "experiment": "score_threshold_ablation",
            "top_k": TOP_K,
            "score_threshold": SCORE_THRESHOLD,
            "baseline": baseline_summary,
            "threshold": threshold_summary,
            "questions": question_results
        }
    )

    print(
        f"\n结果已保存到 {RESULT_FILE}"
    )


if __name__ == "__main__":
    main()
