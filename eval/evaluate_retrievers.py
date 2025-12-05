import sys
from pathlib import Path

# Добавляем корень проекта в sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import json
from pathlib import Path
from typing import List, Dict, Set

import numpy as np
from tqdm import tqdm

from src.rag_pipeline import ScienceRAGPipeline
from src.evaluation.metrics import recall_at_k, precision_at_k, load_validation_qa


# -------------------------------
# 1. Load pipeline once
# -------------------------------
pipeline = ScienceRAGPipeline()


# -------------------------------
# 2. Load validation set
# -------------------------------
VAL_PATH = Path("data/validation/qa.jsonl")

qa_data = load_validation_qa(VAL_PATH)

questions: List[str] = []
gold_sets: List[Set[str]] = []

for qid, record in qa_data.items():
    questions.append(record["question"])
    gold_sets.append(set(record["gold_chunk_ids"]))

print(f"Loaded {len(questions)} validation questions.")


# -------------------------------------------------
# 3. Helper functions to run different retrievers
# -------------------------------------------------
def retrieve_bm25(query: str, top_k=5) -> List[str]:
    res = pipeline.local_bm25.search(query, top_k=top_k)
    return [doc_id for doc_id, _ in res]


def retrieve_dense(query: str, top_k=5) -> List[str]:
    emb = pipeline.embedder.embed_query(query)
    res = pipeline.local_dense.search(emb, top_k=top_k)
    return [doc_id for doc_id, _ in res]


def retrieve_hybrid(query: str, top_k=5) -> List[str]:
    res = pipeline.hybrid_retriever.retrieve(query, final_top_k=top_k)
    return [doc.doc_id for doc in res]


# -------------------------------------------------
# 4. Evaluate all retrievers
# -------------------------------------------------
bm25_results = []
dense_results = []
hybrid_results = []

print("\nRunning retrieval for all questions...\n")

for q in tqdm(questions):
    bm25_results.append(retrieve_bm25(q, top_k=5))
    dense_results.append(retrieve_dense(q, top_k=5))
    hybrid_results.append(retrieve_hybrid(q, top_k=5))


# -------------------------------------------------
# 5. Compute metrics
# -------------------------------------------------
print("\n=== Retrieval Quality Metrics ===\n")

for k in [1, 3, 5]:
    print(f"----- k = {k} -----")

    r_bm25 = recall_at_k(gold_sets, bm25_results, k)
    p_bm25 = precision_at_k(gold_sets, bm25_results, k)

    r_dense = recall_at_k(gold_sets, dense_results, k)
    p_dense = precision_at_k(gold_sets, dense_results, k)

    r_hybrid = recall_at_k(gold_sets, hybrid_results, k)
    p_hybrid = precision_at_k(gold_sets, hybrid_results, k)

    print(f"BM25        R@{k}: {r_bm25:.3f}   P@{k}: {p_bm25:.3f}")
    print(f"Dense       R@{k}: {r_dense:.3f}   P@{k}: {p_dense:.3f}")
    print(f"Hybrid+DAT  R@{k}: {r_hybrid:.3f}   P@{k}: {p_hybrid:.3f}")
    print()


# -------------------------------------------------
# 6. Save results to a CSV for report
# -------------------------------------------------
import csv

OUT_PATH = Path("eval/retrieval_results.csv")
OUT_PATH.parent.mkdir(exist_ok=True)

with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Retriever", "k", "Recall", "Precision"])

    for k in [1, 3, 5]:
        writer.writerow(["BM25", k,
                         recall_at_k(gold_sets, bm25_results, k),
                         precision_at_k(gold_sets, bm25_results, k)])

        writer.writerow(["Dense", k,
                         recall_at_k(gold_sets, dense_results, k),
                         precision_at_k(gold_sets, dense_results, k)])

        writer.writerow(["Hybrid+DAT", k,
                         recall_at_k(gold_sets, hybrid_results, k),
                         precision_at_k(gold_sets, hybrid_results, k)])

print(f"\nResults saved to {OUT_PATH}")
print("Done.")
