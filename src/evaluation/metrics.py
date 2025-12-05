"""Retrieval metrics: Recall@k and Precision@k for local RAG."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set


def recall_at_k(
    gold_docs: Sequence[Set[str]],
    retrieved_docs: Sequence[List[str]],
    k: int,
) -> float:
    """Compute mean Recall@k."""
    assert len(gold_docs) == len(retrieved_docs)
    recalls: List[float] = []
    for gold, retrieved in zip(gold_docs, retrieved_docs):
        if not gold:
            continue
        hit = len(gold.intersection(set(retrieved[:k])))
        recalls.append(hit / float(len(gold)))
    return float(sum(recalls) / len(recalls)) if recalls else 0.0


def precision_at_k(
    gold_docs: Sequence[Set[str]],
    retrieved_docs: Sequence[List[str]],
    k: int,
) -> float:
    """Compute mean Precision@k."""
    assert len(gold_docs) == len(retrieved_docs)
    precisions: List[float] = []
    for gold, retrieved in zip(gold_docs, retrieved_docs):
        topk = retrieved[:k]
        if not topk:
            continue
        hit = len(gold.intersection(set(topk)))
        precisions.append(hit / float(len(topk)))
    return float(sum(precisions) / len(precisions)) if precisions else 0.0


def load_validation_qa(path: Path) -> Dict[str, Dict[str, object]]:
    """
    Load QA pairs from JSONL.

    Each line:
        {
          "id": "q1",
          "question": "...",
          "gold_chunk_ids": ["chunk_001", ...]
        }
    """
    qa: Dict[str, Dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        qa_id = record["id"]
        qa[qa_id] = record
    return qa
