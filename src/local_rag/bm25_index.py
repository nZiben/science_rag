"""BM25 index over local chunks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from rank_bm25 import BM25Okapi


@dataclass
class BM25Meta:
    """Metadata for BM25 index."""
    doc_ids: List[str]


class BM25Index:
    """
    BM25 retrieval over tokenized chunks.
    Stores corpus manually because BM25Okapi doesn't expose it in all versions.
    """

    def __init__(self, bm25: BM25Okapi, meta: BM25Meta, corpus: List[List[str]]):
        self.bm25 = bm25
        self.meta = meta
        self.corpus = corpus

    @classmethod
    def build(
        cls,
        tokenized_docs: Sequence[Sequence[str]],
        doc_ids: Sequence[str],
    ) -> "BM25Index":

        corpus = [list(toks) for toks in tokenized_docs]
        bm25 = BM25Okapi(corpus)
        meta = BM25Meta(doc_ids=list(doc_ids))
        return cls(bm25=bm25, meta=meta, corpus=corpus)

    def save(self, index_path: Path, meta_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)

        data = {"corpus": self.corpus}
        index_path.write_text(
            json.dumps(data, ensure_ascii=False),
            encoding="utf-8",
        )

        meta_path.write_text(
            json.dumps({"doc_ids": self.meta.doc_ids}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_path: Path, meta_path: Path) -> "BM25Index":
        data = json.loads(index_path.read_text(encoding="utf-8"))
        corpus: List[List[str]] = data["corpus"]

        meta_dict: Dict[str, object] = json.loads(meta_path.read_text(encoding="utf-8"))
        doc_ids = list(meta_dict["doc_ids"])

        bm25 = BM25Okapi(corpus)
        meta = BM25Meta(doc_ids=doc_ids)
        return cls(bm25=bm25, meta=meta, corpus=corpus)

    # ⭐⭐⭐ добавлен рабочий метод поиска — ЭТОГО НЕ ХВАТАЛО ⭐⭐⭐
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)

        scored = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results: List[Tuple[str, float]] = []
        for idx, score in scored:
            doc_id = self.meta.doc_ids[idx]
            results.append((doc_id, float(score)))
        return results


def build_bm25_from_chunks_jsonl(
    chunks_path: Path,
    index_path: Path,
    meta_path: Path,
) -> None:

    tokenized_docs: List[List[str]] = []
    doc_ids: List[str] = []

    with open(chunks_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"❌ JSON error on line {i}: {e}")
                print(repr(line[:200]))
                raise

            text = record["text"]
            chunk_id = record["chunk_id"]

            tokenized_docs.append(text.lower().split())
            doc_ids.append(chunk_id)

    index = BM25Index.build(tokenized_docs, doc_ids)
    index.save(index_path, meta_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build BM25 index over local chunks.")
    parser.add_argument("--chunks_path", type=Path, default=Path("data/chunks/local/chunks.jsonl"))
    parser.add_argument("--index_path", type=Path, default=Path("data/chunks/local/bm25_index.json"))
    parser.add_argument("--meta_path", type=Path, default=Path("data/chunks/local/bm25_meta.json"))
    args = parser.parse_args()

    build_bm25_from_chunks_jsonl(
        chunks_path=args.chunks_path,
        index_path=args.index_path,
        meta_path=args.meta_path,
    )


if __name__ == "__main__":
    main()
