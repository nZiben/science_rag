"""FAISS index builder and search utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import faiss  # type: ignore
import numpy as np


@dataclass
class DenseIndexMeta:
    """Metadata associated with a FAISS index."""

    doc_ids: List[str]
    dim: int


class FAISSIndex:
    """
    Thin wrapper around a FAISS inner-product index.

    Assumes embeddings are L2-normalized if cosine similarity is desired.
    """

    def __init__(self, index: faiss.Index, meta: DenseIndexMeta):
        self.index = index
        self.meta = meta

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        doc_ids: Sequence[str],
    ) -> "FAISSIndex":
        """
        Build FAISS index from embeddings.

        Parameters
        ----------
        embeddings:
            2D array with shape (n_docs, dim).
        doc_ids:
            Stable identifiers for each embedding.

        Returns
        -------
        FAISSIndex
        """
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")
        n_docs, dim = embeddings.shape
        if n_docs != len(doc_ids):
            raise ValueError("embeddings and doc_ids length mismatch.")

        # Normalize for cosine similarity.
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype("float32"))

        meta = DenseIndexMeta(doc_ids=list(doc_ids), dim=dim)
        return cls(index=index, meta=meta)

    def save(self, index_path: Path, meta_path: Path) -> None:
        """Persist index and metadata to disk."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))

        meta_dict = {"doc_ids": self.meta.doc_ids, "dim": self.meta.dim}
        meta_path.write_text(json.dumps(meta_dict, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, index_path: Path, meta_path: Path) -> "FAISSIndex":
        """Load index and metadata from disk."""
        index = faiss.read_index(str(index_path))
        meta_dict: Dict[str, object] = json.loads(meta_path.read_text(encoding="utf-8"))
        meta = DenseIndexMeta(
            doc_ids=list(meta_dict["doc_ids"]),
            dim=int(meta_dict["dim"]),
        )
        return cls(index=index, meta=meta)

    def search(
        self,
        query_emb: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Search index with a single query embedding.

        Returns
        -------
        list of (doc_id, score) sorted by descending score.
        """
        if query_emb.ndim == 1:
            query_emb = query_emb[None, :]
        faiss.normalize_L2(query_emb)
        scores, idx = self.index.search(query_emb.astype("float32"), top_k)
        results: List[Tuple[str, float]] = []
        for i, score in zip(idx[0], scores[0]):
            if i < 0:
                continue
            doc_id = self.meta.doc_ids[int(i)]
            results.append((doc_id, float(score)))
        return results


def build_and_save_faiss_index(
    embeddings_path: Path,
    doc_ids_path: Path,
    index_path: Path,
    meta_path: Path,
) -> None:
    """
    Build FAISS index from embeddings and save it.

    embeddings_path:
        .npy file with shape (n_docs, dim).
    doc_ids_path:
        .txt file with one doc_id per line in the same order.
    """
    embeddings = np.load(embeddings_path)
    doc_ids = [
        line.strip()
        for line in doc_ids_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    index = FAISSIndex.build(embeddings, doc_ids)
    index.save(index_path, meta_path)


def main() -> None:
    """CLI entry point for building FAISS index from disk embeddings."""
    import argparse

    parser = argparse.ArgumentParser(description="Build FAISS index for local RAG.")
    parser.add_argument(
        "--embeddings_path",
        type=Path,
        default=Path("data/chunks/local/embeddings.npy"),
    )
    parser.add_argument(
        "--doc_ids_path",
        type=Path,
        default=Path("data/chunks/local/doc_ids.txt"),
    )
    parser.add_argument(
        "--index_path",
        type=Path,
        default=Path("data/faiss/local_dense.index"),
    )
    parser.add_argument(
        "--meta_path",
        type=Path,
        default=Path("data/faiss/local_dense_meta.json"),
    )
    args = parser.parse_args()

    build_and_save_faiss_index(
        embeddings_path=args.embeddings_path,
        doc_ids_path=args.doc_ids_path,
        index_path=args.index_path,
        meta_path=args.meta_path,
    )


if __name__ == "__main__":
    main()
