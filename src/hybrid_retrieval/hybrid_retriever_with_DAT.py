"""Hybrid retriever with Dynamic Alpha Tuning (DAT)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from mistralai import Mistral  # type: ignore
import numpy as np

from src.embeddings.embedder import Embedder
from src.local_rag.bm25_index import BM25Index
from src.local_rag.faiss_builder import FAISSIndex


@dataclass
class RetrievedDoc:
    """Unified retrieval result."""

    doc_id: str
    score: float
    source: str  # "bm25" | "dense" | "hybrid"


class DATAlphaCalculator:
    """
    LLM-based effectiveness scorer and dynamic alpha calculator.

    Follows the idea of DAT: for each query, the LLM grades top-1 BM25 and
    top-1 dense results on a 0–5 scale, then alpha is derived from normalized
    scores.
    """

    def __init__(
        self,
        model: str = "mistral-small-latest",
        system_prompt: str | None = None,
    ) -> None:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY environment variable is missing.")
        self._client = Mistral(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt or (
            "You are a careful evaluator of retrieval quality. "
            "Given a user query and a retrieved text passage, "
            "assign an integer score from 0 (irrelevant) to 5 (perfect answer). "
            "Respond with ONLY the integer."
        )
        self.last_alpha = None
        self.last_bm25_score = None
        self.last_dense_score = None

    def _score_single(self, query: str, text: str) -> int:
        """Ask the LLM to grade a single retrieved chunk."""
        content = (
            f"Query:\n{query}\n\nRetrieved passage:\n{text}\n\n"
            "Score (0-5):"
        )
        res = self._client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": content},
                ],
                stream=False,
            )
        reply = res.choices[0].message.content.strip()
        try:
            score = int("".join(ch for ch in reply if ch.isdigit()))
        except ValueError:
            score = 0
        return max(0, min(score, 5))
    def compute_alpha(
        self,
        query: str,
        bm25_top1_text: str | None,
        dense_top1_text: str | None,
    ) -> float:

        # -----------------------------------------
        # Fallback cases
        # -----------------------------------------
        if bm25_top1_text is None and dense_top1_text is None:
            self.last_alpha = 0.5
            return 0.5

        if bm25_top1_text is None:
            self.last_alpha = 1.0
            return 1.0

        if dense_top1_text is None:
            self.last_alpha = 0.0
            return 0.0

        # -----------------------------------------
        # LLM scoring
        # -----------------------------------------
        s_bm25 = self._score_single(query, bm25_top1_text)
        s_dense = self._score_single(query, dense_top1_text)

        self.last_bm25_score = s_bm25
        self.last_dense_score = s_dense

        if s_bm25 == 0 and s_dense == 0:
            self.last_alpha = 0.5
            return 0.5

        # Priority rules
        if s_dense - s_bm25 >= 2:
            self.last_alpha = 0.9
            return 0.9

        if s_bm25 - s_dense >= 2:
            self.last_alpha = 0.1
            return 0.1

        # -----------------------------------------
        # Default DAT formula
        # -----------------------------------------
        alpha = s_dense / float(s_dense + s_bm25)
        alpha = max(0.0, min(alpha, 1.0))

        self.last_alpha = alpha
        return alpha



class HybridRetrieverWithDAT:
    """
    Hybrid retriever that combines BM25 and dense retrieval using DAT.

    Responsibilities:
    - Perform BM25 and dense retrieval.
    - Ask LLM to grade top-1 results.
    - Compute alpha(q).
    - Fuse scores for final ranking.
    """

    def __init__(
        self,
        bm25_index: BM25Index,
        dense_index: FAISSIndex,
        chunks_path: Path,
        embedder: Embedder,
        dat_alpha: DATAlphaCalculator,
    ) -> None:
        self.bm25_index = bm25_index
        self.dense_index = dense_index
        self.embedder = embedder
        self.dat_alpha = dat_alpha

        # Load mapping chunk_id -> text.
        self._chunk_texts: Dict[str, str] = {}
        with open(chunks_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"❌ JSON ERROR in HybridRetriever on line {i}: {e}")
                    print("Problematic line (first 200 chars):", line[:200])
                    continue  # Skip problematic lines
                
                chunk_id = record.get("chunk_id")
                text = record.get("text")
                if chunk_id and text:
                    self._chunk_texts[chunk_id] = text
                else:
                    print(f"⚠️ Missing fields in line {i}: {record}")
        
        print(f"✅ Loaded {len(self._chunk_texts)} chunks from {chunks_path}")

    def _get_chunk_text(self, chunk_id: str) -> str | None:
        return self._chunk_texts.get(chunk_id)

    def retrieve(
        self,
        query: str,
        top_k_bm25: int = 10,
        top_k_dense: int = 10,
        final_top_k: int = 8,
    ) -> List[RetrievedDoc]:
        """
        Perform DAT-based hybrid retrieval.

        Returns
        -------
        list of RetrievedDoc
            Final ranked list of hybrid results.
        """
        # 1. BM25 retrieval.
        bm25_results = self.bm25_index.search(query, top_k=top_k_bm25)

        # 2. Dense retrieval.
        query_emb = self.embedder.embed_query(query)
        dense_results = self.dense_index.search(query_emb, top_k=top_k_dense)

        # 3. Prepare texts for top-1 of each method.
        bm25_top1_text = (
            self._get_chunk_text(bm25_results[0][0]) if bm25_results else None
        )
        dense_top1_text = (
            self._get_chunk_text(dense_results[0][0]) if dense_results else None
        )

        # 4. Compute alpha(q).
        alpha = self.dat_alpha.compute_alpha(query, bm25_top1_text, dense_top1_text)

        # 5. Normalize and fuse scores per document.
        fused_scores: Dict[str, float] = {}
        bm25_score_map = {doc_id: score for doc_id, score in bm25_results}
        dense_score_map = {doc_id: score for doc_id, score in dense_results}

        # Min-max normalize within each method for fusion.
        def normalize(score_map: Dict[str, float]) -> Dict[str, float]:
            if not score_map:
                return {}
            scores_arr = np.asarray(list(score_map.values()))
            min_s = scores_arr.min()
            max_s = scores_arr.max()
            if max_s == min_s:
                return {k: 1.0 for k in score_map}
            return {k: (v - min_s) / (max_s - min_s) for k, v in score_map.items()}

        bm25_norm = normalize(bm25_score_map)
        dense_norm = normalize(dense_score_map)

        all_ids = set(bm25_norm.keys()) | set(dense_norm.keys())

        for doc_id in all_ids:
            b = bm25_norm.get(doc_id, 0.0)
            d = dense_norm.get(doc_id, 0.0)
            fused_scores[doc_id] = (1.0 - alpha) * b + alpha * d

        # 6. Sort by fused score.
        ranked_ids = sorted(
            fused_scores.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:final_top_k]

        results: List[RetrievedDoc] = []
        for doc_id, score in ranked_ids:
            results.append(
                RetrievedDoc(
                    doc_id=doc_id,
                    score=float(score),
                    source="hybrid",
                )
            )
        return results
