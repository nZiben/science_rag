"""End-to-end RAG pipelines for local, web and hybrid modes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Sequence
import os

from mistralai import Mistral  # type: ignore

from src.embeddings.embedder import Embedder, EmbeddingBackend
from src.hybrid_retrieval.hybrid_retriever_with_DAT import (
    DATAlphaCalculator,
    HybridRetrieverWithDAT,
)
from src.local_rag.bm25_index import BM25Index
from src.local_rag.faiss_builder import FAISSIndex
from src.web_rag.web_retriever import (
    WebDocument,
    build_temp_web_index,
    tavily_search,
)


@dataclass
class RetrievedContext:
    """Context element passed to the generator."""

    doc_id: str
    text: str
    source: str
    score: float
    url: str | None = None
    title: str | None = None


@dataclass
class RAGAnswer:
    """Final RAG answer with supporting contexts."""

    answer: str
    contexts: List[RetrievedContext]
    mode: Literal["local", "web", "hybrid"]


class ScienceRAGPipeline:
    """High-level RAG pipeline used by the Streamlit app."""

    def __init__(
        self,
        local_chunks_path: Path = Path("data/chunks/local/chunks.jsonl"),
        local_faiss_index_path: Path = Path("data/faiss/local_dense.index"),
        local_faiss_meta_path: Path = Path("data/faiss/local_dense_meta.json"),
        local_bm25_index_path: Path = Path("data/chunks/local/bm25_index.json"),
        local_bm25_meta_path: Path = Path("data/chunks/local/bm25_meta.json"),
        mistral_model: str = "mistral-small-latest",
        embedding_backend: EmbeddingBackend = EmbeddingBackend.MISTRAL,
    ) -> None:
        self.local_chunks_path = local_chunks_path
        self._chunk_texts: Dict[str, str] = {}
        self._chunk_meta: Dict[str, Dict[str, str]] = {}
        with open(local_chunks_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"❌ JSON ERROR on line {i}: {e}")
                    print("Line content:", line[:200])
                    continue  # или raise — как хочешь

                chunk_id = record["chunk_id"]
                self._chunk_texts[chunk_id] = record["text"]
                self._chunk_meta[chunk_id] = record.get("metadata", {})
                

        # Local retrievers.
        self.local_dense = FAISSIndex.load(
            local_faiss_index_path,
            local_faiss_meta_path,
        )
        self.local_bm25 = BM25Index.load(
            local_bm25_index_path,
            local_bm25_meta_path,
        )
        self.embedder = Embedder(
            backend=EmbeddingBackend.SENTENCE_TRANSFORMERS,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )


        self.mistral_model = mistral_model
        self.dat_alpha = DATAlphaCalculator()

        self.hybrid_retriever = HybridRetrieverWithDAT(
            bm25_index=self.local_bm25,
            dense_index=self.local_dense,
            chunks_path=self.local_chunks_path,
            embedder=self.embedder,
            dat_alpha=self.dat_alpha,
        )

    def _generate_answer(
        self,
        query: str,
        contexts: Sequence[RetrievedContext],
    ) -> str:
        """Call Mistral chat completion with retrieved contexts."""
        from textwrap import indent

        system_prompt = (
            "You are a helpful scientific assistant. "
            "Answer the question strictly based on the provided context. "
            "Cite sources with [source_id] in the answer where appropriate. "
            "If the answer is not in the context, say that the information "
            "is not available."
        )

        context_blocks = []
        for ctx in contexts:
            header = f"[{ctx.doc_id}]"
            context_blocks.append(f"{header}\n{ctx.text}")

        context_str = indent("\n\n".join(context_blocks), "    ")

        content = (
            f"Question:\n{query}\n\n"
            f"Context:\n{context_str}\n\n"
            "Answer:"
        )
        client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        res = client.chat.complete(
                model=self.mistral_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                stream=False,
            )
        return res.choices[0].message.content.strip()

    def _make_local_contexts(
        self,
        results: Sequence[tuple[str, float]],
        mode: str,
    ) -> List[RetrievedContext]:
        contexts: List[RetrievedContext] = []
        for doc_id, score in results:
            text = self._chunk_texts.get(doc_id, "")
            meta = self._chunk_meta.get(doc_id, {})
            source_id = meta.get("source_id", "")
            contexts.append(
                RetrievedContext(
                    doc_id=doc_id,
                    text=text,
                    source=f"{mode}:{source_id}",
                    score=score,
                )
            )
        return contexts

    def answer_local(self, query: str, top_k: int = 5) -> RAGAnswer:
        """Local RAG over arXiv corpus (dense only)."""
        query_emb = self.embedder.embed_query(query)
        results = self.local_dense.search(query_emb, top_k=top_k)
        contexts = self._make_local_contexts(results, mode="local")
        answer = self._generate_answer(query, contexts)
        return RAGAnswer(answer=answer, contexts=contexts, mode="local")

    def answer_local_hybrid(self, query: str, top_k: int = 5) -> RAGAnswer:
        """Hybrid local RAG with DAT (BM25 + dense)."""
        hybrid_results = self.hybrid_retriever.retrieve(
            query=query,
            top_k_bm25=top_k,
            top_k_dense=top_k,
            final_top_k=top_k,
        )
        results = [(r.doc_id, r.score) for r in hybrid_results]
        contexts = self._make_local_contexts(results, mode="hybrid")
        answer = self._generate_answer(query, contexts)
        return RAGAnswer(answer=answer, contexts=contexts, mode="hybrid")

    def answer_web(self, query: str, max_web_results: int = 5) -> RAGAnswer:
        """Web-RAG over Tavily search results."""
        web_docs: List[WebDocument] = tavily_search(
            query=query,
            max_results=max_web_results,
            search_depth="advanced",
        )
        if not web_docs:
            return RAGAnswer(
                answer="Не удалось найти релевантные источники в интернете.",
                contexts=[],
                mode="web",
            )

        faiss_index, mapping = build_temp_web_index(web_docs, self.embedder)
        query_emb = self.embedder.embed_query(query)
        dense_results = faiss_index.search(query_emb, top_k=min(5, len(web_docs)))

        contexts: List[RetrievedContext] = []
        for doc_id, score in dense_results:
            doc = mapping[doc_id]
            contexts.append(
                RetrievedContext(
                    doc_id=doc_id,
                    text=doc.content,
                    source="web",
                    score=score,
                    url=doc.url,
                    title=doc.title,
                )
            )

        answer = self._generate_answer(query, contexts)
        return RAGAnswer(answer=answer, contexts=contexts, mode="web")
