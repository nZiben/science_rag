"""Streamlit UI for Science RAG."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.embeddings.embedder import EmbeddingBackend
from src.rag_pipeline import ScienceRAGPipeline


def _init_pipeline() -> ScienceRAGPipeline:
    """Lazy-initialize global pipeline and cache in session state."""
    if "pipeline" not in st.session_state:
        st.session_state["pipeline"] = ScienceRAGPipeline(
            local_chunks_path=Path("data/chunks/local/chunks.jsonl"),
            local_faiss_index_path=Path("data/faiss/local_dense.index"),
            local_faiss_meta_path=Path("data/faiss/local_dense_meta.json"),
            local_bm25_index_path=Path("data/chunks/local/bm25_index.json"),
            local_bm25_meta_path=Path("data/chunks/local/bm25_meta.json"),
            mistral_model="mistral-large-latest",
            embedding_backend=EmbeddingBackend.MISTRAL,
        )
    return st.session_state["pipeline"]


def main() -> None:
    """Run Streamlit app."""
    st.set_page_config(
        page_title="Science RAG",
        page_icon="🔬",
        layout="wide",
    )

    st.title("🔬 Science RAG")
    st.caption(
        "Ответы на научные вопросы на основе arXiv (локально) "
        "и интернета (Web-RAG) с гибридным поиском DAT."
    )

    # Sidebar controls.
    st.sidebar.header("Настройки")
    mode = st.sidebar.radio(
        "Режим работы:",
        options=["Локальный RAG", "Web-RAG", "Гибридный локальный (DAT)"],
    )
    top_k = st.sidebar.slider("Top-k документов", min_value=3, max_value=10, value=5)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Переменные окружения:**")
    st.sidebar.code("MISTRAL_API_KEY=\nTAVILY_API_KEY=", language="bash")

    query = st.text_area(
        "Введите научный вопрос:",
        placeholder="Например: Explain the difference between L1 and L2 regularization "
        "in linear models.",
        height=120,
    )

    if st.button("Получить ответ", type="primary") and query.strip():
        pipeline = _init_pipeline()
        with st.spinner("Выполняется retrieval и генерация ответа..."):
            if mode == "Локальный RAG":
                rag_answer = pipeline.answer_local(query, top_k=top_k)
            elif mode == "Web-RAG":
                rag_answer = pipeline.answer_web(query, max_web_results=top_k)
            else:
                rag_answer = pipeline.answer_local_hybrid(query, top_k=top_k)

        st.subheader("Ответ")
        st.markdown(rag_answer.answer)

        st.subheader("Использованные источники")
        for ctx in rag_answer.contexts:
            with st.expander(f"{ctx.doc_id} | score={ctx.score:.3f} | {ctx.source}"):
                if ctx.title:
                    st.markdown(f"**{ctx.title}**")
                if ctx.url:
                    st.markdown(f"[Открыть источник]({ctx.url})")
                st.markdown(ctx.text[:1500] + ("..." if len(ctx.text) > 1500 else ""))


if __name__ == "__main__":
    main()
