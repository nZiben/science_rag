"""Streamlit UI for Science RAG."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
    st.sidebar.header("Планировщик запросов")
    enable_planner = st.sidebar.checkbox(
        "Включить планировщик запросов",
        value=False,
        help="Автоматически переформулирует вопрос при низком качестве ответа и увеличивает top_k на 3",
    )
    planner_threshold = st.sidebar.slider(
        "Порог качества (threshold)",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Средний score документов должен быть >= этого значения",
        disabled=not enable_planner,
    )

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
                rag_answer = pipeline.answer_local(
                    query,
                    top_k=top_k,
                    enable_planner=enable_planner,
                    planner_threshold=planner_threshold,
                    max_planner_iterations=1,
                )
            elif mode == "Web-RAG":
                rag_answer = pipeline.answer_web(
                    query,
                    max_web_results=top_k,
                    enable_planner=enable_planner,
                    planner_threshold=planner_threshold,
                    max_planner_iterations=1,
                )
            else:
                rag_answer = pipeline.answer_local_hybrid(
                    query,
                    top_k=top_k,
                    enable_planner=enable_planner,
                    planner_threshold=planner_threshold,
                    max_planner_iterations=1,
                )

        # Show refinement information if planner was used
        if rag_answer.refinement_history and len(rag_answer.refinement_history) > 0:
            st.subheader("Процесс уточнения запроса")
            
            for i, refinement in enumerate(rag_answer.refinement_history):
                if refinement.was_refined and i < len(rag_answer.refinement_history) - 1:
                    # Show refinement message for iterations that were refined
                    next_refinement = rag_answer.refinement_history[i + 1]
                    
                    st.warning(
                        f"⚠️ **Итерация {refinement.iteration}:** "
                        f"Недостаточный score ({refinement.mean_score:.3f} < {planner_threshold:.3f})"
                    )
                    
                    st.info(
                        f"🔄 **Выполнено уточнение:**\n\n"
                        f"**Новый запрос:** {next_refinement.query}\n\n"
                        f"**Top-k увеличен:** {refinement.top_k} → {next_refinement.top_k}"
                    )
                    
                    if i < len(rag_answer.refinement_history) - 2:
                        st.markdown("---")

        # Show query history if planner was used (simplified version)
        if rag_answer.query_history and len(rag_answer.query_history) > 1:
            with st.expander("📝 История всех запросов"):
                for i, q in enumerate(rag_answer.query_history, 1):
                    st.markdown(f"**Попытка {i}:** {q}")

        st.subheader("Ответ")
        st.markdown(rag_answer.answer)

        # Show quality metrics
        if rag_answer.contexts:
            mean_score = sum(ctx.score for ctx in rag_answer.contexts) / len(
                rag_answer.contexts
            )
            max_score = max(ctx.score for ctx in rag_answer.contexts)
            st.caption(
                f"Средний score: {mean_score:.3f} | Максимальный score: {max_score:.3f}"
            )

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
