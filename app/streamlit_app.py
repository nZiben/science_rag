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
                    max_planner_iterations=2,
                )
            elif mode == "Web-RAG":
                rag_answer = pipeline.answer_web(
                    query,
                    max_web_results=top_k,
                    max_planner_iterations=2,
                )
            else:
                rag_answer = pipeline.answer_local_hybrid(
                    query,
                    top_k=top_k,
                    max_planner_iterations=2,
                )

        # Show refinement information only if there were actual refinements
        if rag_answer.refinement_history and len(rag_answer.refinement_history) > 0:
            # Check if there were any actual refinements
            has_refinements = any(r.was_refined for r in rag_answer.refinement_history)
            
            if has_refinements:
                st.subheader("Процесс уточнения запроса")
                
                for i, refinement in enumerate(rag_answer.refinement_history):
                    if refinement.was_refined and i < len(rag_answer.refinement_history) - 1:
                        # Show refinement message for iterations that were refined
                        next_refinement = rag_answer.refinement_history[i + 1]
                        
                        st.warning(
                            f"⚠️ **Итерация {refinement.iteration}:** "
                            f"Модель оценила качество как недостаточное (средний score: {refinement.mean_score:.3f})"
                        )
                        
                        # Show strategy-specific information
                        strategy_text = ""
                        if refinement.strategy == "reformulate":
                            strategy_text = (
                                f"🔄 **Стратегия: Переформулировка запроса**\n\n"
                                f"**Новый запрос:** {next_refinement.query}\n\n"
                                f"Запрос был переформулирован для улучшения поиска релевантных документов."
                            )
                        elif refinement.strategy == "increase_top_k":
                            strategy_text = (
                                f"📈 **Стратегия: Увеличение количества контекста**\n\n"
                                f"**Top-k увеличен:** {refinement.top_k} → {next_refinement.top_k}\n\n"
                                f"Модель решила, что нужно больше контекста для полного ответа. "
                                f"Количество извлекаемых документов увеличено."
                            )
                        elif refinement.strategy == "both":
                            strategy_text = (
                                f"🔄📈 **Стратегия: Переформулировка запроса и увеличение контекста**\n\n"
                                f"**Новый запрос:** {next_refinement.query}\n\n"
                                f"**Top-k увеличен:** {refinement.top_k} → {next_refinement.top_k}\n\n"
                                f"Модель решила применить обе стратегии одновременно: "
                                f"переформулировать запрос для лучшего поиска и увеличить количество контекста."
                            )
                        else:
                            # Fallback for unknown strategy
                            strategy_text = (
                                f"🔄 **Выполнено автоматическое уточнение:**\n\n"
                                f"**Новый запрос:** {next_refinement.query}\n\n"
                                f"**Top-k увеличен:** {refinement.top_k} → {next_refinement.top_k}"
                            )
                        
                        st.info(strategy_text)
                        
                        if i < len(rag_answer.refinement_history) - 2:
                            st.markdown("---")

        # Show query history if there were reformulations
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
