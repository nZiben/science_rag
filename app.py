import sys
from pathlib import Path

# --- Добавляем путь к проекту ---
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

import streamlit as st
import json

from src.rag_pipeline import ScienceRAGPipeline
from src.hybrid_retrieval.hybrid_retriever_with_DAT import HybridRetrieverWithDAT


# ------------------------------
# Initialize pipeline (cached!)
# ------------------------------
@st.cache_resource(show_spinner=True)
def load_pipeline():
    return ScienceRAGPipeline()


pipeline = load_pipeline()


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="Science RAG", layout="wide")

st.title("🔬 Science RAG with DAT (Local + Web + Hybrid)")
st.write("RAG-система с локальным индексом arXiv, Web-RAG, и гибридным DAT-retrieval.")

query = st.text_input(
    "Введите научный вопрос:",
    placeholder="Например: How does quantization affect transformer inference?",
)

mode = st.radio(
    "Режим RAG:",
    ["Local", "Hybrid (DAT)", "Web"],
    horizontal=True,
)


run_btn = st.button("🔍 Выполнить поиск")

if run_btn and query.strip():

    with st.spinner("Формируем контекст и генерируем ответ..."):
        if mode == "Local":
            answer = pipeline.answer_local(query)

        elif mode == "Hybrid (DAT)":
            answer = pipeline.answer_local_hybrid(query)

        elif mode == "Web":
            answer = pipeline.answer_web(query)

    st.subheader("🧠 Ответ модели:")
    st.write(answer.answer)

    st.subheader("📚 Использованные контексты:")

    for ctx in answer.contexts:
        with st.expander(f"{ctx.doc_id}  |  score={ctx.score:.3f}"):

            if ctx.url:
                st.markdown(f"**URL:** {ctx.url}")
            if ctx.title:
                st.markdown(f"**Title:** {ctx.title}")

            st.markdown(f"**Source:** `{ctx.source}`")
            st.write(ctx.text)


    # --------------------------
    # Hybrid debugging (DAT α)
    # --------------------------
    if mode == "Hybrid (DAT)":
        st.subheader("⚙️ DAT Debug Info")

        alpha = pipeline.hybrid_retriever.dat_alpha.last_alpha
        bm25_score = pipeline.hybrid_retriever.dat_alpha.last_bm25_score
        dense_score = pipeline.hybrid_retriever.dat_alpha.last_dense_score

        st.metric("α(q) — weight of dense retrieval", f"{alpha:.3f}")
        st.write(f"BM25 score = **{bm25_score}**")
        st.write(f"Dense score = **{dense_score}**")
