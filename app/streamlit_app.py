"""Streamlit UI for Science RAG."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import streamlit as st
import traceback

from src.embeddings.embedder import EmbeddingBackend
from src.rag_pipeline import ScienceRAGPipeline


def _init_pipeline() -> ScienceRAGPipeline:
    """Lazy-initialize global pipeline and cache in session state."""
    if "pipeline" not in st.session_state:
        try:
            # Проверка наличия API ключа Mistral
            if not os.getenv("MISTRAL_API_KEY"):
                st.error(
                    "❌ **Ошибка конфигурации:** Не найден API ключ Mistral.\n\n"
                    "Пожалуйста, создайте файл `.env` в корне проекта и добавьте:\n"
                    "```\nMISTRAL_API_KEY=your_api_key_here\n```"
                )
                st.stop()
            
            # Проверка наличия необходимых файлов
            required_files = [
                Path("data/chunks/local/chunks.jsonl"),
                Path("data/faiss/local_dense.index"),
                Path("data/faiss/local_dense_meta.json"),
                Path("data/chunks/local/bm25_index.json"),
                Path("data/chunks/local/bm25_meta.json"),
            ]
            
            missing_files = [f for f in required_files if not f.exists()]
            if missing_files:
                st.error(
                    "❌ **Ошибка:** Не найдены необходимые файлы данных:\n\n"
                    + "\n".join(f"- `{f}`" for f in missing_files)
                    + "\n\nПожалуйста, убедитесь, что данные были загружены и обработаны."
                )
                st.stop()
            
            st.session_state["pipeline"] = ScienceRAGPipeline(
                local_chunks_path=Path("data/chunks/local/chunks.jsonl"),
                local_faiss_index_path=Path("data/faiss/local_dense.index"),
                local_faiss_meta_path=Path("data/faiss/local_dense_meta.json"),
                local_bm25_index_path=Path("data/chunks/local/bm25_index.json"),
                local_bm25_meta_path=Path("data/chunks/local/bm25_meta.json"),
                mistral_model="mistral-large-latest",
                embedding_backend=EmbeddingBackend.MISTRAL,
            )
        except FileNotFoundError as e:
            st.error(
                f"❌ **Ошибка загрузки файла:** Не удалось найти файл.\n\n"
                f"**Детали:** {str(e)}\n\n"
                f"Пожалуйста, убедитесь, что все необходимые файлы данных существуют."
            )
            st.stop()
        except json.JSONDecodeError as e:
            st.error(
                f"❌ **Ошибка чтения данных:** Неверный формат JSON в файле данных.\n\n"
                f"**Детали:** {str(e)}\n\n"
                f"Возможно, файл данных поврежден. Попробуйте пересоздать индексы."
            )
            st.stop()
        except Exception as e:
            st.error(
                f"❌ **Ошибка инициализации pipeline:** Не удалось загрузить pipeline.\n\n"
                f"**Тип ошибки:** {type(e).__name__}\n"
                f"**Сообщение:** {str(e)}\n\n"
                f"**Детали:**\n```\n{traceback.format_exc()}\n```"
            )
            st.stop()
    
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

    query = st.text_area(
        "Введите научный вопрос:",
        placeholder="Например: Explain the difference between L1 and L2 regularization "
        "in linear models.",
        height=120,
    )

    if st.button("Получить ответ", type="primary") and query.strip():
        try:
            pipeline = _init_pipeline()
        except Exception as e:
            st.error(
                f"❌ **Ошибка инициализации:** Не удалось инициализировать pipeline.\n\n"
                f"**Сообщение:** {str(e)}"
            )
            st.stop()
        
        try:
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
        except KeyError as e:
            st.error(
                f"❌ **Ошибка API:** Проблема с конфигурацией API ключа.\n\n"
                f"**Детали:** {str(e)}\n\n"
                f"Проверьте, что переменная окружения `MISTRAL_API_KEY` или `TAVILY_API_KEY` "
                f"установлена корректно."
            )
            if st.checkbox("Показать детали ошибки"):
                st.code(traceback.format_exc())
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Определяем тип ошибки и показываем соответствующее сообщение
            if "API" in error_type or "api" in error_msg.lower() or "key" in error_msg.lower():
                st.error(
                    f"❌ **Ошибка API:** Проблема с доступом к внешнему API.\n\n"
                    f"**Тип ошибки:** {error_type}\n"
                    f"**Сообщение:** {error_msg}\n\n"
                    f"Возможные причины:\n"
                    f"- Неверный или отсутствующий API ключ\n"
                    f"- Превышен лимит запросов\n"
                    f"- Проблемы с сетью\n\n"
                    f"Проверьте настройки API ключей в файле `.env`."
                )
            elif "connection" in error_msg.lower() or "network" in error_msg.lower() or "timeout" in error_msg.lower():
                st.error(
                    f"❌ **Ошибка сети:** Проблема с подключением к интернету.\n\n"
                    f"**Тип ошибки:** {error_type}\n"
                    f"**Сообщение:** {error_msg}\n\n"
                    f"Проверьте подключение к интернету и попробуйте снова."
                )
            elif "index" in error_msg.lower() or "faiss" in error_msg.lower() or "bm25" in error_msg.lower():
                st.error(
                    f"❌ **Ошибка индекса:** Проблема с загрузкой или использованием индекса поиска.\n\n"
                    f"**Тип ошибки:** {error_type}\n"
                    f"**Сообщение:** {error_msg}\n\n"
                    f"Возможно, индекс поврежден или не был создан. "
                    f"Попробуйте пересоздать индексы."
                )
            elif "embedding" in error_msg.lower() or "embed" in error_msg.lower():
                st.error(
                    f"❌ **Ошибка эмбеддингов:** Проблема с генерацией эмбеддингов.\n\n"
                    f"**Тип ошибки:** {error_type}\n"
                    f"**Сообщение:** {error_msg}\n\n"
                    f"Возможные причины:\n"
                    f"- Проблема с моделью эмбеддингов\n"
                    f"- Недостаточно памяти\n"
                    f"- Проблемы с загрузкой модели"
                )
            else:
                st.error(
                    f"❌ **Ошибка при обработке запроса:** Произошла непредвиденная ошибка.\n\n"
                    f"**Тип ошибки:** {error_type}\n"
                    f"**Сообщение:** {error_msg}\n\n"
                    f"Попробуйте:\n"
                    f"- Проверить корректность запроса\n"
                    f"- Обновить страницу\n"
                    f"- Проверить логи приложения"
                )
            
            if st.checkbox("Показать детали ошибки"):
                st.code(traceback.format_exc())
            
            st.stop()

        # Проверка наличия результата
        if not rag_answer:
            st.error("❌ **Ошибка:** Не удалось получить ответ от pipeline.")
            st.stop()
        
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
        try:
            if rag_answer.query_history and len(rag_answer.query_history) > 1:
                with st.expander("📝 История всех запросов"):
                    for i, q in enumerate(rag_answer.query_history, 1):
                        st.markdown(f"**Попытка {i}:** {q}")
        except Exception as e:
            st.warning(f"⚠️ Не удалось отобразить историю запросов: {str(e)}")

        try:
            st.subheader("Ответ")
            if rag_answer.answer:
                st.markdown(rag_answer.answer)
            else:
                st.warning("⚠️ Ответ пуст. Возможно, не удалось сгенерировать ответ на основе найденных источников.")
        except Exception as e:
            st.error(f"❌ **Ошибка отображения ответа:** {str(e)}")

        # Show quality metrics
        try:
            if rag_answer.contexts:
                mean_score = sum(ctx.score for ctx in rag_answer.contexts) / len(
                    rag_answer.contexts
                )
                max_score = max(ctx.score for ctx in rag_answer.contexts)
                st.caption(
                    f"Средний score: {mean_score:.3f} | Максимальный score: {max_score:.3f}"
                )
            else:
                st.warning("⚠️ Не найдено источников для отображения метрик.")
        except Exception as e:
            st.warning(f"⚠️ Не удалось вычислить метрики качества: {str(e)}")

        try:
            st.subheader("Использованные источники")
            if rag_answer.contexts:
                for ctx in rag_answer.contexts:
                    try:
                        with st.expander(f"{ctx.doc_id} | score={ctx.score:.3f} | {ctx.source}"):
                            if ctx.title:
                                st.markdown(f"**{ctx.title}**")
                            if ctx.url:
                                st.markdown(f"[Открыть источник]({ctx.url})")
                            text_preview = ctx.text[:1500] if ctx.text else "Текст недоступен"
                            st.markdown(text_preview + ("..." if len(ctx.text) > 1500 else ""))
                    except Exception as e:
                        st.warning(f"⚠️ Не удалось отобразить источник {ctx.doc_id}: {str(e)}")
            else:
                st.info("ℹ️ Источники не найдены. Попробуйте изменить запрос или увеличить количество документов (top-k).")
        except Exception as e:
            st.error(f"❌ **Ошибка отображения источников:** {str(e)}")


if __name__ == "__main__":
    main()
