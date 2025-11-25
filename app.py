"""
Streamlit интерфейс для RAG системы с Mistral API и Tavily
"""
import streamlit as st
from rag_system import RAGSystem
from urllib.parse import urlparse

def get_domain(url):
    """Извлекает домен из URL"""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split('/')[0]
        # Убираем www. если есть
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except:
        return url

# Настройка страницы
st.set_page_config(
    page_title="RAG Система - Поиск и Ответы",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомный CSS для улучшения внешнего вида
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .answer-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .answer-box p {
        margin: 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Инициализация сессии
if "rag_system" not in st.session_state:
    try:
        st.session_state.rag_system = RAGSystem()
        st.session_state.initialized = True
    except Exception as e:
        st.session_state.initialized = False
        st.session_state.error = str(e)

# Инициализация чатов
if "chats" not in st.session_state:
    st.session_state.chats = {}
    st.session_state.chat_counter = 0

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

# Инициализация значения слайдера количества источников
if "max_results" not in st.session_state:
    st.session_state.max_results = 5

# Автоматически создаем новый чат при первом запуске, если его нет
if st.session_state.current_chat_id is None or st.session_state.current_chat_id not in st.session_state.chats:
    if st.session_state.chat_counter == 0:
        st.session_state.chat_counter = 1
    else:
        st.session_state.chat_counter += 1
    new_chat_id = f"chat_{st.session_state.chat_counter}"
    st.session_state.chats[new_chat_id] = {
        "title": "Новый чат" if st.session_state.chat_counter == 1 else f"Новый чат {st.session_state.chat_counter}",
        "messages": []
    }
    st.session_state.current_chat_id = new_chat_id

# Заголовок
st.markdown('<h1 class="main-header">🔍 RAG Система с Mistral и Tavily</h1>', unsafe_allow_html=True)
st.markdown("---")

# Боковая панель
with st.sidebar:
    st.header("💬 Чаты")
    
    # Кнопка создания нового чата
    if st.button("➕ Новый чат", use_container_width=True, type="primary", key="new_chat_button"):
        st.session_state.chat_counter += 1
        new_chat_id = f"chat_{st.session_state.chat_counter}"
        st.session_state.chats[new_chat_id] = {
            "title": f"Новый чат {st.session_state.chat_counter}",
            "messages": []
        }
        st.session_state.current_chat_id = new_chat_id
        st.rerun()
    
    st.markdown("---")
    
    # Список чатов
    if st.session_state.chats:
        for chat_id in sorted(st.session_state.chats.keys(), reverse=True):
            chat = st.session_state.chats[chat_id]
            # Определяем название чата (первый вопрос или "Новый чат")
            chat_title = chat.get("title", "Новый чат")
            if chat.get("messages"):
                first_question = chat["messages"][0].get("question", "")
                if first_question:
                    # Обновляем название чата, если оно еще не обновлено
                    if chat_title == "Новый чат" or not chat_title:
                        chat_title = first_question[:30] + ("..." if len(first_question) > 30 else "")
                        chat["title"] = chat_title
                        st.session_state.chats[chat_id] = chat
            
            # Отмечаем текущий чат
            if chat_id == st.session_state.current_chat_id:
                st.markdown(f"**📌 {chat_title}**")
            else:
                if st.button(f"💬 {chat_title}", key=f"chat_btn_{chat_id}", use_container_width=True):
                    st.session_state.current_chat_id = chat_id
                    st.rerun()
    else:
        st.info("Создайте новый чат для начала работы")

# Основной контент
if not st.session_state.initialized:
    st.error(f"❌ Ошибка инициализации: {st.session_state.error}")
    st.info("""
    Убедитесь, что:
    1. Создан файл `.env` с `MISTRAL_API_KEY` и `TAVILY_API_KEY`
    2. API ключи корректны и активны
    """)
else:
    # Получаем текущий чат (он уже создан автоматически при запуске)
    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    
    # Показываем историю диалога
    if current_chat["messages"]:
            for idx, msg in enumerate(current_chat["messages"]):
                # Вопрос пользователя
                with st.chat_message("user"):
                    st.write(msg['question'])
                
                # Ответ ассистента
                with st.chat_message("assistant"):
                    st.markdown(msg["answer"])
                    if msg.get("sources"):
                        with st.expander("🔗 Источники"):
                            for source in msg["sources"]:
                                if source.get('url'):
                                    st.markdown(f"- [{source['url']}]({source['url']})")
                                else:
                                    st.markdown(f"- {source.get('title', '')}")
    
    # Проверяем, идет ли обработка запроса
    is_processing = "processing_question" in st.session_state and st.session_state.processing_question
    
    if not is_processing:
        # Форма для вопроса (показывается только когда не идет обработка)
        with st.form("question_form", clear_on_submit=False):
            # Используем пустое значение, если поле было очищено
            question_value = st.session_state.get("question_input", "")
            question = st.text_area(
                "💬 Задайте ваш вопрос:",
                height=100,
                placeholder="Например: Что такое квантовая физика?",
                help="Введите вопрос, на который хотите получить ответ с использованием информации из интернета",
                key="question_input",
                value=question_value
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                submit_button = st.form_submit_button("🔍 Найти ответ", use_container_width=True)
            
            with col2:
                max_results = st.slider(
                    "Количество источников", 
                    3, 10, 
                    value=st.session_state.max_results,
                    help="Максимальное количество источников для поиска",
                    key="max_results_slider"
                )
                # Сохраняем значение в session_state при каждом рендере
                # Это гарантирует, что значение сохранится даже если форма очистится
                if "max_results_slider" in st.session_state:
                    st.session_state.max_results = st.session_state.max_results_slider
        
        # Обработка вопроса (вне формы)
        if submit_button and question:
            # Сохраняем вопрос для отображения во время обработки
            st.session_state.processing_question = question
            st.session_state.processing_max_results = st.session_state.get("max_results", 5)
            st.rerun()
    
    # Отображение вопроса и обработка (вместо формы)
    if is_processing:
        question = st.session_state.processing_question
        max_results = st.session_state.processing_max_results
        
        # Показываем вопрос пользователя
        with st.chat_message("user"):
            st.write(question)
        
        # Поиск источников
        with st.spinner("🔍 Поиск информации в интернете..."):
            search_results = st.session_state.rag_system.search_internet(question, max_results=max_results)
        
        # Отображение результатов поиска
        if not search_results:
            st.text("❌ Источники не найдены")
            # Сохраняем сообщение об ошибке
            message_data = {
                "question": question,
                "answer": "❌ К сожалению, не удалось найти информацию по вашему запросу.",
                "sources": []
            }
            current_chat["messages"].append(message_data)
            if len(current_chat["messages"]) == 1:
                current_chat["title"] = question[:30] + ("..." if len(question) > 30 else "")
            st.session_state.chats[st.session_state.current_chat_id] = current_chat
            
            # Очищаем поле ввода и флаг обработки после сохранения ответа
            if "question_input" in st.session_state:
                del st.session_state.question_input
            if "processing_question" in st.session_state:
                del st.session_state.processing_question
            # Явно устанавливаем пустое значение для поля ввода
            st.session_state.question_input = ""
            
            st.rerun()
        else:
            # Формируем сообщение о найденных источниках
            sources_count = len(search_results)
            st.text(f"✅ Найдено {sources_count} источников")
            
            # Формируем контекст и генерируем ответ с учетом истории диалога
            context = st.session_state.rag_system.format_context(search_results)
            
            # Получаем историю предыдущих сообщений (кроме текущего вопроса)
            chat_history = current_chat["messages"]
            
            # Генерация ответа с loader'ом
            with st.spinner("🤖 Генерация ответа с помощью Mistral AI..."):
                answer = st.session_state.rag_system.generate_answer(
                    question, 
                    context, 
                    chat_history=chat_history
                )
            st.text("✅ Ответ сгенерирован")
            
            # Сохраняем в текущий чат
            sources_list = [{"title": r["title"], "url": r["url"]} for r in search_results if r.get("url")]
            message_data = {
                "question": question,
                "answer": answer,
                "sources": sources_list
            }
            current_chat["messages"].append(message_data)
            
            # Обновляем название чата, если это первый вопрос
            if len(current_chat["messages"]) == 1:
                current_chat["title"] = question[:30] + ("..." if len(question) > 30 else "")
            
            # Обновляем состояние чата
            st.session_state.chats[st.session_state.current_chat_id] = current_chat
            
            # Очищаем поле ввода и флаг обработки после сохранения ответа
            if "question_input" in st.session_state:
                del st.session_state.question_input
            if "processing_question" in st.session_state:
                del st.session_state.processing_question
            # Явно устанавливаем пустое значение для поля ввода
            st.session_state.question_input = ""
            
            # Обновляем интерфейс
            st.rerun()
    
    elif submit_button and not question:
        st.warning("⚠️ Пожалуйста, введите вопрос перед отправкой.")

# Футер
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "RAG Система с Mistral AI и Tavily | Powered by Streamlit"
    "</div>",
    unsafe_allow_html=True
)

