"""
Streamlit интерфейс для RAG системы с Mistral API и Tavily
"""
import streamlit as st
from rag_system import RAGSystem
from urllib.parse import urlparse
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO
from datetime import datetime
import re
import os

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

def parse_markdown_to_elements(text, h1_style, h2_style, h3_style, normal_style, list_item_style):
    """Преобразует markdown в список элементов для PDF"""
    if not text:
        return []
    
    elements = []
    lines = text.split('\n')
    in_list = False
    list_type = None
    list_items = []
    
    def format_text(content):
        """Обрабатывает форматирование в тексте"""
        # Экранируем HTML
        content = content.replace('&', '&amp;')
        content = content.replace('<', '&lt;')
        content = content.replace('>', '&gt;')
        # Жирный текст
        content = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', content)
        content = re.sub(r'__(.+?)__', r'<b>\1</b>', content)
        # Курсив
        content = re.sub(r'(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', content)
        # Ссылки
        content = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2" color="blue"><u>\1</u></a>', content)
        # Код
        content = re.sub(r'`([^`]+)`', r'<font face="Courier">\1</font>', content)
        return content
    
    def flush_list():
        """Выводит накопленные элементы списка"""
        nonlocal list_items, in_list, list_type
        if list_items:
            for idx, item in enumerate(list_items, 1):
                formatted = format_text(item)
                if list_type == 'ol':
                    style = ParagraphStyle(
                        'ListNumbered',
                        parent=list_item_style,
                        bulletText=f'{idx}.'
                    )
                else:
                    style = ParagraphStyle(
                        'ListBullet',
                        parent=list_item_style,
                        bulletText='•'
                    )
                elements.append(Paragraph(formatted, style))
            list_items = []
            in_list = False
    
    for line in lines:
        line = line.rstrip()
        
        # Заголовки
        if line.startswith('###'):
            flush_list()
            content = format_text(line[3:].strip())
            elements.append(Paragraph(content, h3_style))
        elif line.startswith('##'):
            flush_list()
            content = format_text(line[2:].strip())
            elements.append(Paragraph(content, h2_style))
        elif line.startswith('#'):
            flush_list()
            content = format_text(line[1:].strip())
            elements.append(Paragraph(content, h1_style))
        # Нумерованные списки
        elif re.match(r'^\d+\.\s+', line):
            if not in_list or list_type != 'ol':
                flush_list()
                in_list = True
                list_type = 'ol'
            content = re.sub(r'^\d+\.\s+', '', line)
            list_items.append(content)
        # Маркированные списки
        elif re.match(r'^[-*]\s+', line):
            if not in_list or list_type != 'ul':
                flush_list()
                in_list = True
                list_type = 'ul'
            content = re.sub(r'^[-*]\s+', '', line)
            list_items.append(content)
        # Пустая строка
        elif not line.strip():
            flush_list()
            elements.append(Spacer(1, 0.1*inch))
        # Обычный текст
        else:
            flush_list()
            content = format_text(line)
            if content.strip():
                elements.append(Paragraph(content, normal_style))
    
    flush_list()
    return elements

def export_chat_to_pdf(chat_data):
    """Экспортирует чат в PDF формат"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, 
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=72)
    
    # Регистрируем шрифт с поддержкой кириллицы
    # Используем стандартный шрифт для кириллицы
    try:
        # Пробуем использовать стандартные шрифты Windows с поддержкой кириллицы
        font_paths = [
            r'C:\Windows\Fonts\arial.ttf',
            r'C:\Windows\Fonts\calibri.ttf',
            r'C:\Windows\Fonts\tahoma.ttf',
        ]
        
        font_registered = False
        font_name = 'CyrillicFont'
        font_bold_name = 'CyrillicFontBold'
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    pdfmetrics.registerFont(TTFont(font_name, font_path))
                    pdfmetrics.registerFont(TTFont(font_bold_name, font_path))
                    font_registered = True
                    break
                except:
                    continue
        
        # Если не удалось зарегистрировать шрифт из системы, используем встроенные шрифты
        if not font_registered:
            # Используем встроенные шрифты reportlab с поддержкой Unicode
            font_name = 'Helvetica'
            font_bold_name = 'Helvetica-Bold'
    except:
        # Fallback на стандартные шрифты
        font_name = 'Helvetica'
        font_bold_name = 'Helvetica-Bold'
    
    # Стили
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor='#1f77b4',
        spaceAfter=30,
        alignment=TA_LEFT,
        fontName=font_name
    )
    
    question_style = ParagraphStyle(
        'QuestionStyle',
        parent=styles['Normal'],
        fontSize=12,
        textColor='#333333',
        spaceAfter=12,
        leftIndent=20,
        fontName=font_bold_name
    )
    
    answer_style = ParagraphStyle(
        'AnswerStyle',
        parent=styles['Normal'],
        fontSize=11,
        textColor='#000000',
        spaceAfter=20,
        leftIndent=20,
        alignment=TA_JUSTIFY,
        fontName=font_name
    )
    
    source_style = ParagraphStyle(
        'SourceStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor='#666666',
        spaceAfter=10,
        leftIndent=40,
        fontName=font_name
    )
    
    # Создаем нормальный стиль с кириллическим шрифтом
    normal_style = ParagraphStyle(
        'NormalCyrillic',
        parent=styles['Normal'],
        fontSize=11,
        fontName=font_name
    )
    
    # Стили для заголовков
    h1_style = ParagraphStyle(
        'H1Style',
        parent=styles['Heading1'],
        fontSize=14,
        fontName=font_bold_name,
        textColor='#1f77b4',
        spaceAfter=12,
        spaceBefore=12
    )
    
    h2_style = ParagraphStyle(
        'H2Style',
        parent=styles['Heading2'],
        fontSize=13,
        fontName=font_bold_name,
        textColor='#2c3e50',
        spaceAfter=10,
        spaceBefore=10
    )
    
    h3_style = ParagraphStyle(
        'H3Style',
        parent=styles['Heading3'],
        fontSize=12,
        fontName=font_bold_name,
        textColor='#34495e',
        spaceAfter=8,
        spaceBefore=8
    )
    
    # Стиль для списков
    list_item_style = ParagraphStyle(
        'ListItemStyle',
        parent=styles['Normal'],
        fontSize=11,
        fontName=font_name,
        leftIndent=30,
        bulletIndent=15,
        spaceAfter=6
    )
    
    # Содержимое PDF
    story = []
    
    # Заголовок
    chat_title = chat_data.get("title", "Экспорт чата")
    story.append(Paragraph(f"Чат: {chat_title}", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Дата экспорта
    export_date = datetime.now().strftime("%d.%m.%Y %H:%M")
    story.append(Paragraph(f"Дата экспорта: {export_date}", normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Сообщения чата
    messages = chat_data.get("messages", [])
    if not messages:
        story.append(Paragraph("В этом чате пока нет сообщений.", normal_style))
    else:
        for idx, msg in enumerate(messages, 1):
            # Вопрос пользователя
            question = msg.get("question", "")
            story.append(Paragraph(f"<b>Вопрос {idx}:</b>", question_style))
            question_elements = parse_markdown_to_elements(
                question, h1_style, h2_style, h3_style, answer_style, list_item_style
            )
            if question_elements:
                story.extend(question_elements)
            else:
                story.append(Paragraph(question, answer_style))
            story.append(Spacer(1, 0.15*inch))
            
            # Ответ ассистента
            answer = msg.get("answer", "")
            story.append(Paragraph("<b>Ответ:</b>", question_style))
            # Парсим markdown и добавляем элементы
            answer_elements = parse_markdown_to_elements(
                answer, h1_style, h2_style, h3_style, answer_style, list_item_style
            )
            story.extend(answer_elements)
            story.append(Spacer(1, 0.15*inch))
            
            # Источники
            sources = msg.get("sources", [])
            if sources:
                story.append(Paragraph("<b>Источники:</b>", question_style))
                story.append(Spacer(1, 0.05*inch))
                for source_idx, source in enumerate(sources, 1):
                    source_title = source.get('title', 'Без названия')
                    source_url = source.get('url', '')
                    if source_url:
                        source_text = f"{source_idx}. <a href=\"{source_url}\" color=\"blue\"><u>{source_title}</u></a> - {source_url}"
                    else:
                        source_text = f"{source_idx}. {source_title}"
                    story.append(Paragraph(source_text, source_style))
                story.append(Spacer(1, 0.1*inch))
            
            # Разделитель между сообщениями
            if idx < len(messages):
                story.append(Spacer(1, 0.25*inch))
                # Простая линия-разделитель
                story.append(Paragraph("─" * 80, normal_style))
                story.append(Spacer(1, 0.25*inch))
    
    # Генерация PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

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
    .chat-item {
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border: 2px solid #e0e0e0;
        background-color: #f8f9fa;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    .chat-item:hover {
        background-color: #e9ecef;
        border-color: #1f77b4;
        transform: translateX(2px);
    }
    .chat-item-active {
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        background: linear-gradient(135deg, #1f77b4 0%, #2c5aa0 100%);
        color: white;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(31, 119, 180, 0.3);
    }
    .chat-item-active:hover {
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.4);
    }
    .chat-title {
        font-size: 0.95rem;
        margin: 0;
        word-wrap: break-word;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    .chat-icon {
        margin-right: 0.5rem;
        font-size: 1.1rem;
    }
    /* Стилизация кнопок чатов */
    .stButton button[kind="secondary"] {
        background-color: #f8f9fa !important;
        color: #333 !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem 1rem !important;
        font-weight: normal !important;
        text-align: left !important;
        transition: all 0.2s ease !important;
        box-shadow: none !important;
        width: 100% !important;
    }
    .stButton button[kind="secondary"]:hover {
        background-color: #e9ecef !important;
        border-color: #1f77b4 !important;
        transform: translateX(2px) !important;
    }
    /* Альтернативный селектор для кнопок чатов */
    button[data-testid*="chat_btn"] {
        background-color: #f8f9fa !important;
        color: #333 !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem 1rem !important;
        font-weight: normal !important;
        text-align: left !important;
    }
    button[data-testid*="chat_btn"]:hover {
        background-color: #e9ecef !important;
        border-color: #1f77b4 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Инициализация сессии
# Инициализация выбранной модели
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "mistral-large-latest"

if "rag_system" not in st.session_state:
    try:
        st.session_state.rag_system = RAGSystem(model=st.session_state.selected_model)
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
    st.header("⚙️ Настройки")
    
    # Выбор модели Mistral
    mistral_models = {
        "mistral-large-latest": "Large",
        "mistral-medium-latest": "Medium",
        "mistral-small-latest": "Small",
        "mistral-tiny": "Tiny"
    }
    
    selected_model = st.selectbox(
        "🤖 Модель Mistral",
        options=list(mistral_models.keys()),
        format_func=lambda x: mistral_models[x],
        index=list(mistral_models.keys()).index(st.session_state.selected_model) if st.session_state.selected_model in mistral_models else 0,
        help="Выберите версию модели Mistral для генерации ответов"
    )
    
    # Если модель изменилась, переинициализируем RAG систему
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        try:
            st.session_state.rag_system = RAGSystem(model=selected_model)
            st.session_state.initialized = True
            if "error" in st.session_state:
                del st.session_state.error
        except Exception as e:
            st.session_state.initialized = False
            st.session_state.error = str(e)
    
    st.markdown("---")
    
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
            
            # Отмечаем текущий чат с красивым оформлением
            if chat_id == st.session_state.current_chat_id:
                # Активный чат - выделенный блок
                st.markdown(
                    f'<div class="chat-item-active">'
                    f'<span class="chat-icon">📌</span>'
                    f'<span class="chat-title">{chat_title}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                # Неактивные чаты - карточки с кнопками
                chat_button_key = f"chat_btn_{chat_id}"
                if st.button(f"💬 {chat_title}", key=chat_button_key, use_container_width=True, type="secondary"):
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
    
    # Кнопка экспорта в PDF (в окне чата)
    if current_chat.get("messages"):
        col1, col2 = st.columns([1, 5])
        with col1:
            try:
                pdf_buffer = export_chat_to_pdf(current_chat)
                chat_title = current_chat.get("title", "chat").replace(" ", "_")
                # Очищаем название от недопустимых символов для имени файла
                chat_title = re.sub(r'[<>:"/\\|?*]', '_', chat_title)
                filename = f"{chat_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                st.download_button(
                    label="📄 Экспорт в PDF",
                    data=pdf_buffer,
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_pdf_button"
                )
            except Exception as e:
                st.error(f"Ошибка при создании PDF: {e}")
        with col2:
            st.write("")  # Пустое пространство для выравнивания
    
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

