# RAG система с Mistral API и Tavily

RAG (Retrieval-Augmented Generation) система для ответов на вопросы пользователя с использованием информации из интернета.

## Возможности

- 🔍 Поиск информации в интернете через Tavily API
- 🤖 Генерация ответов с помощью Mistral API
- 🎛️ Выбор модели Mistral (Large, Medium, Small, Tiny)
- 📚 Предоставление источников информации
- 📄 Экспорт чатов в PDF формат
- 💬 Интерактивный режим работы (консольный и веб-интерфейс)
- 🌐 Современный Streamlit веб-интерфейс

## Установка

1. **Создайте виртуальное окружение:**
   ```bash
   python -m venv venv
   ```

2. **Активируйте виртуальное окружение:**
   
   Windows:
   ```bash
   venv\Scripts\activate
   ```
   
   Linux/Mac:
   ```bash
   source venv/bin/activate
   ```

3. **Установите зависимости:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Настройте переменные окружения:**
   
   Скопируйте `.env.example` в `.env`:
   ```bash
   copy .env.example .env
   ```
   
   Или вручную создайте файл `.env` и добавьте ваши API ключи:
   ```
   MISTRAL_API_KEY=your_mistral_api_key_here
   TAVILY_API_KEY=tvly-dev-your_key_here
   ```

## Использование

### 🌐 Веб-интерфейс (Streamlit) - Рекомендуется

Запустите Streamlit приложение:
```bash
streamlit run app.py
```

Откройте браузер по адресу, который появится в консоли (обычно `http://localhost:8501`).

**Преимущества веб-интерфейса:**
- 🎨 Современный и удобный интерфейс
- 📊 История вопросов и ответов в виде отдельных чатов
- 🔗 Интерактивные ссылки на источники
- ⚙️ Настройка количества источников через слайдер
- 🎛️ Выбор модели Mistral (Large, Medium, Small, Tiny)
- 📄 Экспорт чатов в PDF с сохранением форматирования

### 💻 Консольный режим

Запустите скрипт:
```bash
python rag_system.py
```

Введите ваш вопрос и получите ответ с источниками информации.

### Использование как библиотеки

```python
from rag_system import RAGSystem

# Использование модели по умолчанию (mistral-large-latest)
rag = RAGSystem()

# Или выбор конкретной модели
rag = RAGSystem(model="mistral-medium-latest")

result = rag.answer_question("Что такое квантовая физика?")

print(result["answer"])
for source in result["sources"]:
    print(f"- {source['title']}: {source['url']}")
```

## Требования

- Python 3.8+
- Mistral API ключ (получите на https://console.mistral.ai/)
- Tavily API ключ (ваш ключ начинается с `tvly-dev-`)

## Структура проекта

```
science_rag/
├── rag_system.py      # Основной код RAG системы
├── app.py             # Streamlit веб-интерфейс
├── requirements.txt   # Зависимости проекта
├── .env.example      # Пример файла с переменными окружения
├── .env              # Ваши API ключи (создайте сами)
└── README.md         # Документация
```

## Примечания

- Убедитесь, что ваш Tavily API ключ активен и начинается с `tvly-dev-`
- Для работы Mistral API необходим активный ключ от Mistral AI
- Система автоматически ищет информацию в интернете и генерирует ответы на русском языке

