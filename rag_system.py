"""
RAG система с использованием Mistral API и Tavily для поиска информации в интернете
"""
import os
from dotenv import load_dotenv
from mistralai import Mistral
from tavily import TavilyClient

# Загружаем переменные окружения
load_dotenv()


class RAGSystem:
    """RAG система для ответов на вопросы с использованием информации из интернета"""
    
    def __init__(self):
        """Инициализация RAG системы с API ключами"""
        # Получаем API ключи из переменных окружения
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        if not mistral_api_key:
            raise ValueError("MISTRAL_API_KEY не найден в переменных окружения")
        if not tavily_api_key:
            raise ValueError("TAVILY_API_KEY не найден в переменных окружения")
        
        # Инициализируем клиенты
        self.mistral_client = Mistral(api_key=mistral_api_key)
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        
        # Модель Mistral для генерации ответов
        self.model = "mistral-large-latest"
    
    def search_internet(self, query: str, max_results: int = 5) -> list:
        """
        Поиск информации в интернете с помощью Tavily
        
        Args:
            query: Поисковый запрос
            max_results: Максимальное количество результатов
            
        Returns:
            Список результатов поиска
        """
        try:
            response = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=False,
                include_raw_content=False
            )
            
            results = []
            
            # Добавляем результаты поиска
            if response.get("results"):
                for result in response["results"]:
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content": result.get("content", "")
                    })
            
            return results
        except Exception as e:
            print(f"Ошибка при поиске в интернете: {e}")
            return []
    
    def format_context(self, search_results: list) -> str:
        """
        Форматирование результатов поиска в контекст для промпта
        
        Args:
            search_results: Список результатов поиска
            
        Returns:
            Отформатированный контекст
        """
        if not search_results:
            return "Информация не найдена."
        
        context_parts = []
        
        # Добавляем результаты поиска
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"[Источник {i}]: {result['title']}")
            if result.get("url"):
                context_parts.append(f"URL: {result['url']}")
            context_parts.append(f"Содержание: {result['content']}\n")
        
        return "\n".join(context_parts)
    
    def generate_answer(self, question: str, context: str, chat_history: list = None) -> str:
        """
        Генерация ответа с использованием Mistral API
        
        Args:
            question: Вопрос пользователя
            context: Контекст из результатов поиска
            chat_history: История предыдущих сообщений в формате [{"question": "...", "answer": "..."}, ...]
            
        Returns:
            Сгенерированный ответ
        """
        system_prompt = """Ты полезный AI-ассистент, который отвечает на вопросы пользователя, 
используя предоставленную информацию из интернета. Отвечай точно, информативно и структурированно. 
Если информация недостаточна, укажи на это. Отвечай на русском языке.
Учитывай контекст предыдущих сообщений в диалоге для более точных ответов."""
        
        # Формируем историю диалога
        messages = [{"role": "system", "content": system_prompt}]
        
        # Добавляем историю диалога, если она есть
        if chat_history:
            for msg in chat_history:
                messages.append({"role": "user", "content": msg["question"]})
                messages.append({"role": "assistant", "content": msg["answer"]})
        
        # Добавляем текущий вопрос с контекстом из интернета
        user_prompt = f"""Используй следующую информацию из интернета для ответа на вопрос пользователя:

{context}

Вопрос пользователя: {question}

Ответ:"""
        
        messages.append({"role": "user", "content": user_prompt})
        
        try:
            response = self.mistral_client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Ошибка при генерации ответа: {e}"
    
    def answer_question(self, question: str) -> dict:
        """
        Основной метод для ответа на вопрос пользователя
        
        Args:
            question: Вопрос пользователя
            
        Returns:
            Словарь с ответом и источниками
        """
        print(f"🔍 Поиск информации по запросу: {question}")
        search_results = self.search_internet(question)
        
        if not search_results:
            return {
                "answer": "К сожалению, не удалось найти информацию по вашему запросу.",
                "sources": []
            }
        
        print(f"✅ Найдено {len(search_results)} источников")
        
        # Выводим список источников
        print("\n📚 Найденные источники:")
        for i, result in enumerate(search_results, 1):
            if result.get("url"):
                print(f"  {i}. {result['title']}")
                print(f"     {result['url']}")
            else:
                print(f"  {i}. {result['title']}")
        print()
        
        context = self.format_context(search_results)
        
        print("🤖 Генерация ответа с помощью Mistral...")
        answer = self.generate_answer(question, context)
        
        sources = [{"title": r["title"], "url": r["url"]} for r in search_results if r.get("url")]
        
        return {
            "answer": answer,
            "sources": sources
        }


def main():
    """Основная функция для интерактивного использования"""
    try:
        rag = RAGSystem()
        print("=" * 60)
        print("RAG система готова к работе!")
        print("Введите 'выход' или 'exit' для завершения")
        print("=" * 60)
        
        while True:
            question = input("\n💬 Ваш вопрос: ").strip()
            
            if question.lower() in ["выход", "exit", "quit"]:
                print("До свидания!")
                break
            
            if not question:
                print("Пожалуйста, введите вопрос.")
                continue
            
            result = rag.answer_question(question)
            
            print("\n" + "=" * 60)
            print("📝 ОТВЕТ:")
            print("=" * 60)
            print(result["answer"])
            
            if result["sources"]:
                print("\n" + "=" * 60)
                print("📚 ИСТОЧНИКИ:")
                print("=" * 60)
                for i, source in enumerate(result["sources"], 1):
                    print(f"{i}. {source['title']}")
                    print(f"   {source['url']}")
            
            print("\n" + "=" * 60)
    
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("\nУбедитесь, что:")
        print("1. Создан файл .env с MISTRAL_API_KEY и TAVILY_API_KEY")
        print("2. API ключи корректны и активны")


if __name__ == "__main__":
    main()

