# **Science RAG**

Retrieval-Augmented Generation система для научных вопросов на основе корпуса arXiv и Web-поиска.

Проект объединяет три режима работы RAG:

* **Локальный RAG** — поиск по собственному корпусу статей arXiv;
* **Web-RAG** — поиск по интернету через Tavily API;
* **Гибридный RAG с DAT** — объединение BM25 + dense с динамической настройкой веса α.

Система отвечает на вопросы в областях  Computation and Language (cs.CL), Artificial Intelligence (cs.AI), Machine Learning (cs.LG).

---

# 1. **Структура репозитория**

```
science-rag/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   │   └── arxiv_pdfs/
│   ├── processed/
│   │   └── texts/
│   ├── chunks/
│   │   └── local/
│   ├── faiss/
│   │   ├── local_dense.index
│   │   ├── local_dense_meta.json
│   │   ├── web_dense.index
│   │   └── web_dense_meta.json
│   └── validation/
│       └── qa_pairs.jsonl
├── docs/
│   ├── checkpoint_report.md
│   └── roadmap.md
├── src/
│   ├── data_ingestion/
│   │   ├── arxiv_downloader.py
│   │   └── pdf_to_text.py
│   ├── embeddings/
│   │   └── embedder.py
│   ├── local_rag/
│   │   ├── chunker.py
│   │   ├── faiss_builder.py
│   │   └── bm25_index.py
│   ├── web_rag/
│   │   └── web_retriever.py
│   ├── hybrid_retrieval/
│   │   └── hybrid_retriever_with_DAT.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── rag_pipeline.py
├── README.md
├── requirements.txt
└── pyproject.toml  (опционально)
```

---

# 2. **Архитектура системы**

### **Пайплайн данных**

```
arXiv API 
    → Downloader → PDFs → PDF → Text → Cleaning + Chunking
                                             │
                                             ▼
                               Embeddings → FAISS (dense)
                                 Tokens → BM25 (sparse)
```

### **Пайплайн ответа**

```
User Query
     ▼
RAG Pipeline (local / web / hybrid DAT)
     ▼
Retriever → Top-k Documents
     ▼
Mistral Chat → Final Answer + Sources
```

### Web-RAG:

```
Query → Tavily Search → Content Cleaning → Chunking → Embeddings → FAISS → RAG
```

---

# 3. **Сбор данных**

Скрипты находятся в `src/data_ingestion/`.

### 1) Скачать PDF с arXiv

```
python -m src.data_ingestion.arxiv_downloader \
  --query "cs.LG OR cs.CL" \
  --max_results 200 \
  --output_dir data/raw/arxiv_pdfs
```

### 2) Конвертация PDF → текст

```
python -m src.data_ingestion.pdf_to_text \
  --pdf_dir data/raw/arxiv_pdfs \
  --output_dir data/processed/texts
```

---

# 4. **Подготовка данных для RAG**

### Чанкование

```
python -m src.local_rag.chunker \
  --texts_dir data/processed/texts \
  --output_path data/chunks/local/chunks.jsonl \
  --max_chars 1200 \
  --overlap 200
```

### Embeddings + FAISS

```
python -m src.embeddings.build_embeddings
python -m src.local_rag.faiss_builder
```

### BM25-индекс

```
python -m src.local_rag.bm25_index
```

Все итоговые артефакты находятся в:

* `data/chunks/local/chunks.jsonl` — **основной корпус RAG**
* `data/faiss/local_dense.index` — FAISS
* `data/chunks/local/bm25_index.json` — BM25

---

# 5. **Retrievers**

Проект поддерживает три режима:

### 1) **Dense RAG (FAISS)**

Использует Sentence-Transformers / Mistral Embeddings.

### 2) **Sparse RAG (BM25)**

Лучше работает на терминологических запросах.

### 3) **Hybrid RAG с DAT**

Dynamic Alpha Tuning вычисляет вес α(q):

* если dense лучше → α↑
* если BM25 лучше → α↓

Это позволяет адаптировать retrieval под каждый запрос.

---

# 6. **Web-RAG**

Используется Tavily API:

```
query → Tavily → content → chunk → embed → FAISS → retrieve → answer
```

Подходит для вопросов, не покрытых локальным arXiv-корпусом.

---

# 7. **Валидация retrieval**

Файл:
`data/validation/qa_pairs.jsonl`

Формат:

```json
{"id": "q1", "question": "...", "gold_chunk_ids": ["doc1_0003_ab12"]}
```

Метрики в `src/evaluation/metrics.py`:

* Recall@k
* Precision@k

Используются для сравнения BM25 / dense / hybrid.

---

# 8. **Запуск Streamlit-приложения**

```
export MISTRAL_API_KEY="..."
export TAVILY_API_KEY="..."

streamlit run app/streamlit_app.py
```

В приложении доступны режимы:

* Local RAG
* Hybrid RAG (DAT)
* Web-RAG

---

# 9. **Примеры данных**

Сэмпл корпуса чанков:
`data/chunks/local/chunks.jsonl`

Пример строки:

```json
{
  "chunk_id": "2301.01234_0001_af12cd98",
  "source_id": "2301.01234",
  "text": "In this paper we investigate...",
  "metadata": {"source_id": "2301.01234"}
}
```

---

# 11. **Установка**

```
git clone https://github.com/.../science-rag.git
cd science-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
