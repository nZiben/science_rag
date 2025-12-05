# 🔬 Science RAG: Retrieval-Augmented QA for Scientific Articles

**Science RAG** — это учебно-прикладной проект Retrieval-Augmented Generation
для ответов на узкоспециализированные научные вопросы (ML, CS, математика,
физика, биология) на основе:

- локального корпуса статей arXiv;
- Web-RAG поверх интернета;
- гибридного поиска BM25 + dense с динамической настройкой α (DAT).

Проект ориентирован на:

- демонстрацию end-to-end архитектуры RAG;
- сравнение dense / sparse / hybrid retrieval;
- эксперимент с Dynamic Alpha Tuning (DAT) как meta-retriever над локальным
  научным корпусом.

---

## Архитектура

### Обзор

```text
arXiv API → Downloader → PDFs → PDF→Text → Cleaning+Chunking
                                   │
                                   v
                      Embeddings → FAISS (dense)
                      Tokens    → BM25 (sparse)

User Query
   │
   v
RAG Pipeline ──► Retriever (local / web / hybrid DAT)
   │
   v
Mistral Chat Completion → Answer + Sources
