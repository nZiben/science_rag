"""Web RAG retrieval with Tavily Search API."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import requests

from src.embeddings.embedder import Embedder
from src.local_rag.faiss_builder import FAISSIndex


TAVILY_URL = "https://api.tavily.com/search"


@dataclass
class WebDocument:
    """Single web document returned by Tavily."""

    doc_id: str
    url: str
    title: str
    content: str


def tavily_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "advanced",
) -> List[WebDocument]:
    """
    Query Tavily Search API and return structured documents.

    Parameters
    ----------
    query:
        User query.
    max_results:
        Maximum number of sources to retrieve.
    search_depth:
        "basic" or "advanced".

    Returns
    -------
    list of WebDocument
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY environment variable is missing.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
    }
    resp = requests.post(TAVILY_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    docs: List[WebDocument] = []
    for i, result in enumerate(data.get("results", [])):
        docs.append(
            WebDocument(
                doc_id=f"web_{i}",
                url=result.get("url", ""),
                title=result.get("title", ""),
                content=result.get("content", ""),
            )
        )
    return docs


def build_temp_web_index(
    web_docs: Sequence[WebDocument],
    embedder: Embedder,
) -> Tuple[FAISSIndex, Dict[str, WebDocument]]:
    """
    Build an in-memory FAISS index for a set of web documents.

    Parameters
    ----------
    web_docs:
        List of WebDocument objects.
    embedder:
        Embedder instance.

    Returns
    -------
    (FAISSIndex, mapping chunk_id -> WebDocument)
    """
    if not web_docs:
        raise ValueError("web_docs is empty.")

    texts = [doc.content for doc in web_docs]
    doc_ids = [doc.doc_id for doc in web_docs]
    embeddings = embedder.embed_documents(texts)

    # Build FAISS index in memory.
    faiss_index = FAISSIndex.build(embeddings, doc_ids)
    mapping = {doc.doc_id: doc for doc in web_docs}
    return faiss_index, mapping


def save_web_docs_to_jsonl(
    docs: Sequence[WebDocument],
    output_path: Path,
) -> None:
    """Persist web documents for debugging / audit."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f_out:
        for doc in docs:
            record = {
                "doc_id": doc.doc_id,
                "url": doc.url,
                "title": doc.title,
                "content": doc.content,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
