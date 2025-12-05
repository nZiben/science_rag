"""Embedding wrapper: Mistral or Sentence-Transformers."""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import List, Sequence

import numpy as np

try:
    from mistralai import Mistral
except ImportError:  # pragma: no cover
    Mistral = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

logger = logging.getLogger(__name__)


class EmbeddingBackend(str, Enum):
    """Supported embedding backends."""

    MISTRAL = "mistral"
    SENTENCE_TRANSFORMERS = "sentence-transformers"


class Embedder:
    """
    Simple embedding interface.

    Supports:
    - Mistral embeddings API (model `mistral-embed`).
    - Sentence-Transformers local models.
    """

    def __init__(
        self,
        backend: EmbeddingBackend = EmbeddingBackend.MISTRAL,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.backend = backend
        self.model_name = model_name

        if backend == EmbeddingBackend.MISTRAL:
            if Mistral is None:
                raise ImportError("mistralai is not installed.")
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise RuntimeError("MISTRAL_API_KEY environment variable is missing.")
            self._client = Mistral(api_key=api_key)
            self._st_model = None
        else:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers is not installed.")
            self._st_model = SentenceTransformer(model_name)
            self._client = None

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """
        Embed a batch of documents.

        Returns
        -------
        np.ndarray
            Array of shape (n_documents, dim).
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        if self.backend == EmbeddingBackend.MISTRAL:
            assert self._client is not None
            with self._client as client:
                res = client.embeddings.create(
                    model=self.model_name,
                    inputs=list(texts),
                )
            vectors = [item.embedding for item in res.data]
            return np.asarray(vectors, dtype=np.float32)

        assert self._st_model is not None
        vectors = self._st_model.encode(list(texts), convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query as 1D vector."""
        emb = self.embed_documents([text])
        if emb.size == 0:
            return emb
        return emb[0]
