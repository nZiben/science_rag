"""Query replanning planner for improving RAG answers."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List

from mistralai import Mistral  # type: ignore

if TYPE_CHECKING:
    from src.rag_pipeline import RetrievedContext


class QueryPlanner:
    """Planner that reformulates queries when answer quality is insufficient."""

    def __init__(self, mistral_model: str = "mistral-small-latest") -> None:
        """Initialize the query planner.

        Args:
            mistral_model: Mistral model to use for query reformulation.
        """
        self.mistral_model = mistral_model
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

    def evaluate_quality(
        self, contexts: List["RetrievedContext"], threshold: float
    ) -> bool:
        """Evaluate if the retrieved contexts have sufficient quality.

        Args:
            contexts: List of retrieved contexts with scores.
            threshold: Quality threshold (mean score must be >= threshold).

        Returns:
            True if quality is sufficient, False otherwise.
        """
        if not contexts:
            return False

        mean_score = sum(ctx.score for ctx in contexts) / len(contexts)
        return mean_score >= threshold

    def reformulate_query(
        self,
        original_query: str,
        previous_answer: str,
        contexts: List["RetrievedContext"],
    ) -> str:
        """Reformulate the query to improve retrieval quality.

        Args:
            original_query: The original user query.
            previous_answer: The answer generated from previous retrieval.
            contexts: The contexts retrieved in the previous attempt.

        Returns:
            A reformulated query string.
        """
        mean_score = (
            sum(ctx.score for ctx in contexts) / len(contexts) if contexts else 0.0
        )
        max_score = max((ctx.score for ctx in contexts), default=0.0)

        system_prompt = (
            "You are a query reformulation assistant. Your task is to reformulate "
            "a search query to improve retrieval quality. The original query did not "
            "retrieve sufficiently relevant documents (low similarity scores).\n\n"
            "IMPORTANT RULES:\n"
            "- Return ONLY the reformulated query, nothing else\n"
            "- Do NOT add questions, clarifications, or additional text\n"
            "- Do NOT ask for more information or details\n"
            "- Simply reformulate the original query using different words, "
            "more specific terminology, or clearer phrasing\n"
            "- Keep the same intent and meaning as the original query\n"
            "- Use the SAME LANGUAGE as the original query\n"
            "- The output should be a single, clear search query"
        )

        context_info = (
            f"Mean relevance score: {mean_score:.3f}, "
            f"Max relevance score: {max_score:.3f}"
        )

        content = (
            f"Original query: {original_query}\n\n"
            f"Previous answer (may be incomplete): {previous_answer}\n\n"
            f"Retrieval quality metrics: {context_info}\n\n"
            f"Reformulate the original query to improve search results. "
            f"Return ONLY the reformulated query, no additional text or questions."
        )

        res = self.client.chat.complete(
            model=self.mistral_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            stream=False,
        )

        reformulated = res.choices[0].message.content.strip()
        # Remove quotes if the model wrapped the query in them
        if reformulated.startswith('"') and reformulated.endswith('"'):
            reformulated = reformulated[1:-1]
        elif reformulated.startswith("'") and reformulated.endswith("'"):
            reformulated = reformulated[1:-1]

        return reformulated

