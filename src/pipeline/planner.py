"""Query replanning planner for improving RAG answers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal

from mistralai import Mistral  # type: ignore

if TYPE_CHECKING:
    from src.rag_pipeline import RetrievedContext


@dataclass
class QualityAssessment:
    """Result of quality assessment with recommended strategy."""

    is_sufficient: bool
    strategy: Literal["sufficient", "reformulate", "increase_top_k", "both"]
    reasoning: str = ""


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
        self,
        query: str,
        answer: str,
        contexts: List["RetrievedContext"],
    ) -> QualityAssessment:
        """Evaluate if the retrieved contexts and answer have sufficient quality using LLM.

        Args:
            query: The original user query.
            answer: The generated answer.
            contexts: List of retrieved contexts with scores.

        Returns:
            QualityAssessment with strategy recommendation.
        """
        if not contexts:
            return QualityAssessment(
                is_sufficient=False,
                strategy="reformulate",
                reasoning="No contexts retrieved",
            )

        mean_score = sum(ctx.score for ctx in contexts) / len(contexts)
        max_score = max((ctx.score for ctx in contexts), default=0.0)
        min_score = min((ctx.score for ctx in contexts), default=0.0)

        system_prompt = (
            "You are a quality assessment assistant. Your task is to evaluate "
            "whether a RAG answer and its retrieved contexts are sufficient to answer "
            "the user's question, and recommend the best improvement strategy.\n\n"
            "Consider:\n"
            "- Whether the answer addresses the question\n"
            "- Whether the answer is complete and informative\n"
            "- Whether the retrieved contexts are relevant (based on similarity scores)\n"
            "- Whether the answer quality could be improved with better retrieval\n\n"
            "STRATEGIES:\n"
            "- 'SUFFICIENT': The answer quality is good enough, no changes needed\n"
            "- 'REFORMULATE': The query is poorly formulated or doesn't match the domain. "
            "The answer is bad because wrong documents were retrieved. Need to reformulate the query.\n"
            "- 'INCREASE_TOP_K': The query is good but we need more context. "
            "The answer might be incomplete because we didn't retrieve enough relevant documents. "
            "Increasing top_k might help.\n"
            "- 'BOTH': Both problems exist - the query needs reformulation AND we need more context. "
            "Use this when the answer is both incomplete and based on wrong documents.\n\n"
            "Respond with ONLY one word: 'SUFFICIENT', 'REFORMULATE', 'INCREASE_TOP_K', or 'BOTH'."
        )

        context_previews = []
        for i, ctx in enumerate(contexts[:3], 1):  # Show top 3 contexts
            preview = ctx.text[:200] + "..." if len(ctx.text) > 200 else ctx.text
            context_previews.append(f"Context {i} (score: {ctx.score:.3f}): {preview}")

        content = (
            f"User query: {query}\n\n"
            f"Generated answer: {answer}\n\n"
            f"Retrieval statistics:\n"
            f"- Mean similarity score: {mean_score:.3f}\n"
            f"- Max similarity score: {max_score:.3f}\n"
            f"- Min similarity score: {min_score:.3f}\n"
            f"- Number of contexts: {len(contexts)}\n\n"
            f"Sample contexts:\n" + "\n\n".join(context_previews) + "\n\n"
            f"What strategy should be used? Respond with ONLY: 'SUFFICIENT', 'REFORMULATE', 'INCREASE_TOP_K', or 'BOTH'."
        )

        res = self.client.chat.complete(
            model=self.mistral_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            stream=False,
        )

        response = res.choices[0].message.content.strip().upper()
        
        if "SUFFICIENT" in response:
            return QualityAssessment(
                is_sufficient=True,
                strategy="sufficient",
                reasoning="Answer quality is sufficient",
            )
        elif "BOTH" in response:
            return QualityAssessment(
                is_sufficient=False,
                strategy="both",
                reasoning="Need both query reformulation and more context",
            )
        elif "INCREASE_TOP_K" in response or "INCREASE" in response:
            return QualityAssessment(
                is_sufficient=False,
                strategy="increase_top_k",
                reasoning="Need more context, increasing top_k",
            )
        else:  # REFORMULATE or default
            return QualityAssessment(
                is_sufficient=False,
                strategy="reformulate",
                reasoning="Query needs reformulation",
            )

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

