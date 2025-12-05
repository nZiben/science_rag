"""Utilities for downloading scientific papers from arXiv."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import arxiv

logger = logging.getLogger(__name__)


def download_arxiv_papers(
    query: str,
    max_results: int,
    output_dir: Path,
    categories: Optional[Iterable[str]] = None,
) -> List[Path]:
    """
    Download arXiv PDFs by query and optional categories.

    Parameters
    ----------
    query:
        Free-text or fielded arXiv query.
    max_results:
        Maximum number of papers to download.
    output_dir:
        Directory where PDFs will be saved.
    categories:
        Optional iterable of category names, e.g. ['cs.LG', 'cs.CL'].

    Returns
    -------
    list of Path
        Paths to downloaded PDF files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if categories:
        cat_query = " OR ".join(f"cat:{c}" for c in categories)
        full_query = f"({query}) AND ({cat_query})"
    else:
        full_query = query

    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    client = arxiv.Client()
    pdf_paths: List[Path] = []

    for result in client.results(search):
        paper_id = result.get_short_id().replace("/", "_")
        pdf_path = output_dir / f"{paper_id}.pdf"
        if pdf_path.exists():
            logger.info("Skipping existing PDF %s", pdf_path)
            pdf_paths.append(pdf_path)
            continue

        logger.info("Downloading %s -> %s", result.title, pdf_path)
        result.download_pdf(filename=str(pdf_path))
        pdf_paths.append(pdf_path)

    return pdf_paths


def main() -> None:
    """CLI entry point for quick manual data download."""
    import argparse

    parser = argparse.ArgumentParser(description="Download arXiv PDFs.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--max_results", type=int, default=50)
    parser.add_argument("--output_dir", type=Path, default=Path("data/raw/arxiv_pdfs"))
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        help="Optional arXiv categories, e.g. cs.LG cs.CL",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    download_arxiv_papers(
        query=args.query,
        max_results=args.max_results,
        output_dir=args.output_dir,
        categories=args.categories,
    )


if __name__ == "__main__":
    main()
