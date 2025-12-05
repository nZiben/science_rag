"""Convert downloaded PDFs into plain text files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

from pypdf import PdfReader

logger = logging.getLogger(__name__)


def pdf_to_text(pdf_path: Path) -> str:
    """
    Extract plain text from a PDF file with Unicode error handling.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.

    Returns
    -------
    str
        Concatenated text of all pages with cleaned Unicode.
    """
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        # Clean problematic Unicode characters before appending
        cleaned_text = clean_unicode(page_text)
        pages.append(cleaned_text)
    return "\n".join(pages)


def clean_unicode(text: str) -> str:
    """
    Clean problematic Unicode characters that cause encoding errors.
    
    Parameters
    ----------
    text:
        Input text that may contain problematic Unicode.
    
    Returns
    -------
    str
        Cleaned text safe for UTF-8 encoding.
    """
    # Method 1: Replace surrogates and problematic characters
    # Remove surrogate pairs (U+D800 to U+DFFF)
    text = ''.join(char for char in text if not ('\ud800' <= char <= '\udfff'))
    
    # Method 2: Handle specific problematic characters
    # \ud835 is a mathematical alphanumeric symbol
    text = text.replace('\ud835', '')
    
    # Method 3: Use encode/decode with error handling as fallback
    try:
        # Try to encode as UTF-8 to check for errors
        text.encode('utf-8')
        return text
    except UnicodeEncodeError:
        # If there are errors, clean them
        return text.encode('utf-8', 'ignore').decode('utf-8')


def bulk_pdf_to_text(
    pdf_dir: Path,
    output_dir: Path,
    suffix: str = ".txt",
) -> List[Path]:
    """
    Convert all PDFs in a directory to text files.

    Parameters
    ----------
    pdf_dir:
        Directory with PDF files.
    output_dir:
        Directory to save text files.
    suffix:
        Suffix for output files.

    Returns
    -------
    list of Path
        Paths to generated text files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_paths: List[Path] = []

    pdf_files: Iterable[Path] = pdf_dir.glob("*.pdf")

    for pdf_path in pdf_files:
        stem = pdf_path.stem
        txt_path = output_dir / f"{stem}{suffix}"
        if txt_path.exists():
            logger.info("Skipping existing text file %s", txt_path)
            txt_paths.append(txt_path)
            continue

        logger.info("Converting %s -> %s", pdf_path, txt_path)
        try:
            text = pdf_to_text(pdf_path)
            # Additional safety: clean the entire text before writing
            final_text = clean_unicode(text)
            txt_path.write_text(final_text, encoding="utf-8")
            txt_paths.append(txt_path)
        except Exception as e:
            logger.error("Failed to convert %s: %s", pdf_path, e)
            # Create empty file or skip - here we skip
            continue

    return txt_paths


def main() -> None:
    """CLI entry point for batch PDF → text conversion."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert arXiv PDFs to text.")
    parser.add_argument(
        "--pdf_dir",
        type=Path,
        default=Path("data/raw/arxiv_pdfs"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed/texts"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = bulk_pdf_to_text(args.pdf_dir, args.output_dir)
    logger.info("Successfully converted %d PDFs to text", len(result))


if __name__ == "__main__":
    main()