"""Simple text chunking utilities for RAG."""

from __future__ import annotations
import logging
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class DocumentChunk:
    """Represents a single text chunk with metadata."""

    chunk_id: str
    source_id: str
    text: str
    metadata: Dict[str, str]


def _make_chunk_id(source_id: str, idx: int, text: str) -> str:
    """Stable chunk identifier based on source and content hash."""
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f"{source_id}_{idx:04d}_{digest}"

def chunk_text(
    text: str,
    source_id: str,
    max_chars: int = 1200,
    overlap: int = 200,
    extra_metadata: Dict[str, str] | None = None,
) -> List[DocumentChunk]:
    """
    Split long text into overlapping character-based chunks.
    
    FIXED: Added protection against infinite loops
    """
    extra_metadata = extra_metadata or {}
    chunks: List[DocumentChunk] = []

    start = 0
    idx = 0
    n = len(text)

    # Защита от бесконечного цикла
    max_iterations = n * 2  # Максимальное разумное число итераций
    
    iteration = 0
    while start < n and iteration < max_iterations:
        iteration += 1
        
        end = min(start + max_chars, n)
        chunk_text_str = text[start:end].strip()
        
        # Если после strip() текст пустой, пропускаем
        if not chunk_text_str:
            start = end  # Двигаемся вперед
            continue
        
        chunk_id = _make_chunk_id(source_id, idx, chunk_text_str)
        metadata = {"source_id": source_id, **extra_metadata}
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                source_id=source_id,
                text=chunk_text_str,
                metadata=metadata,
            )
        )
        idx += 1
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: защита от зацикливания
        new_start = end - overlap
        if new_start <= start:
            # Если new_start не двигает нас вперед, двигаемся на max_chars
            new_start = start + max_chars
        
        start = new_start
    
    if iteration >= max_iterations:
        logging.warning(f"File {source_id}: reached iteration limit ({max_iterations})")
    
    return chunks

def chunk_corpus_to_jsonl(
    texts_dir: Path,
    output_path: Path,
    max_chars: int = 1200,
    overlap: int = 200,
) -> None:
    import json
    import logging
    
    logger = logging.getLogger(__name__)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    txt_files = list(texts_dir.glob("*.txt"))
    logger.info(f"Found {len(txt_files)} text files to process")

    with output_path.open("w", encoding="utf-8") as f_out:
        for i, txt_path in enumerate(txt_files):
            if i % 10 == 0:
                logger.info(f"Processing file {i+1}/{len(txt_files)}: {txt_path.name}")
            
            source_id = txt_path.stem
            text = txt_path.read_text(encoding="utf-8")
            chunks = chunk_text(text, source_id, max_chars, overlap)
            
            for ch in chunks:
                record = {
                    "chunk_id": ch.chunk_id,
                    "source_id": ch.source_id,
                    "text": ch.text,
                    "metadata": ch.metadata,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved {len(txt_files)} files to {output_path}")

def main() -> None:
    """CLI entry point for corpus chunking."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    parser = argparse.ArgumentParser(description="Chunk processed texts into JSONL.")
    parser.add_argument(
        "--texts_dir",
        type=Path,
        default=Path("data/processed/texts"),
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("data/chunks/local/chunks.jsonl"),
    )
    parser.add_argument("--max_chars", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    args = parser.parse_args()

    chunk_corpus_to_jsonl(
        texts_dir=args.texts_dir,
        output_path=args.output_path,
        max_chars=args.max_chars,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
