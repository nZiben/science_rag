"""
Build sentence-transformer embeddings for all local chunks.
"""

import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer


def main() -> None:
    chunks_path = Path("data/chunks/local/chunks.jsonl")
    embeddings_out = Path("data/chunks/local/embeddings.npy")
    doc_ids_out = Path("data/chunks/local/doc_ids.txt")

    texts = []
    doc_ids = []

    # ⭐ Читаем строго построчно, без read_text()
    with open(chunks_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"❌ JSON error on line {i}: {e}")
                print(repr(line[:300]))
                raise
            doc_ids.append(rec["chunk_id"])
            texts.append(rec["text"])

    print("Loaded chunks:", len(doc_ids))

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)

    embeddings_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_out, embeddings)
    doc_ids_out.write_text("\n".join(doc_ids), encoding="utf-8")

    print("Saved:", embeddings_out, doc_ids_out)


if __name__ == "__main__":
    main()
