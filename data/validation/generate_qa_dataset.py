import os
import time
import json
from pathlib import Path
from mistralai import Mistral

API_KEY = os.getenv("MISTRAL_API_KEY")
if not API_KEY:
    raise ValueError("❌ MISTRAL_API_KEY is missing. Please run: export MISTRAL_API_KEY=...")

client = Mistral(api_key=API_KEY)

CHUNKS_PATH = Path("data/chunks/local/chunks.jsonl")
OUT_PATH = Path("data/validation/qa.jsonl")

def generate_question(text: str):
    """Generate 1–2 QA questions with retry logic."""
    prompt = (
        "You are creating a validation dataset for a scientific RAG system.\n"
        "Generate 1–2 short factual questions based strictly on the text below.\n"
        "Questions must be answerable ONLY from this text.\n\n"
        f"TEXT:\n{text}"
    )

    for attempt in range(5):
        try:
            res = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
            )
            content = res.choices[0].message.content.strip()
            lines = [l.strip(" -•") for l in content.split("\n") if len(l.strip()) > 10]
            return lines

        except Exception as e:
            print(f"⚠️ Error on attempt {attempt+1}/5: {e}")
            time.sleep(2 + attempt)

    print("❌ Failed after 5 retries.")
    return []

qa_items = []

with CHUNKS_PATH.open(encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 200:  # ~200 chunks → ~300 вопросов
            break

        record = json.loads(line)
        chunk_id = record["chunk_id"]
        text = record["text"]

        print(f"Generating questions for chunk {chunk_id}...")
        questions = generate_question(text)
        time.sleep(1)  # pause to avoid rate limits

        for q in questions:
            qa_items.append({
                "id": f"q_{len(qa_items)}",
                "question": q,
                "gold_chunk_ids": [chunk_id]
            })

OUT_PATH.parent.mkdir(exist_ok=True)

with OUT_PATH.open("w", encoding="utf-8") as f:
    for item in qa_items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✨ Saved {len(qa_items)} QA pairs to {OUT_PATH}")
