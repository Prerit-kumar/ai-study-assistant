import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

from pdf_reader import read_pdf
from chunker import chunk_text

load_dotenv()
client = OpenAI()

PAPERS_DIR = "../papers"
OUT_DIR = "../finetune_data"
CHUNKS_PER_PAPER = 15
NUM_TEST_PAPERS = 5
SEED = 42
WORKERS = 8

NOT_FOUND = "I couldn't find this in the provided documents."
SYSTEM_PROMPT = (
    "You are a study assistant. Answer the question using ONLY the provided context. "
    f'If the answer is not in the context, reply exactly: "{NOT_FOUND}"'
)

GEN_PROMPT = """You are creating training data for a research-paper question answering model.

Read the passage below and produce:
1. Exactly 2 "answerable" question-answer pairs. The answer must be fully supported by the passage, 1-3 sentences, with specific details (numbers, names, methods) when available.
2. Exactly 1 "unanswerable" question. It should be a realistic question about the same paper or topic, but its answer must NOT be found in the passage.

Rules:
- Questions must be self-contained. Never say "the passage", "the text" or "according to the context".
- Do not use outside knowledge in the answers.

Return JSON only, in this format:
{{"answerable": [{{"question": "...", "answer": "..."}}, {{"question": "...", "answer": "..."}}],
  "unanswerable": [{{"question": "..."}}]}}

Passage:
\"\"\"
{chunk}
\"\"\"
"""


def clean_text(text):
    text = re.sub(r"--- Page \d+ ---", " ", text)
    idx = text.rfind("References")
    if idx > len(text) * 0.5:
        text = text[:idx]
    return re.sub(r"\s+", " ", text).strip()


def good_chunk(chunk):
    words = chunk.split()
    if len(words) < 150:
        return False
    letters = sum(c.isalpha() for c in chunk)
    return letters / max(len(chunk), 1) >= 0.6


def make_example(chunk, question, answer, paper, kind):
    user = f"Context:\n{chunk}\n\nQuestion: {question}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
        "paper": paper,
        "type": kind,
    }


def generate_for_chunk(paper, chunk):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": GEN_PROMPT.format(chunk=chunk)}],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        data = json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"   generation failed for {paper}: {e}")
        return []

    examples = []
    for item in data.get("answerable", []):
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if q and a:
            examples.append(make_example(chunk, q, a, paper, "answerable"))
    for item in data.get("unanswerable", []):
        q = str(item.get("question", "")).strip()
        if q:
            examples.append(make_example(chunk, q, NOT_FOUND, paper, "unanswerable"))
    return examples


def main():
    random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    pdfs = sorted(f for f in os.listdir(PAPERS_DIR) if f.lower().endswith(".pdf"))
    print(f"Found {len(pdfs)} PDFs")

    random.shuffle(pdfs)
    test_papers = set(pdfs[-NUM_TEST_PAPERS:])
    print("Held-out test papers:", sorted(test_papers))

    jobs = []
    for pdf in pdfs:
        name = pdf[:-4]
        print(f"Reading {name} ...")
        try:
            text = clean_text(read_pdf(os.path.join(PAPERS_DIR, pdf)))
        except Exception as e:
            print(f"   could not read {name}: {e}")
            continue
        chunks = [c for c in chunk_text(text) if good_chunk(c)]
        random.shuffle(chunks)
        for c in chunks[:CHUNKS_PER_PAPER]:
            jobs.append((name, c, pdf in test_papers))

    print(f"\nGenerating examples from {len(jobs)} chunks ...")

    train_path = os.path.join(OUT_DIR, "train.jsonl")
    test_path = os.path.join(OUT_DIR, "test.jsonl")
    n_train = n_test = 0

    with open(train_path, "w", encoding="utf-8") as ftrain, open(
        test_path, "w", encoding="utf-8"
    ) as ftest:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {
                pool.submit(generate_for_chunk, name, chunk): is_test
                for name, chunk, is_test in jobs
            }
            for i, fut in enumerate(as_completed(futures), start=1):
                is_test = futures[fut]
                for ex in fut.result():
                    line = json.dumps(ex, ensure_ascii=False) + "\n"
                    if is_test:
                        ftest.write(line)
                        n_test += 1
                    else:
                        ftrain.write(line)
                        n_train += 1
                if i % 25 == 0:
                    print(f"  {i}/{len(jobs)} chunks done")

    print(f"\nDone. Train examples: {n_train}, Test examples: {n_test}")
    print(f"Saved in: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()