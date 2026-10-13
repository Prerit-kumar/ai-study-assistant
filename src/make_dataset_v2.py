"""Fine-tuning dataset v2: realistic RAG contexts built from the fixed 40-paper index.

What changed from v1:
  - Text comes from the fixed index (no glued words, 180-word chunks).
  - Each example shows the model 5 numbered chunks found by our own retrieval
    (vector search + cross-encoder rerank), exactly like the app, not just the one right chunk.
  - Answerable: the chunk with the answer is among the 5 (with distractors around it).
  - Unanswerable: a real question about paper X, but the 5 chunks come from OTHER papers,
    so the right behaviour is to refuse.
  - The target answer ends with "USED_CHUNKS: n" (or NONE), the same format the app parses.
  - 5 papers are held out completely for the test set.

Run from the src folder with the project venv (about 10-15 minutes, about $0.25 of gpt-4o-mini):
    python make_dataset_v2.py
"""

import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

from embed_store import collection, model
from hybrid_search import reranker

load_dotenv()
client = OpenAI(max_retries=3)

OUT_DIR = "../finetune_data_v2"
SEED = 42
NUM_TEST_PAPERS = 5
TRAIN_CHUNKS_PER_PAPER = 20     # x 35 papers x 3 examples = about 2,100 training examples
TEST_CHUNKS_PER_PAPER = 10      # x 5 papers x 3 examples = about 150 test examples
CONTEXT_CHUNKS = 5              # chunks shown to the model per example
CANDIDATES = 15                 # vector-search candidates before reranking
WORKERS = 8

NOT_FOUND = "I couldn't find this in the provided documents."
USED_MARKER = "USED_CHUNKS:"

SYSTEM_PROMPT = (
    "You are a research assistant. Answer the question using ONLY the numbered context chunks. "
    f'If the context does not contain the answer, reply exactly: "{NOT_FOUND}" '
    f"After your answer, add one final line in exactly this format: {USED_MARKER} followed by the "
    f"numbers of the chunks you used (for example: {USED_MARKER} 2), or {USED_MARKER} NONE if you "
    "could not answer."
)

GEN_PROMPT = """You are creating training data for a research-paper question answering model.

The passage below comes from the paper "{paper}". Write exactly 3 question-answer pairs.

Rules:
- Each answer must be fully supported by the passage: 1-3 sentences, with specific details (numbers, names, methods) when available.
- Each question must be specific to this paper, so that another paper would not answer it. Name the method or model when that helps.
- Questions must be self-contained. Never say "the passage", "the text" or "according to the context".
- Do not use outside knowledge.

Return JSON only, in this format:
{{"pairs": [{{"question": "...", "answer": "..."}}, {{"question": "...", "answer": "..."}}, {{"question": "...", "answer": "..."}}]}}

Passage:
\"\"\"
{chunk}
\"\"\"
"""

REFERENCE_LIKE = re.compile(r"et al\.|arXiv|\(\d{4}[a-z]?\)|\b(?:19|20)\d{2}\b")


def good_chunk(text):
    """Skip short chunks, reference lists and tables of numbers."""
    if len(text.split()) < 120:
        return False
    if len(REFERENCE_LIKE.findall(text)) > 12:
        return False
    letters = sum(c.isalpha() for c in text)
    return letters / max(len(text), 1) >= 0.6


def load_chunks():
    data = collection.get(include=["documents", "metadatas"])
    by_paper = {}
    for chunk_id, text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        by_paper.setdefault(meta["source"], []).append((chunk_id, text))
    return by_paper


def generate_pairs(job):
    paper, chunk_id, text = job
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": GEN_PROMPT.format(paper=paper, chunk=text)}],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        data = json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"   generation failed for {paper}: {e}")
        return job, []

    pairs = []
    for item in data.get("pairs", []):
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if q and a:
            pairs.append((q, a))
    return job, pairs


def retrieve(question, allowed_papers, exclude_paper=None):
    """Our app's retrieval (vector search + cross-encoder rerank), limited to some papers."""
    sources = [p for p in allowed_papers if p != exclude_paper]
    results = collection.query(
        query_embeddings=model.encode([question]).tolist(),
        n_results=CANDIDATES,
        where={"source": {"$in": sources}},
    )
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    if not docs:
        return []
    scores = reranker.predict([[question, d] for d in docs])
    ranked = sorted(zip(ids, docs, metas, scores), key=lambda x: x[3], reverse=True)
    return [(i, d, m["source"]) for i, d, m, _ in ranked[:CONTEXT_CHUNKS]]


def build_example(context, question, target, paper, kind, gold, reference):
    labeled = [f"[{i}] (from {source})\n{text}" for i, (_, text, source) in enumerate(context, start=1)]
    user = "Context:\n" + "\n\n---\n\n".join(labeled) + f"\n\nQuestion: {question}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ],
        "paper": paper,
        "type": kind,
        "gold": gold,
        "question": question,
        "reference": reference,
    }


def main():
    rng = random.Random(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    by_paper = load_chunks()
    papers = sorted(by_paper)
    rng.shuffle(papers)
    test_papers = sorted(papers[-NUM_TEST_PAPERS:])
    train_papers = sorted(papers[:-NUM_TEST_PAPERS])
    print(f"{len(papers)} papers. Held-out test papers: {test_papers}")

    jobs = []
    for paper in papers:
        chunks = [(cid, text) for cid, text in by_paper[paper] if good_chunk(text)]
        rng.shuffle(chunks)
        limit = TEST_CHUNKS_PER_PAPER if paper in test_papers else TRAIN_CHUNKS_PER_PAPER
        for cid, text in chunks[:limit]:
            jobs.append((paper, cid, text))
    print(f"Generating questions from {len(jobs)} chunks with gpt-4o-mini ...")

    generated = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(generate_pairs, job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), start=1):
            generated.append(fut.result())
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)} chunks done")

    print("\nBuilding retrieval contexts (local, no API cost) ...")
    counts = {"train": 0, "test": 0, "gold_retrieved": 0, "gold_inserted": 0}
    files = {
        "train": open(os.path.join(OUT_DIR, "train.jsonl"), "w", encoding="utf-8"),
        "test": open(os.path.join(OUT_DIR, "test.jsonl"), "w", encoding="utf-8"),
    }
    try:
        for n, ((paper, gold_id, gold_text), pairs) in enumerate(sorted(generated, key=lambda g: g[0][1]), start=1):
            if len(pairs) < 3:
                continue
            split = "test" if paper in test_papers else "train"
            allowed = test_papers if split == "test" else train_papers
            examples = []

            # Two answerable examples: the gold chunk is among the 5 chunks.
            for question, answer in pairs[:2]:
                context = retrieve(question, allowed)
                if gold_id in [c[0] for c in context]:
                    counts["gold_retrieved"] += 1
                else:
                    counts["gold_inserted"] += 1
                    context = context[: CONTEXT_CHUNKS - 1]
                    context.insert(rng.randint(0, len(context)), (gold_id, gold_text, paper))
                gold = [c[0] for c in context].index(gold_id) + 1
                target = f"{answer}\n{USED_MARKER} {gold}"
                examples.append(build_example(context, question, target, paper, "answerable", gold, answer))

            # One unanswerable example: a question about this paper, context from other papers only.
            question, _ = pairs[2]
            context = retrieve(question, allowed, exclude_paper=paper)
            target = f"{NOT_FOUND}\n{USED_MARKER} NONE"
            examples.append(build_example(context, question, target, paper, "unanswerable", None, NOT_FOUND))

            for ex in examples:
                files[split].write(json.dumps(ex, ensure_ascii=False) + "\n")
                counts[split] += 1
            if n % 100 == 0:
                print(f"  {n}/{len(generated)} chunks done")
    finally:
        for f in files.values():
            f.close()

    total_answerable = counts["gold_retrieved"] + counts["gold_inserted"]
    print(f"\nDone. Train examples: {counts['train']}, Test examples: {counts['test']}")
    print(f"Retrieval found the right chunk by itself for {counts['gold_retrieved']}/{total_answerable} "
          f"answerable questions (the rest had it inserted).")
    print(f"Saved in: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
