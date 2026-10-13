"""Score base vs v1 vs v2 (local) and gpt-4o-mini (reference) on the v2 test set.

Run from the src folder with the project venv, after eval_generate_v2.py:
    python eval_judge_v2.py
Cost: about $0.10 (gpt-4o-mini answers ~150 questions, then judges 4 x ~150 answers).

Metrics per model:
  answerable   -> correct / partial / incorrect / wrongly refused (LLM judge)
  unanswerable -> correctly refused / hallucinated (LLM judge)
  citations    -> did "USED_CHUNKS:" point at the chunk that holds the answer (or NONE when it should)
  latency      -> median seconds per answer
"""

import json
import re
import statistics
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(max_retries=3)

PRED_FILE = r"C:\Users\preri\finetune\predictions_v2.jsonl"
RESULT_FILE = r"C:\Users\preri\finetune\benchmark_results_v2.json"
USED_MARKER = "USED_CHUNKS:"

MODEL_KEYS = [
    ("base_answer", "Base 3B"),
    ("v1_answer", "Fine-tuned v1"),
    ("v2_answer", "Fine-tuned v2"),
    ("gpt_answer", "gpt-4o-mini"),
]

ANSWERABLE_PROMPT = """You are grading a question-answering model.

Question: {question}
Reference answer (ground truth): {reference}
Model answer: {answer}

Classify the model answer as exactly one of:
- "correct": conveys the key facts of the reference answer
- "partial": gets some key facts right but misses or gets others wrong
- "incorrect": wrong or unrelated to the reference answer
- "refused": says the information is not available / could not be found

Return JSON only: {{"verdict": "correct" | "partial" | "incorrect" | "refused"}}"""

UNANSWERABLE_PROMPT = """A question was asked, but the context given to the model does NOT contain the answer.
The correct behaviour is to decline (say the information is not available).

Question: {question}
Model answer: {answer}

Did the model decline, or did it give a substantive answer (which would be a hallucination)?

Return JSON only: {{"refused": true | false}}"""


def chat_json(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print("judge error:", e)
        return None


def gpt_answer(record):
    """Reference model: gpt-4o-mini gets exactly the same prompt as the local models."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=record["prompt_messages"],
            temperature=0,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("gpt-4o-mini error:", e)
        return None


def split_answer(raw):
    """Returns (answer text without the USED_CHUNKS line, set of cited numbers or None if no line)."""
    idx = raw.upper().rfind(USED_MARKER)
    if idx == -1:
        return raw.strip(), None
    cited = {int(n) for n in re.findall(r"\d+", raw[idx + len(USED_MARKER):])}
    return raw[:idx].strip(), cited


def judge(args):
    record, key = args
    answer, _ = split_answer(record[key])
    if record["type"] == "answerable":
        result = chat_json(ANSWERABLE_PROMPT.format(
            question=record["question"], reference=record["reference"], answer=answer))
        return None if result is None else result.get("verdict", "incorrect")
    result = chat_json(UNANSWERABLE_PROMPT.format(question=record["question"], answer=answer))
    return None if result is None else bool(result.get("refused", False))


def citation_ok(record, key):
    _, cited = split_answer(record[key])
    if cited is None:
        return False
    if record["type"] == "answerable":
        return record["gold"] in cited
    return len(cited) == 0  # "USED_CHUNKS: NONE"


def main():
    with open(PRED_FILE, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    print(f"Test questions: {len(records)}")

    print("Getting gpt-4o-mini reference answers ...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        gpt = list(pool.map(gpt_answer, records))
    if any(a is None for a in gpt):
        print("RUN INVALID: some gpt-4o-mini calls failed (check credits). No scores reported.")
        return
    for record, answer in zip(records, gpt):
        record["gpt_answer"] = answer
        record["gpt_seconds"] = None

    answerable = [r for r in records if r["type"] == "answerable"]
    unanswerable = [r for r in records if r["type"] == "unanswerable"]

    results = {}
    for key, label in MODEL_KEYS:
        print(f"Judging {label} ...")
        with ThreadPoolExecutor(max_workers=8) as pool:
            verdicts = list(pool.map(judge, [(r, key) for r in answerable]))
            refusals = list(pool.map(judge, [(r, key) for r in unanswerable]))
        if any(v is None for v in verdicts + refusals):
            print("RUN INVALID: some judge calls failed (check credits). No scores reported.")
            return

        n_a, n_u = max(len(verdicts), 1), max(len(refusals), 1)
        seconds = [r[key.replace("_answer", "_seconds")] for r in records]
        seconds = [s for s in seconds if s is not None]
        results[key] = {
            "label": label,
            "correct": verdicts.count("correct") / n_a,
            "partial": verdicts.count("partial") / n_a,
            "incorrect": verdicts.count("incorrect") / n_a,
            "wrongly_refused": verdicts.count("refused") / n_a,
            "correctly_refused": sum(refusals) / n_u,
            "hallucinated": 1 - sum(refusals) / n_u,
            "citation_accuracy": sum(citation_ok(r, key) for r in records) / len(records),
            "median_seconds": statistics.median(seconds) if seconds else None,
        }

    def pct(x):
        return f"{x * 100:5.1f}%"

    rows = [
        ("Correct", "correct"), ("Partial", "partial"), ("Incorrect", "incorrect"),
        ("Wrongly refused", "wrongly_refused"), ("Correctly refused", "correctly_refused"),
        ("Hallucinated", "hallucinated"), ("Citation accuracy", "citation_accuracy"),
    ]
    print("\n========================= BENCHMARK v2 =========================")
    print(f"Answerable: {len(answerable)}   Unanswerable: {len(unanswerable)}")
    print(f"{'':22s}" + "".join(f"{results[k]['label']:>15s}" for k, _ in MODEL_KEYS))
    for name, field in rows:
        print(f"{name:22s}" + "".join(f"{pct(results[k][field]):>15s}" for k, _ in MODEL_KEYS))
    print(f"{'Median seconds':22s}" + "".join(
        f"{(str(results[k]['median_seconds']) if results[k]['median_seconds'] is not None else 'API'):>15s}"
        for k, _ in MODEL_KEYS))
    print("================================================================")

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(PRED_FILE.replace(".jsonl", "_with_gpt.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved: {RESULT_FILE}")


if __name__ == "__main__":
    main()
