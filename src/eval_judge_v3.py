"""Score fine-tuned v2 vs v3 (local) and gpt-4o-mini (reference) on both test sets.

Run from the src folder with the project venv, after eval_generate_v3.py:
    python eval_judge_v3.py
Cost: about $0.20.

Test sets:
  docs   - unseen everyday documents (the reason for v3)
  papers - the old research-paper test (v3 must not get worse here)

Question types:
  answerable   = lookup / reasoning / summary (docs) or answerable (papers): judged correct / partial / incorrect / refused
  unanswerable = absent (the document's own chunk is shown, but the answer isn't in it)
                 or unanswerable (only other documents were retrieved): the model must refuse
"""

import json
import re
import statistics
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(max_retries=3)

PRED_FILES = {
    "docs": r"C:\Users\preri\finetune\predictions_v3_docs.jsonl",
    "papers": r"C:\Users\preri\finetune\predictions_v3_papers.jsonl",
}
RESULT_FILE = r"C:\Users\preri\finetune\benchmark_results_v3.json"
USED_MARKER = "USED_CHUNKS:"
ANSWERABLE_TYPES = {"answerable", "lookup", "reasoning", "summary"}

MODEL_KEYS = [
    ("v2_answer", "Fine-tuned v2"),
    ("v3_answer", "Fine-tuned v3"),
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
            model="gpt-4o-mini", messages=record["prompt_messages"], temperature=0, max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("gpt-4o-mini error:", e)
        return None


def split_answer(raw):
    idx = raw.upper().rfind(USED_MARKER)
    if idx == -1:
        return raw.strip(), None
    cited = {int(n) for n in re.findall(r"\d+", raw[idx + len(USED_MARKER):])}
    return raw[:idx].strip(), cited


def judge(args):
    record, key = args
    answer, _ = split_answer(record[key])
    if record["type"] in ANSWERABLE_TYPES:
        result = chat_json(ANSWERABLE_PROMPT.format(
            question=record["question"], reference=record["reference"], answer=answer))
        return None if result is None else result.get("verdict", "incorrect")
    result = chat_json(UNANSWERABLE_PROMPT.format(question=record["question"], answer=answer))
    return None if result is None else bool(result.get("refused", False))


def citation_ok(record, key):
    _, cited = split_answer(record[key])
    if cited is None:
        return False
    if record["type"] in ANSWERABLE_TYPES:
        return record["gold"] in cited
    return len(cited) == 0


def summarize(records, key):
    ans = [r for r in records if r["type"] in ANSWERABLE_TYPES]
    una = [r for r in records if r["type"] not in ANSWERABLE_TYPES]
    verdicts = [r["judged"][key] for r in ans]
    refusals = [r["judged"][key] for r in una]
    n_a, n_u = len(ans), len(una)
    a = (lambda x: x / n_a) if n_a else (lambda x: None)   # None = no questions of this kind in the group
    u = (lambda x: x / n_u) if n_u else (lambda x: None)
    return {
        "n_answerable": len(ans),
        "n_unanswerable": len(una),
        "correct": a(verdicts.count("correct")),
        "partial": a(verdicts.count("partial")),
        "incorrect": a(verdicts.count("incorrect")),
        "wrongly_refused": a(verdicts.count("refused")),
        "correctly_refused": u(sum(refusals)),
        "hallucinated": u(n_u - sum(refusals)),
        "citation_accuracy": sum(citation_ok(r, key) for r in records) / max(len(records), 1),
    }


def pct(x):
    return "    -" if x is None else f"{x * 100:5.1f}%"


def print_table(title, records):
    rows = [
        ("Correct", "correct"), ("Partial", "partial"), ("Wrongly refused", "wrongly_refused"),
        ("Hallucinated", "hallucinated"), ("Citation accuracy", "citation_accuracy"),
    ]
    stats = {key: summarize(records, key) for key, _ in MODEL_KEYS}
    any_stats = next(iter(stats.values()))
    print(f"\n--- {title}  (answerable {any_stats['n_answerable']}, must-refuse {any_stats['n_unanswerable']})")
    print(f"{'':22s}" + "".join(f"{label:>15s}" for _, label in MODEL_KEYS))
    for name, field in rows:
        print(f"{name:22s}" + "".join(f"{pct(stats[k][field]):>15s}" for k, _ in MODEL_KEYS))
    return stats


def main():
    results = {}
    for set_name, path in PRED_FILES.items():
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        print(f"\n{set_name}: {len(records)} test questions. Getting gpt-4o-mini answers ...")
        with ThreadPoolExecutor(max_workers=8) as pool:
            gpt = list(pool.map(gpt_answer, records))
        if any(a is None for a in gpt):
            print("RUN INVALID: some gpt-4o-mini calls failed (check credits). No scores reported.")
            return
        for record, answer in zip(records, gpt):
            record["gpt_answer"] = answer
            record["judged"] = {}

        for key, label in MODEL_KEYS:
            print(f"  judging {label} ...")
            with ThreadPoolExecutor(max_workers=8) as pool:
                judged = list(pool.map(judge, [(r, key) for r in records]))
            if any(j is None for j in judged):
                print("RUN INVALID: some judge calls failed (check credits). No scores reported.")
                return
            for record, j in zip(records, judged):
                record["judged"][key] = j

        print(f"\n=================== {set_name.upper()} ===================")
        results[set_name] = {"all": print_table("All questions", records)}
        if set_name == "docs":
            new_types = [r for r in records if r["paper"].startswith("test-new")]
            seen_types = [r for r in records if not r["paper"].startswith("test-new")]
            ocr = [r for r in records if r.get("ocr")]
            results[set_name]["seen_types"] = print_table("Document types seen in training", seen_types)
            results[set_name]["new_types"] = print_table("Document types NEVER seen in training", new_types)
            if ocr:
                results[set_name]["ocr"] = print_table("Documents read with OCR", ocr)
            for qtype in ["lookup", "reasoning", "summary", "absent", "unanswerable"]:
                subset = [r for r in records if r["type"] == qtype]
                if subset:
                    results[set_name][qtype] = print_table(f"Question type: {qtype}", subset)

        seconds = {k: [r.get(k.replace("_answer", "_seconds")) for r in records] for k, _ in MODEL_KEYS[:2]}
        for k, values in seconds.items():
            values = [v for v in values if v is not None]
            if values:
                results[set_name]["all"][k]["median_seconds"] = statistics.median(values)

        with open(path.replace(".jsonl", "_judged.jsonl"), "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {RESULT_FILE}")


if __name__ == "__main__":
    main()
