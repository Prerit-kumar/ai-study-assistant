"""Benchmark for the multi-document comparison feature.

Usage (from the src folder, inside the venv):
    python compare_eval.py              -> uses the models set in comparison.py
    python compare_eval.py gpt-4o       -> uses gpt-4o for extraction + table

Prints precision / recall of the selected papers, number checks per row,
empty rows, and time per question. Saves full results to a JSON file.
A run where any OpenAI call failed is marked INVALID instead of scored.
"""

import json
import re
import sys
import time

import comparison
from comparison import compare_documents_detailed, get_documents_in_store, NOT_FOUND

BENCHMARK = [
    {
        "name": "attention",
        "question": "Compare how these papers reduce the complexity of self-attention. Give a table.",
        "expected": ["BigBird.pdf", "Linformer.pdf", "Longformer.pdf", "Performer.pdf",
                     "Reformer.pdf", "FlashAttention.pdf"],
        "acceptable": ["Mamba.pdf", "Mistral_7B.pdf", "Transformer_XL.pdf"],
        "row_checks": {
            "Reformer.pdf": {"must": ["log"], "must_not": []},
        },
    },
    {
        "name": "retrieval",
        "question": "Compare how these papers use retrieval to improve language models. Give a table.",
        "expected": ["DPR_Dense_Passage_Retrieval.pdf", "ColBERT.pdf",
                     "RAG_Retrieval_Augmented_Generation.pdf", "REALM.pdf", "RETRO.pdf", "Self_RAG.pdf"],
        "acceptable": ["Toolformer.pdf", "ReAct.pdf"],
        "row_checks": {},
    },
    {
        "name": "peft",
        "question": "Compare the parameter-efficient fine-tuning methods in these papers. Give a table.",
        "expected": ["Adapters_Parameter_Efficient.pdf", "LoRA.pdf", "Prefix_Tuning.pdf", "QLoRA.pdf"],
        "acceptable": [],
        "row_checks": {},
    },
    {
        "name": "model_sizes",
        "question": ("Compare the model sizes and number of training tokens of the language models "
                     "in these papers. Give a table."),
        "expected": ["GPT3_Few_Shot_Learners.pdf", "Chinchilla.pdf", "LLaMA.pdf", "Llama2.pdf",
                     "Mistral_7B.pdf", "Mixtral.pdf"],
        "acceptable": ["ALBERT.pdf", "BERT.pdf", "RoBERTa.pdf", "T5.pdf", "DistilBERT.pdf", "ELECTRA.pdf",
                       "QLoRA.pdf", "RETRO.pdf", "Switch_Transformers.pdf", "InstructGPT.pdf",
                       "Scaling_Laws.pdf", "Transformer_XL.pdf"],
        "row_checks": {
            "GPT3_Few_Shot_Learners.pdf": {"must": ["175"], "must_not": []},
            "Chinchilla.pdf": {"must": ["70", "1.4"], "must_not": ["280", "300"]},
            "LLaMA.pdf": {"must": ["65"], "must_not": []},
            "Llama2.pdf": {"must": ["70"], "must_not": ["65"]},
            "Mixtral.pdf": {"must": ["47"], "must_not": []},
        },
    },
    {
        "name": "trap_no_answer",
        "question": "Compare the image segmentation methods in these papers. Give a table.",
        "expected": [],
        "acceptable": [],
        "row_checks": {},
    },
]


def table_rows(answer):
    """Returns {first_cell_lowercase: full_row_text} for every markdown table row."""
    rows = {}
    for line in answer.splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells:
            rows[cells[0].lower()] = line
    return rows


def find_row(rows, doc):
    key = doc.lower()
    short = re.sub(r"\.pdf$", "", key)
    for first_cell, line in rows.items():
        if first_cell in (key, short) or first_cell.startswith(short):
            return line
    return None


def empty_rows(rows):
    count = 0
    for first_cell, line in rows.items():
        if first_cell == "paper":
            continue
        cells = [c.strip().lower() for c in line.strip("|").split("|")][1:]
        if cells and all(c.startswith("not stated") for c in cells):
            count += 1
    return count


def evaluate(case, documents):
    start = time.time()
    result = compare_documents_detailed(case["question"], documents)
    seconds = time.time() - start

    if result.get("error"):
        return {"name": case["name"], "seconds": round(seconds, 1), "api_error": result["error"][:200]}

    answer = result["answer"]
    used = set(result["documents_used"])
    expected = set(case["expected"])
    allowed = expected | set(case["acceptable"])

    if not expected:
        passed = answer.strip().startswith(NOT_FOUND) or not used
        return {
            "name": case["name"], "seconds": round(seconds, 1), "trap_passed": passed,
            "used": sorted(used), "answer": answer,
        }

    correct = used & allowed
    precision = len(correct) / len(used) if used else 0.0
    recall = len(used & expected) / len(expected)

    rows = table_rows(answer)
    checks = []
    for doc, rule in case["row_checks"].items():
        row = find_row(rows, doc)
        if row is None:
            checks.append((doc, False, "row missing"))
            continue
        missing = [m for m in rule["must"] if m not in row]
        forbidden = [m for m in rule["must_not"] if m in row]
        ok = not missing and not forbidden
        detail = []
        if missing:
            detail.append(f"missing {missing}")
        if forbidden:
            detail.append(f"has wrong {forbidden}")
        checks.append((doc, ok, "; ".join(detail) or "ok"))

    return {
        "name": case["name"],
        "seconds": round(seconds, 1),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "wrong_papers": sorted(used - allowed),
        "missed_papers": sorted(expected - used),
        "row_checks": checks,
        "empty_rows": empty_rows(rows),
        "used": sorted(used),
        "answer": answer,
    }


def main():
    if len(sys.argv) > 1:
        comparison.CHAT_MODEL = sys.argv[1]
    label = f"select={comparison.SELECT_MODEL}, extract={comparison.CHAT_MODEL}"
    print(f"Running comparison benchmark ({label})\n")

    documents = get_documents_in_store()
    results = []
    for case in BENCHMARK:
        print(f"--- {case['name']} ---")
        r = evaluate(case, documents)
        results.append(r)
        if "api_error" in r:
            print(f"  API ERROR (this question is not scored): {r['api_error']}")
            continue
        if "trap_passed" in r:
            print(f"  trap passed: {r['trap_passed']}  ({r['seconds']}s)")
            continue
        print(f"  precision {r['precision']:.2f}  recall {r['recall']:.2f}  ({r['seconds']}s)")
        if r["wrong_papers"]:
            print(f"  wrong papers:  {', '.join(r['wrong_papers'])}")
        if r["missed_papers"]:
            print(f"  missed papers: {', '.join(r['missed_papers'])}")
        for doc, ok, detail in r["row_checks"]:
            print(f"  [{'PASS' if ok else 'FAIL'}] {doc}: {detail}")
        if r["empty_rows"]:
            print(f"  empty 'Not stated' rows: {r['empty_rows']}")

    out = f"compare_eval_{comparison.CHAT_MODEL}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    errors = [r for r in results if "api_error" in r]
    print("\n=== SUMMARY ===")
    print(f"models: {label}")
    if errors:
        print(f"RUN INVALID: {len(errors)} of {len(results)} questions hit OpenAI errors.")
        print("Fix the API problem (credits / rate limit) and run again. No scores reported.")
        print(f"\nDetails saved to {out}")
        return

    scored = [r for r in results if "precision" in r]
    checks = [c for r in scored for c in r["row_checks"]]
    print(f"avg precision: {sum(r['precision'] for r in scored) / len(scored):.2f}")
    print(f"avg recall:    {sum(r['recall'] for r in scored) / len(scored):.2f}")
    print(f"number checks: {sum(1 for c in checks if c[1])}/{len(checks)} passed")
    print(f"empty rows:    {sum(r['empty_rows'] for r in scored)}")
    traps = [r for r in results if "trap_passed" in r]
    print(f"trap question: {'passed' if all(r['trap_passed'] for r in traps) else 'FAILED'}")
    print(f"total time:    {sum(r['seconds'] for r in results):.0f}s")
    print(f"\nFull results saved to {out}")


if __name__ == "__main__":
    main()