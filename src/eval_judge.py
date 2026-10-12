import json
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

PRED_FILE = r"C:\Users\preri\finetune\predictions.jsonl"
RESULT_FILE = r"C:\Users\preri\finetune\benchmark_results.json"

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

UNANSWERABLE_PROMPT = """A question was asked about a passage that does NOT contain the answer.
The correct behaviour is to decline (say the information is not available).

Question: {question}
Model answer: {answer}

Did the model decline, or did it give a substantive answer (which would be a hallucination)?

Return JSON only: {{"refused": true | false}}"""


def ask_judge(prompt):
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
        return {}


def judge_one(args):
    record, model_key = args
    answer = record[model_key]
    if record["type"] == "answerable":
        prompt = ANSWERABLE_PROMPT.format(
            question=record["question"],
            reference=record["reference"],
            answer=answer,
        )
        return ask_judge(prompt).get("verdict", "incorrect")

    prompt = UNANSWERABLE_PROMPT.format(
        question=record["question"], answer=answer
    )
    return bool(ask_judge(prompt).get("refused", False))


def main():
    records = []
    with open(PRED_FILE, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    answerable = [r for r in records if r["type"] == "answerable"]
    unanswerable = [r for r in records if r["type"] == "unanswerable"]
    print(f"Answerable: {len(answerable)}, Unanswerable: {len(unanswerable)}")

    results = {}
    for key in ["base_answer", "finetuned_answer"]:
        print(f"Judging {key} ...")
        with ThreadPoolExecutor(max_workers=8) as pool:
            ans_verdicts = list(
                pool.map(judge_one, [(r, key) for r in answerable])
            )
            unans_refused = list(
                pool.map(judge_one, [(r, key) for r in unanswerable])
            )

        n_a = max(len(ans_verdicts), 1)
        n_u = max(len(unans_refused), 1)
        results[key] = {
            "correct": ans_verdicts.count("correct") / n_a,
            "partial": ans_verdicts.count("partial") / n_a,
            "incorrect": ans_verdicts.count("incorrect") / n_a,
            "wrongly_refused": ans_verdicts.count("refused") / n_a,
            "correctly_refused": sum(unans_refused) / n_u,
            "hallucinated": 1 - sum(unans_refused) / n_u,
        }

    def pct(x):
        return f"{x * 100:5.1f}%"

    b = results["base_answer"]
    f_ = results["finetuned_answer"]

    print("\n================ BENCHMARK ================")
    print(f"{'':28s}{'Base':>10s}{'Fine-tuned':>14s}")
    print(f"Answerable questions (N={len(answerable)})")
    print(f"  {'Correct':26s}{pct(b['correct']):>10s}{pct(f_['correct']):>14s}")
    print(f"  {'Partial':26s}{pct(b['partial']):>10s}{pct(f_['partial']):>14s}")
    print(f"  {'Incorrect':26s}{pct(b['incorrect']):>10s}{pct(f_['incorrect']):>14s}")
    print(f"  {'Wrongly refused':26s}{pct(b['wrongly_refused']):>10s}{pct(f_['wrongly_refused']):>14s}")
    print(f"Unanswerable questions (N={len(unanswerable)})")
    print(f"  {'Correctly refused':26s}{pct(b['correctly_refused']):>10s}{pct(f_['correctly_refused']):>14s}")
    print(f"  {'Hallucinated':26s}{pct(b['hallucinated']):>10s}{pct(f_['hallucinated']):>14s}")
    print("===========================================")

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {RESULT_FILE}")


if __name__ == "__main__":
    main()