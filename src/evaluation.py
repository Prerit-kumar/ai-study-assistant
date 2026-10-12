import time
from rag_pipeline import generate_answer

TEST_QUESTIONS = [
    {"question": "What is entity resolution?", "expected_source": "test.pdf"},
    {"question": "What is the Markov assumption in language modeling?", "expected_source": "test3.pdf"},
    {"question": "What is the F_0.5 score used for?", "expected_source": "test.pdf"},
    {"question": "What is Kneser-Ney smoothing?", "expected_source": "test3.pdf"},
    {"question": "What is text normalization in NLP?", "expected_source": "test2.pdf"},
    {"question": "What is vector semantics?", "expected_source": "test4.pdf"},
]


def run_evaluation():
    results = []
    total_time = 0

    for item in TEST_QUESTIONS:
        question = item["question"]
        expected_source = item["expected_source"]

        start_time = time.time()
        answer, sources = generate_answer(question, top_k=5)
        elapsed = time.time() - start_time
        total_time += elapsed

        # Check if the expected source is among the sources used
        retrieval_correct = expected_source in sources

        # Check if the model said it couldn't find the answer (possible failure)
        said_not_found = "couldn't find" in answer.lower()

        results.append({
            "question": question,
            "expected_source": expected_source,
            "sources_used": sources,
            "retrieval_correct": retrieval_correct,
            "said_not_found": said_not_found,
            "time_seconds": round(elapsed, 2),
            "answer_preview": answer[:150]
        })

    return results, total_time


def print_report(results, total_time):
    correct_count = sum(1 for r in results if r["retrieval_correct"])
    not_found_count = sum(1 for r in results if r["said_not_found"])
    avg_time = total_time / len(results)

    print("=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    for i, r in enumerate(results, 1):
        status = "✅ PASS" if r["retrieval_correct"] else "❌ FAIL"
        print(f"\n[{i}] {status} — {r['question']}")
        print(f"    Expected: {r['expected_source']} | Got: {', '.join(r['sources_used'])}")
        print(f"    Time: {r['time_seconds']}s | Said 'not found': {r['said_not_found']}")

    print("\n" + "=" * 60)
    print(f"SUMMARY: {correct_count}/{len(results)} retrieval-correct ({correct_count/len(results)*100:.0f}%)")
    print(f"Answers saying 'couldn't find': {not_found_count}/{len(results)}")
    print(f"Average response time: {avg_time:.2f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    results, total_time = run_evaluation()
    print_report(results, total_time)