import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import OpenAI

from hybrid_search import hybrid_search

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=3)

USED_MARKER = "USED_CHUNKS:"
NOT_FOUND = "I couldn't find this in the provided documents."

# The fine-tuned model runs in its own small server (finetune/model_server.py, on the GPU).
LOCAL_MODEL_URL = os.getenv("LOCAL_MODEL_URL", "http://127.0.0.1:8001")
LOCAL_CONTEXT_CHUNKS = 5  # the fine-tuned model was trained with 5 chunks per question

# Exactly the system prompt the fine-tuned model was trained with (make_dataset_v2.py).
LOCAL_SYSTEM_PROMPT = (
    "You are a research assistant. Answer the question using ONLY the numbered context chunks. "
    f'If the context does not contain the answer, reply exactly: "{NOT_FOUND}" '
    f"After your answer, add one final line in exactly this format: {USED_MARKER} followed by the "
    f"numbers of the chunks you used (for example: {USED_MARKER} 2), or {USED_MARKER} NONE if you "
    "could not answer."
)


def _friendly_error(err):
    text = str(err)
    if "insufficient_quota" in text or "credit_balance" in text:
        return ("The answer could not be generated because the OpenAI account has no credits left. "
                "Add credits and try again.")
    if "429" in text or "rate_limit" in text.lower():
        return "OpenAI's rate limit was reached. Wait a minute and try again."
    return "A request to OpenAI failed, so the answer could not be generated. Please try again."


def _context(results):
    labeled = [f"[{i}] (from {source})\n{chunk}" for i, (chunk, source) in enumerate(results, start=1)]
    return "\n\n---\n\n".join(labeled)


def _split(raw, results):
    """Separates the answer from its USED_CHUNKS line. Returns (answer, cited chunk numbers or None)."""
    idx = raw.upper().rfind(USED_MARKER)
    if idx == -1:
        return raw.strip(), None
    numbers = set()
    for token in raw[idx + len(USED_MARKER):].replace(",", " ").split():
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits and 1 <= int(digits) <= len(results):
            numbers.add(int(digits))
    return raw[:idx].strip(), numbers


def is_refusal(answer):
    return "couldn't find this in the provided documents" in answer.lower().replace("\u2019", "'")


def ask_gpt(query, results):
    """Returns (answer, cited chunk numbers or None, failed?)."""
    prompt = f"""You are a helpful research assistant. Use the context below to answer the question.
You may synthesize and compare information across different parts of the context, even if the comparison itself isn't explicitly written out — as long as each individual fact comes from the context.
If the context genuinely doesn't contain the information needed, say "{NOT_FOUND}"

After your answer, add one final line in exactly this format:
{USED_MARKER} 1, 3
List only the numbers of the context chunks whose information you actually used. If your answer is "{NOT_FOUND}", write {USED_MARKER} NONE.
Do not mention chunk numbers anywhere else in your answer.

Context:
{_context(results)}

Question: {query}

Answer:"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
    except Exception as e:
        print(f"[ask] OpenAI call failed: {e}")
        return _friendly_error(e), set(), True
    answer, numbers = _split(response.choices[0].message.content.strip(), results)
    return answer, numbers, False


def ask_local(query, results):
    """Asks the fine-tuned model. Returns (answer, cited chunk numbers or None, seconds), or None if it isn't running."""
    messages = [
        {"role": "system", "content": LOCAL_SYSTEM_PROMPT},
        {"role": "user", "content": "Context:\n" + _context(results) + f"\n\nQuestion: {query}"},
    ]
    request = urllib.request.Request(
        f"{LOCAL_MODEL_URL}/generate",
        data=json.dumps({"messages": messages}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            data = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        print(f"[ask] local model unavailable: {e}")
        return None
    answer, numbers = _split(data["answer"], results)
    return answer, numbers, data.get("seconds")


def local_model_available():
    try:
        with urllib.request.urlopen(f"{LOCAL_MODEL_URL}/health", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def _sources(numbers, results):
    return sorted({results[n - 1][1] for n in (numbers or set())})


def generate_answer(query, top_k=5, sources=None):
    """GPT only (the default). Returns (answer, set of source documents).
    `sources`: only search these documents (None = all)."""
    results = hybrid_search(query, top_k=top_k, sources=sources)
    if not results:
        return NOT_FOUND, set()
    answer, numbers, failed = ask_gpt(query, results)
    if failed:
        return answer, set()
    if numbers is None:  # model ignored the format: fall back to all retrieved sources
        return answer, {source for _, source in results}
    return answer, set(_sources(numbers, results))


def generate_answer_both(query, sources=None):
    """Asks GPT and the fine-tuned model the same question with the same chunks, and compares them.

    Verdicts (the fine-tuned model hallucinated least in our benchmark, so it acts as a second opinion):
      agree             both answered and cited at least one same chunk
      differ            both answered from different chunks
      warning           GPT answered but the fine-tuned model found nothing: GPT may not be supported
      local_only        GPT refused or failed, the fine-tuned model answered
      not_found         neither model found it
      local_unavailable the model server isn't running; GPT answer only
    """
    results = hybrid_search(query, top_k=LOCAL_CONTEXT_CHUNKS, sources=sources)
    if not results:
        empty = {"answer": NOT_FOUND, "sources": [], "refused": True}
        return {"gpt": empty, "local": None, "verdict": "not_found", "best": None}
    start = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        gpt_future = pool.submit(ask_gpt, query, results)
        local_future = pool.submit(ask_local, query, results)
        gpt_answer, gpt_numbers, gpt_failed = gpt_future.result()
        local = local_future.result()

    gpt_refused = gpt_failed or is_refusal(gpt_answer)
    gpt = {
        "answer": gpt_answer,
        "sources": [] if gpt_refused else _sources(gpt_numbers if gpt_numbers is not None else range(1, len(results) + 1), results),
        "refused": gpt_refused,
    }

    if local is None:
        verdict, best, local_part = "local_unavailable", "gpt", None
    else:
        local_answer, local_numbers, seconds = local
        local_refused = is_refusal(local_answer)
        local_part = {
            "answer": local_answer,
            "sources": [] if local_refused else _sources(local_numbers, results),
            "refused": local_refused,
            "seconds": seconds,
        }
        if gpt_refused and local_refused:
            verdict, best = "not_found", None
        elif local_refused:
            verdict, best = "warning", None
        elif gpt_refused:
            verdict, best = "local_only", "local"
        elif (gpt_numbers or set()) & (local_numbers or set()):
            verdict, best = "agree", "gpt"
        else:
            verdict, best = "differ", "gpt"

    print(f"[ask/both] verdict={verdict} in {time.time() - start:.1f}s")
    return {"gpt": gpt, "local": local_part, "verdict": verdict, "best": best}


if __name__ == "__main__":
    query = "How many parameters does GPT-3 have?"
    answer, sources = generate_answer(query, top_k=5)
    print(f"Question: {query}\n\nAnswer:\n{answer}\n\nSources used: {', '.join(sources)}")
