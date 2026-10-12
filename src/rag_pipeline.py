import os

from dotenv import load_dotenv
from openai import OpenAI

from hybrid_search import hybrid_search

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=3)

USED_MARKER = "USED_CHUNKS:"


def _friendly_error(err):
    text = str(err)
    if "insufficient_quota" in text or "credit_balance" in text:
        return ("The answer could not be generated because the OpenAI account has no credits left. "
                "Add credits and try again.")
    if "429" in text or "rate_limit" in text.lower():
        return "OpenAI's rate limit was reached. Wait a minute and try again."
    return "A request to OpenAI failed, so the answer could not be generated. Please try again."


def generate_answer(query, top_k=5):
    # Step 1: Retrieve relevant chunks using hybrid search + reranking
    results = hybrid_search(query, top_k=top_k)

    # Step 2: Label each chunk so the model can tell us which ones it used
    labeled_chunks = []
    for i, (chunk, source) in enumerate(results, start=1):
        labeled_chunks.append(f"[{i}] (from {source})\n{chunk}")

    context = "\n\n---\n\n".join(labeled_chunks)

    # Step 3: Build the prompt
    prompt = f"""You are a helpful research assistant. Use the context below to answer the question.
You may synthesize and compare information across different parts of the context, even if the comparison itself isn't explicitly written out — as long as each individual fact comes from the context.
If the context genuinely doesn't contain the information needed, say "I couldn't find this in the provided documents."

After your answer, add one final line in exactly this format:
{USED_MARKER} 1, 3
List only the numbers of the context chunks whose information you actually used. If your answer is "I couldn't find this in the provided documents.", write {USED_MARKER} NONE.
Do not mention chunk numbers anywhere else in your answer.

Context:
{context}

Question: {query}

Answer:"""

    # Step 4: Call the LLM. On any OpenAI error, return a clear message and no sources
    # instead of crashing the request (which the frontend would show as "backend unreachable").
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
    except Exception as e:
        print(f"[ask] OpenAI call failed: {e}")
        return _friendly_error(e), set()

    raw = response.choices[0].message.content.strip()

    # Step 5: Separate the answer from the list of chunks the model used
    idx = raw.upper().rfind(USED_MARKER)
    if idx != -1:
        answer = raw[:idx].strip()
        tail = raw[idx + len(USED_MARKER):]

        used_numbers = set()
        for token in tail.replace(",", " ").split():
            digits = "".join(ch for ch in token if ch.isdigit())
            if digits:
                used_numbers.add(int(digits))

        sources_used = {
            results[n - 1][1]
            for n in used_numbers
            if 1 <= n <= len(results)
        }
    else:
        # Model ignored the format: fall back to all retrieved sources
        answer = raw
        sources_used = {source for _, source in results}

    return answer, sources_used


if __name__ == "__main__":
    query = "Compare entity resolution with n-gram language modeling. How are they different approaches?"
    answer, sources = generate_answer(query, top_k=5)

    print(f"Question: {query}\n")
    print(f"Answer:\n{answer}\n")
    print(f"Sources used: {', '.join(sources)}")