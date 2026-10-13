"""Ask-mode evaluation on the 40-paper corpus (no LLM judge, so scoring is free).

Usage (from the src folder, venv active):
    python evaluation.py

Answerable questions check that
  - the expected paper appears in the cited Sources, and
  - the answer contains the key fact (each inner list is "any of these").
Unanswerable questions check that the system refuses instead of guessing.
Cost: about $0.01-0.02 per run with gpt-4o-mini.
"""

import json
import time

from rag_pipeline import generate_answer

TOP_K = 8  # same as the app's default
NOT_FOUND = "couldn't find"
API_ERROR_MARKERS = ["could not be generated", "rate limit was reached"]

# facts: every inner list must be matched by at least one of its strings (case-insensitive)
ANSWERABLE = [
    {"q": "How many parameters does GPT-3 have?", "source": "GPT3_Few_Shot_Learners.pdf", "facts": [["175"]]},
    {"q": "How many training tokens was Chinchilla trained on?", "source": "Chinchilla.pdf", "facts": [["1.4"]]},
    {"q": "By what factor does LoRA reduce the number of trainable parameters compared to full fine-tuning of GPT-3 175B?",
     "source": "LoRA.pdf", "facts": [["10,000", "10000"]]},
    {"q": "What new 4-bit data type does QLoRA introduce?", "source": "QLoRA.pdf", "facts": [["normalfloat", "nf4"]]},
    {"q": "How much smaller and faster is DistilBERT compared to BERT?", "source": "DistilBERT.pdf",
     "facts": [["40%", "40 %"], ["60%", "60 %"]]},
    {"q": "What BLEU score did the Transformer achieve on the WMT 2014 English-to-German translation task?",
     "source": "Attention_Is_All_You_Need.pdf", "facts": [["28.4"]]},
    {"q": "What are the two pre-training tasks used by BERT?", "source": "BERT.pdf",
     "facts": [["masked"], ["next sentence"]]},
    {"q": "How does the attention cost of Longformer scale with sequence length?", "source": "Longformer.pdf",
     "facts": [["linear"]]},
    {"q": "What technique does Reformer use to replace dot-product attention?", "source": "Reformer.pdf",
     "facts": [["locality-sensitive", "locality sensitive", "lsh"]]},
    {"q": "Which two attention mechanisms does Mistral 7B use?", "source": "Mistral_7B.pdf",
     "facts": [["grouped-query", "grouped query", "gqa"], ["sliding window", "swa"]]},
    {"q": "How many total parameters does Mixtral have, and how many are active per token?", "source": "Mixtral.pdf",
     "facts": [["47"], ["13"]]},
    {"q": "On how many tokens was Llama 2 pretrained?", "source": "Llama2.pdf",
     "facts": [["2 trillion", "2t", "2.0t", "2.0 trillion"]]},
    {"q": "Which BERT pre-training objective did RoBERTa remove?", "source": "RoBERTa.pdf",
     "facts": [["next sentence"]]},
    {"q": "What pre-training task does ELECTRA use instead of masked language modeling?", "source": "ELECTRA.pdf",
     "facts": [["replaced token detection"]]},
    {"q": "What two parameter-reduction techniques does ALBERT use?", "source": "ALBERT.pdf",
     "facts": [["factorized", "factorization"], ["cross-layer", "parameter sharing"]]},
    {"q": "What model architecture does Dense Passage Retrieval use to encode questions and passages?",
     "source": "DPR_Dense_Passage_Retrieval.pdf", "facts": [["dual-encoder", "dual encoder", "bi-encoder"]]},
    {"q": "What interaction mechanism does ColBERT use between queries and documents?", "source": "ColBERT.pdf",
     "facts": [["late interaction"]]},
    {"q": "What two kinds of memory does RAG (retrieval-augmented generation) combine?",
     "source": "RAG_Retrieval_Augmented_Generation.pdf", "facts": [["non-parametric", "nonparametric"]]},
    {"q": "What is the key idea that makes FlashAttention fast?", "source": "FlashAttention.pdf",
     "facts": [["io-aware", "tiling", "memory reads"]]},
    {"q": "What is chain-of-thought prompting?", "source": "Chain_of_Thought.pdf",
     "facts": [["intermediate reasoning", "reasoning steps"]]},
    {"q": "What does the ReAct approach combine in language models?", "source": "ReAct.pdf",
     "facts": [["reasoning"], ["action", "acting"]]},
    {"q": "How does Toolformer learn when to call external tools?", "source": "Toolformer.pdf",
     "facts": [["self-supervised", "api"]]},
    {"q": "What does Direct Preference Optimization avoid compared to standard RLHF?", "source": "DPO.pdf",
     "facts": [["reward model", "reinforcement learning"]]},
    {"q": "How many parameters does the InstructGPT model have that human labelers preferred over the 175B GPT-3?",
     "source": "InstructGPT.pdf", "facts": [["1.3"]]},
    {"q": "Where in a long context do language models use relevant information best?",
     "source": "Lost_in_the_Middle.pdf", "facts": [["beginning", "start"], ["end"]]},
    {"q": "What special tokens does Self-RAG generate?", "source": "Self_RAG.pdf", "facts": [["reflection"]]},
    {"q": "What type of model is Mamba built on?", "source": "Mamba.pdf", "facts": [["state space", "ssm"]]},
    {"q": "How does language model loss scale with model size according to the scaling laws paper?",
     "source": "Scaling_Laws.pdf", "facts": [["power law", "power-law"]]},
    {"q": "What framework does T5 use to treat every NLP problem?", "source": "T5.pdf", "facts": [["text-to-text"]]},
    {"q": "What two techniques does Transformer-XL introduce?", "source": "Transformer_XL.pdf",
     "facts": [["recurrence"], ["positional"]]},
]

UNANSWERABLE = [
    "Who won the 2022 FIFA World Cup?",
    "What is the capital of Australia?",
    "How do I bake sourdough bread at home?",
    "What is the boiling point of mercury?",
    "What was NVIDIA's stock price at the end of 2023?",
]


def is_api_error(answer):
    return any(marker in answer.lower() for marker in API_ERROR_MARKERS)


def facts_found(answer, facts):
    text = answer.lower()
    return all(any(option.lower() in text for option in group) for group in facts)


def ask(question):
    start = time.time()
    answer, sources = generate_answer(question, top_k=TOP_K)
    return answer, set(sources), round(time.time() - start, 2)


def run():
    results = {"answerable": [], "unanswerable": []}

    print(f"Answerable questions ({len(ANSWERABLE)}):")
    for item in ANSWERABLE:
        answer, sources, seconds = ask(item["q"])
        if is_api_error(answer):
            return None, answer
        refused = NOT_FOUND in answer.lower()
        r = {
            "question": item["q"],
            "expected_source": item["source"],
            "sources": sorted(sources),
            "source_ok": item["source"] in sources,
            "facts_ok": (not refused) and facts_found(answer, item["facts"]),
            "refused": refused,
            "seconds": seconds,
            "answer": answer,
        }
        results["answerable"].append(r)
        mark = "PASS" if r["source_ok"] and r["facts_ok"] else "FAIL"
        detail = []
        if not r["source_ok"]:
            detail.append(f"source missing (got: {', '.join(r['sources']) or 'none'})")
        if refused:
            detail.append("wrongly refused")
        elif not r["facts_ok"]:
            detail.append("key fact missing")
        print(f"  [{mark}] {item['q'][:70]}  {'; '.join(detail)}  ({seconds}s)")

    print(f"\nUnanswerable questions ({len(UNANSWERABLE)}):")
    for q in UNANSWERABLE:
        answer, sources, seconds = ask(q)
        if is_api_error(answer):
            return None, answer
        refused = NOT_FOUND in answer.lower()
        results["unanswerable"].append({
            "question": q, "refused": refused, "sources": sorted(sources), "seconds": seconds, "answer": answer,
        })
        print(f"  [{'PASS' if refused else 'FAIL - hallucinated'}] {q}  ({seconds}s)")

    return results, None


def summary(results):
    ans = results["answerable"]
    una = results["unanswerable"]
    n = len(ans)
    both = sum(1 for r in ans if r["source_ok"] and r["facts_ok"])
    times = sorted(r["seconds"] for r in ans + una)

    print("\n=== SUMMARY ===")
    print(f"Correct source cited:     {sum(r['source_ok'] for r in ans)}/{n}")
    print(f"Key fact in answer:       {sum(r['facts_ok'] for r in ans)}/{n}")
    print(f"Fully correct (both):     {both}/{n}  ({both / n:.0%})")
    print(f"Wrongly refused:          {sum(r['refused'] for r in ans)}/{n}")
    print(f"Correctly refused:        {sum(r['refused'] for r in una)}/{len(una)}")
    print(f"Hallucinated (should refuse): {sum(not r['refused'] for r in una)}/{len(una)}")
    print(f"Latency: median {times[len(times) // 2]:.1f}s, slowest {times[-1]:.1f}s")


if __name__ == "__main__":
    results, error = run()
    if error:
        print("\nRUN INVALID: an OpenAI request failed, so no scores are reported.")
        print(error)
    else:
        summary(results)
        with open("ask_eval_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("\nFull answers saved to ask_eval_results.json")