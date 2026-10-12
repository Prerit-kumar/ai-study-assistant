import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from embed_store import collection, model

load_dotenv()
client = OpenAI(max_retries=3)

CHAT_MODEL = "gpt-4o-mini"
SELECT_MODEL = "gpt-4o"        # stronger model only for the paper-selection call

DIRECT_LIMIT = 8               # up to this many docs: single prompt
DIRECT_CHUNKS_PER_DOC = 6
EXTRACT_CHUNKS_PER_DOC = 5
PROFILE_WORDS = 350            # opening words of each paper (title + abstract) used for selection
SELECT_BATCH = 25              # papers per selection call
MAX_SELECTED = 15              # max rows in the final table
MAP_WORKERS = 6
NOT_FOUND = "I couldn't find this in the provided documents."

# ---- cost control ---------------------------------------------------------
# Every OpenAI answer is saved in llm_cache/. The same prompt is never paid for twice.
# Turn the cache off for one run (PowerShell):  $env:LLM_CACHE="0"
CACHE_ENABLED = os.getenv("LLM_CACHE", "1") != "0"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llm_cache")

# Approximate USD per 1M tokens: (input, cached input, output). Check OpenAI's pricing page if these change.
PRICES = {
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4o": (2.50, 1.25, 10.00),
}

_lock = threading.Lock()
_stats = {"hits": 0, "calls": 0, "cost": 0.0}


def _price_for(model_name):
    for name in sorted(PRICES, key=len, reverse=True):
        if model_name.startswith(name):
            return PRICES[name]
    return PRICES["gpt-4o"]


def _track_cost(model_name, usage):
    if usage is None:
        return
    prompt_tokens = usage.prompt_tokens or 0
    output_tokens = usage.completion_tokens or 0
    details = getattr(usage, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
    inp, cin, out = _price_for(model_name)
    cost = ((prompt_tokens - cached) * inp + cached * cin + output_tokens * out) / 1_000_000
    with _lock:
        _stats["cost"] += cost


def _chat(model_name, prompt, json_mode=False, temperature=0, max_tokens=None):
    """One OpenAI chat call with a disk cache. Returns the message text."""
    key_source = json.dumps([model_name, prompt, json_mode, temperature, max_tokens])
    key = hashlib.sha256(key_source.encode("utf-8")).hexdigest()
    path = os.path.join(CACHE_DIR, key + ".json")

    if CACHE_ENABLED and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                content = json.load(f)["content"]
            with _lock:
                _stats["hits"] += 1
            return content
        except Exception:
            pass  # broken cache file: call the API again

    kwargs = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    _track_cost(model_name, response.usage)
    with _lock:
        _stats["calls"] += 1

    if CACHE_ENABLED:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"model": model_name, "content": content}, f)
    return content


# ---- prompts --------------------------------------------------------------
# The selection prompt puts the fixed text (rules + papers) FIRST and the question LAST,
# so OpenAI's automatic prompt caching can reuse the long prefix across different questions.

SELECT_PROMPT = """You are selecting which research papers are relevant to a comparison question. The papers come first; the question is given at the end.

There are two kinds of questions:
- A question about a TOPIC or METHOD (for example "how do these papers reduce the cost of attention"): select papers whose own contribution addresses that topic.
- A question about an ATTRIBUTE that many papers report (for example model size, training data, datasets used, benchmark results): select every paper that presents its OWN model or method of the kind the question asks about and is likely to report that attribute for it, even if the attribute is not the paper's main topic.

Label each selected paper:
- "primary": the paper clearly fits the question as described above.
- "partial": the subject is one clear, named component the paper itself introduces for the same purpose the question is about, not just a mention.

Do NOT select:
- papers that only use, mention or build on the subject;
- papers that improve something else (for example parameter count, fine-tuning, training efficiency, retrieval or prompting when the question is about attention);
- papers where the same word appears with a different meaning (for example "retrieval" used as the name of an evaluation task, or "sparse" describing a different part of the model);
- papers about a different field or data type than the question names (for example, papers about text models do not answer a question about images). Never substitute a related or nearby topic.

If no paper fits, return an empty list.

Return JSON only, in this format:
{{"selected": [{{"doc": "<exact file name>", "relevance": "primary", "why": "<one short sentence>"}}]}}

Papers ({n}), each with its title and abstract:
{profiles}

Question: {question}
"""

EXTRACT_PROMPT = """You are extracting facts from a research paper for a comparison table.

Question: {question}

Why this paper is relevant: {why}

Excerpts from "{doc}":
{excerpts}

Extract 2-5 short facts from the excerpts that DIRECTLY answer the question for this paper. Choose the facts the question is asking about (for example the method's name, how it works, and the key numbers or results that matter for this question).

Rules:
- Every fact must answer what the question asks. General background about the paper does not count.
- If the excerpts contain nothing that directly answers the question, return an empty list.
- Report only facts about the model or method THIS paper itself introduces. Papers often list other models in comparison tables or related work (baselines, earlier versions, competitors). Do NOT report their numbers, even if they appear in the excerpts.
- If a number's owner is unclear from the excerpts, leave it out.
- Use only what the excerpts state. No outside knowledge.
- Copy formulas, complexities and numbers exactly as written in the excerpts.

Return JSON only:
{{"findings": ["...", "..."]}}
"""

REDUCE_PROMPT = """You are a research assistant. Below are facts extracted from several papers that were selected as relevant to the question.

Question: {question}

Facts per paper:
{findings}

Write a clear answer:
1. A markdown table with exactly one row per paper listed above. The first column must be "Paper" containing the file name. Choose the other columns to suit the question, and only create a column if the facts fill it for most papers. Put each row on its own line.
2. Then a short summary (2-3 sentences) of the main differences.

Rules:
- Use only the facts above. Do not invent information. If a fact is missing for a cell, write "Not stated".
- Put each fact in the column it actually belongs to.
- Report formulas, complexities and numbers exactly as written. Do not relabel or reinterpret them.
- The summary may only discuss the papers in the table, and every number in it must match the table exactly. Do not name or compare any other model.
- Do not use HTML tags such as <br>. Inside a table cell write one short line, separating items with semicolons.
"""

DIRECT_PROMPT = """You are a research assistant comparing several documents.

Below are excerpts from each document. Use them to answer the question. You may synthesize and compare across documents. Organize your answer so it's clear what comes from each document. If using a markdown table, make sure each row is on its own line with proper line breaks between rows. If a document's excerpts do not contain relevant information, say so for that document instead of guessing.

Report only facts about each document's own model or method, not the baselines it compares against. Report formulas, complexities and numbers exactly as they appear in the excerpts. Do not relabel or reinterpret them (for example, never call O(n log n) "linear").

{context}

Question: {question}

Answer:"""


class ComparisonFailed(Exception):
    """Raised when any OpenAI call fails, so we never show a partial table."""


def _friendly_error(err):
    text = str(err)
    if "insufficient_quota" in text or "credit_balance" in text:
        return ("The comparison could not run because the OpenAI account has no credits left. "
                "Add credits and try again.")
    if "429" in text or "rate_limit" in text.lower():
        return "OpenAI's rate limit was reached, so the comparison could not finish. Wait a minute and try again."
    return "A request to OpenAI failed, so the comparison could not be completed. Please try again."


# ---------------------------------------------------------------- helpers

def get_documents_in_store():
    data = collection.get(include=["metadatas"])
    return sorted({meta["source"] for meta in data["metadatas"]})


def _chunk_order(chunk_id):
    match = re.search(r"(\d+)$", str(chunk_id))
    return int(match.group(1)) if match else 0


def _normalize(vectors):
    vectors = np.array(vectors, dtype=float)
    if vectors.ndim == 1:
        return vectors / (np.linalg.norm(vectors) + 1e-10)
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)


def _load_doc(doc):
    """Returns (ordered_chunks, ordered_embeddings) for one document."""
    data = collection.get(where={"source": doc}, include=["documents", "embeddings"])
    chunks = data["documents"]
    if chunks is None or len(chunks) == 0:
        return [], None
    order = sorted(range(len(chunks)), key=lambda i: _chunk_order(data["ids"][i]))
    ordered_chunks = [chunks[i] for i in order]
    ordered_emb = _normalize([data["embeddings"][i] for i in order])
    return ordered_chunks, ordered_emb


def _profile(chunks, words=PROFILE_WORDS):
    """The opening of the paper: usually title + abstract."""
    return " ".join(" ".join(chunks[:2]).split()[:words])


def _top_chunks(chunks, emb, query_vec, k):
    scores = emb @ query_vec
    top = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in sorted(top)], float(scores[top[0]])


def _squash(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


COMPLEXITY_RE = re.compile(r"O\s*\([^()]*(?:\([^()]*\)[^()]*)*\)")


def _complexities_are_grounded(finding, excerpts_squashed):
    for expr in COMPLEXITY_RE.findall(finding):
        if _squash(expr) not in excerpts_squashed:
            return False, expr
    return True, None


def _is_separator(line):
    return set(line.strip()) <= set("|-: ")


def drop_empty_rows(answer, docs):
    """Removes table rows where every cell except the first says "Not stated".

    Returns (new_answer, removed_docs). Done in code so it works every time.
    """
    lines = answer.splitlines()
    kept_lines = []
    removed = []
    header_seen = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and not _is_separator(stripped):
            if not header_seen:
                header_seen = True          # first table line is the header
                kept_lines.append(line)
                continue
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            values = [c.lower() for c in cells[1:]]
            if values and all(v.startswith("not stated") or v in ("", "-", "n/a") for v in values):
                first = cells[0].lower()
                match = next(
                    (d for d in docs if first in (d.lower(), re.sub(r"\.pdf$", "", d.lower()))),
                    cells[0],
                )
                removed.append(match)
                continue
        kept_lines.append(line)
    return "\n".join(kept_lines), removed


# ---------------------------------------------------------------- small sets

def per_document_retrieval(query, documents, chunks_per_doc=3):
    """Return {document: [top chunks]}, ordered by how well each document matches."""
    query_vec = _normalize(model.encode(query))
    scored = []
    for doc in documents:
        chunks, emb = _load_doc(doc)
        if not chunks:
            continue
        top, best = _top_chunks(chunks, emb, query_vec, chunks_per_doc)
        scored.append((best, doc, top))
    scored.sort(key=lambda item: item[0], reverse=True)
    return {doc: top for _, doc, top in scored}


def compare_direct(query, retrieved):
    parts = [f"=== Document: {doc} ===\n" + "\n\n".join(chunks) for doc, chunks in retrieved.items()]
    prompt = DIRECT_PROMPT.format(context="\n\n".join(parts), question=query)
    return _chat(CHAT_MODEL, prompt, temperature=0.2)


# ---------------------------------------------------------------- large sets

def _match_name(name, lookup):
    key = re.sub(r"\.pdf$", "", str(name).strip().lower())
    return lookup.get(key)


def select_documents(query, profiles):
    """profiles: {doc: opening text}. Returns {doc: {"relevance", "why"}}.

    Raises ComparisonFailed if any selection call fails.
    """
    docs = list(profiles)
    lookup = {re.sub(r"\.pdf$", "", d.lower()): d for d in docs}
    batches = [docs[i:i + SELECT_BATCH] for i in range(0, len(docs), SELECT_BATCH)]

    def run(batch):
        text = "\n\n".join(f"=== {doc} ===\n{profiles[doc]}" for doc in batch)
        prompt = SELECT_PROMPT.format(question=query, n=len(batch), profiles=text)
        try:
            content = _chat(SELECT_MODEL, prompt, json_mode=True, temperature=0, max_tokens=1500)
            data = json.loads(content)
        except Exception as e:
            print(f"[select] batch failed: {e}")
            return {}, e

        picked = {}
        for item in data.get("selected", []):
            doc = _match_name(item.get("doc", ""), lookup)
            if doc is None or doc not in batch:
                print(f"[select] ignoring unknown name: {item.get('doc')!r}")
                continue
            relevance = "primary" if str(item.get("relevance", "")).lower() == "primary" else "partial"
            picked[doc] = {"relevance": relevance, "why": str(item.get("why", "")).strip()}
            print(f"[select] {relevance.upper():8} {doc}: {picked[doc]['why'][:100]}")
        return picked, None

    selected = {}
    errors = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for picked, error in pool.map(run, batches):
            selected.update(picked)
            if error is not None:
                errors.append(error)

    if errors:
        raise ComparisonFailed(str(errors[0]))
    return selected


def extract_from_doc(job):
    """Returns (doc, findings_list, error). error is None on success."""
    query, doc, why, excerpts = job
    prompt = EXTRACT_PROMPT.format(question=query, why=why, doc=doc, excerpts=excerpts)
    try:
        content = _chat(CHAT_MODEL, prompt, json_mode=True, temperature=0, max_tokens=600)
        data = json.loads(content)
    except Exception as e:
        print(f"[extract] failed for {doc}: {e}")
        return doc, [], e

    excerpts_squashed = _squash(excerpts)
    findings = []
    for item in data.get("findings", []):
        item = str(item).strip()
        if not item:
            continue
        ok, bad = _complexities_are_grounded(item, excerpts_squashed)
        if not ok:
            print(f"[extract] dropped unverified complexity in {doc}: {bad}")
            continue
        findings.append(item)
    if not findings:
        print(f"[extract] no direct answer in excerpts for {doc}")
    return doc, findings, None


def reduce_findings(query, findings):
    text = "\n\n".join(
        f"### {doc}\n" + "\n".join(f"- {f}" for f in facts)
        for doc, facts in findings.items()
    )
    prompt = REDUCE_PROMPT.format(question=query, findings=text)
    return _chat(CHAT_MODEL, prompt, temperature=0.2)


def _compare(query, documents, chunks_per_doc=None):
    # Small sets: one prompt with excerpts from every document.
    if len(documents) <= DIRECT_LIMIT:
        retrieved = per_document_retrieval(query, documents, chunks_per_doc or DIRECT_CHUNKS_PER_DOC)
        if not retrieved:
            return {"answer": NOT_FOUND, "documents_used": []}
        return {"answer": compare_direct(query, retrieved), "documents_used": list(retrieved.keys())}

    # Large sets: 1) select from abstracts  2) extract facts  3) build one table.
    loaded = {}
    for doc in documents:
        chunks, emb = _load_doc(doc)
        if chunks:
            loaded[doc] = (chunks, emb)
    if not loaded:
        return {"answer": NOT_FOUND, "documents_used": []}

    profiles = {doc: _profile(chunks) for doc, (chunks, _) in loaded.items()}
    selected = select_documents(query, profiles)

    primary = [d for d in selected if selected[d]["relevance"] == "primary"]
    partial = [d for d in selected if selected[d]["relevance"] == "partial"]
    print(f"[compare] {len(primary)} primary, {len(partial)} partial, of {len(loaded)} papers")

    if not selected:
        return {"answer": NOT_FOUND, "documents_used": []}

    # The table holds primary papers only. If there are none, fall back to partial ones.
    table_docs = primary if primary else partial
    side_docs = partial if primary else []
    kept, over_limit = table_docs[:MAX_SELECTED], table_docs[MAX_SELECTED:]

    jobs = []
    for doc in kept:
        chunks, emb = loaded[doc]
        query_vec = _normalize(model.encode(f"{query} {selected[doc]['why']}"))
        top, _ = _top_chunks(chunks, emb, query_vec, EXTRACT_CHUNKS_PER_DOC)
        excerpts = "\n\n".join([profiles[doc]] + top)
        jobs.append((query, doc, selected[doc]["why"], excerpts))

    with ThreadPoolExecutor(max_workers=MAP_WORKERS) as pool:
        outputs = list(pool.map(extract_from_doc, jobs))

    errors = [error for _, _, error in outputs if error is not None]
    if errors:
        raise ComparisonFailed(str(errors[0]))

    findings = {doc: facts for doc, facts, _ in outputs if facts}
    no_details = [doc for doc, facts, _ in outputs if not facts]

    if not findings:
        return {"answer": NOT_FOUND, "documents_used": []}

    answer = reduce_findings(query, findings)

    # Remove rows that answer nothing ("Not stated" everywhere), in code.
    answer, empty_docs = drop_empty_rows(answer, list(findings.keys()))
    if empty_docs:
        print(f"[compare] removed empty rows: {empty_docs}")
    used = [d for d in findings if d not in empty_docs]
    no_details = sorted(set(no_details) | set(empty_docs))

    if not used:
        return {"answer": NOT_FOUND, "documents_used": []}

    notes = []
    if not primary:
        notes.append("No paper has this topic as its main contribution; the table shows papers where it is one component.")
    if side_docs:
        lines = "; ".join(f"{d} ({selected[d]['why']})" for d in sorted(side_docs))
        notes.append(f"Possibly related, not in the table ({len(side_docs)}): {lines}")
    if no_details:
        notes.append(
            f"Selected, but the retrieved excerpts did not answer the question ({len(no_details)}): "
            + ", ".join(no_details)
        )
    if over_limit:
        notes.append(f"Left out to keep the table readable ({len(over_limit)}): " + ", ".join(over_limit))
    not_selected = sorted(set(loaded) - set(selected))
    notes.append(f"Not selected ({len(not_selected)}): " + ", ".join(not_selected))

    answer += "\n\n---\n" + "\n\n".join(f"*{n}*" for n in notes)
    return {"answer": answer, "documents_used": used}


def compare_documents_detailed(query, documents, chunks_per_doc=None):
    """Returns {"answer": str, "documents_used": [doc, ...]}.

    If any OpenAI call fails, returns a clear error message (and an "error" key)
    instead of a partial table. Prints the estimated cost of each comparison.
    """
    with _lock:
        _stats.update({"hits": 0, "calls": 0, "cost": 0.0})
    try:
        result = _compare(query, documents, chunks_per_doc)
    except Exception as e:
        print(f"[compare] FAILED: {e}")
        result = {"answer": _friendly_error(e), "documents_used": [], "error": str(e)}
    print(
        f"[cost] ~${_stats['cost']:.4f} | cache hits {_stats['hits']}, new calls {_stats['calls']}"
        + ("" if CACHE_ENABLED else " (cache OFF)")
    )
    return result


def compare_documents(query, documents, chunks_per_doc=None):
    """Backward-compatible wrapper that returns only the answer text."""
    return compare_documents_detailed(query, documents, chunks_per_doc)["answer"]