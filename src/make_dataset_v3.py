"""Fine-tuning dataset v3: teach the model to work on ANY document, not only research papers.

Why: v2 was trained only on research-paper questions. On other PDFs (a blood report, a bill, a
boarding pass) it refused even when the answer was right there, so the two-model check gave false alarms.

What v3 adds (the format stays exactly the same as v2, so model_server.py and the app need no change):
  1. About 200 invented everyday documents written by gpt-4o-mini: lab reports, invoices, tickets,
     marksheets, timetables, manuals, minutes... All names and numbers are made up, no real person's data.
  2. About a quarter of them are drawn as page images and read back with the same OCR the app uses,
     so the model also learns from messy OCR text.
  3. New kinds of questions: values from tables, simple comparisons, short summaries, and
     "plausible but not in the document" questions that must be refused.
  4. About 1,200 examples from v2 (research papers) are mixed back in, so the model doesn't forget papers.
  5. Test set: 30 unseen documents, including 4 document types never seen in training.
     The old v2 paper test set is kept, to check that paper performance didn't drop.

Run from the src folder with the project venv (about 20-30 minutes, well under $1 of gpt-4o-mini):
    python make_dataset_v3.py
If it stops halfway, run it again: documents and questions already made are reused (saved in ../finetune_data_v3).
"""

import json
import os
import random
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from chunker import chunk_text
from embed_store import model
from hybrid_search import reranker

load_dotenv()
client = OpenAI(max_retries=3)

OUT_DIR = "../finetune_data_v3"
V2_TRAIN = "../finetune_data_v2/train.jsonl"
SEED = 42
WORKERS = 8

DOCS_PER_TRAIN_TYPE = 7          # x 26 types = 182 training documents
TEST_DOCS_PER_SEEN_TYPE = 1      # 18 test documents of types seen in training
TEST_DOCS_PER_UNSEEN_TYPE = 3    # x 4 held-out types = 12 test documents
CHUNKS_PER_TRAIN_DOC = 2
CHUNKS_PER_TEST_DOC = 1
OCR_SHARE = 0.25                 # share of documents passed through OCR
V2_EXAMPLES = 1200               # research-paper examples mixed back in
CONTEXT_CHUNKS = 5
CANDIDATES = 15

NOT_FOUND = "I couldn't find this in the provided documents."
USED_MARKER = "USED_CHUNKS:"

# Same system prompt as v2 and as rag_pipeline.LOCAL_SYSTEM_PROMPT. Do not change it.
SYSTEM_PROMPT = (
    "You are a research assistant. Answer the question using ONLY the numbered context chunks. "
    f'If the context does not contain the answer, reply exactly: "{NOT_FOUND}" '
    f"After your answer, add one final line in exactly this format: {USED_MARKER} followed by the "
    f"numbers of the chunks you used (for example: {USED_MARKER} 2), or {USED_MARKER} NONE if you "
    "could not answer."
)

TRAIN_TYPES = [
    "blood test / pathology lab report with a results table (test, value, reference range, unit)",
    "shop or service invoice with line items, quantities, taxes (GST) and total",
    "flight boarding pass and e-ticket with passenger, flight, seat, gate and baggage details",
    "university semester marksheet / grade card with subjects, credits, grades and SGPA",
    "weekly college class timetable with days, times, rooms and teachers",
    "product user manual section with setup steps, specifications table and troubleshooting",
    "company meeting minutes with attendees, decisions and action items with owners and dates",
    "monthly salary slip with earnings, deductions and net pay",
    "residential rental agreement with parties, rent, deposit, term and clauses",
    "hotel booking confirmation with dates, room type, price breakdown and cancellation policy",
    "hospital discharge summary with diagnosis, treatment, medicines and follow-up",
    "doctor's prescription with medicines, doses, timing and duration",
    "insurance policy schedule with insured details, sum insured, premium and exclusions",
    "news article about a local event",
    "company HR leave and attendance policy",
    "quarterly business report with a revenue and expenses table and commentary",
    "event or conference schedule with sessions, speakers and halls",
    "restaurant menu with sections, items and prices",
    "job offer letter with role, salary components, joining date and conditions",
    "bank account statement with dated transactions, debits, credits and balance",
    "school circular / notice to parents with dates and instructions",
    "recipe with ingredients table, servings and steps",
    "government scheme notification with eligibility, benefits and deadlines",
    "laboratory experiment report with aim, apparatus, observations table and result",
    "car service record with parts replaced, labour charges and next service due",
    "internship completion certificate and evaluation form with ratings",
]
# Never used for training: they test whether the model generalises to new kinds of documents.
TEST_ONLY_TYPES = [
    "electricity bill with meter readings, units consumed, slab charges and due date",
    "train ticket (Indian Railways style) with PNR, coach, berth, stations and fare",
    "product warranty card with model, serial number, warranty period and terms",
    "course syllabus with units, topics, hours, assessment weightage and textbooks",
]

DOC_PROMPT = """Write the full text of one realistic document of this type:
{doc_type}

Variety for this one: {variety}

Rules:
- Write it the way the text looks when copied out of a PDF: a title, headings, short paragraphs, and tables
  written as rows with columns separated by spaces (no Markdown, no | pipes, no ** bold).
- Use specific, varied details: names, dates, IDs, amounts, units. Everything must be INVENTED (fictional people
  and organisations, fake numbers). Never use real people.
- Include a few notable details someone might ask about (for example a value outside its normal range, a deadline,
  a penalty, a highest/lowest item).
- Length: {words} words.

Return JSON only: {{"title": "...", "text": "..."}}"""

QA_PROMPT = """You are creating training data for a model that answers questions about documents.

Full document (for your reference):
\"\"\"
{document}
\"\"\"

The model will be shown only this PART of the document:
\"\"\"
{chunk}
\"\"\"

Write 4 questions a person who uploaded this document might ask:
1. "lookup": a specific value or detail. The answer must be fully supported by the PART.
2. "reasoning": a simple comparison or check that uses only facts in the PART (which is highest, is X above its
   range, how many days between two dates, what is the difference). Give the answer with the facts it relies on.
3. "summary": what the PART says about something, in 1-3 sentences, supported only by the PART.
4. "absent": a natural question someone could ask about this kind of document whose answer is NOT anywhere in the
   FULL document (not even partly). It must sound plausible, not silly.

Rules: answers 1-3 sentences, only facts from the PART, no outside knowledge or medical/legal advice.
Questions must be natural ("What is my hemoglobin level?", "When is the rent due?"), never mention "the passage" or "the chunk".

Return JSON only:
{{"pairs": [{{"kind": "lookup", "question": "...", "answer": "..."}}, {{"kind": "reasoning", "question": "...", "answer": "..."}},
{{"kind": "summary", "question": "...", "answer": "..."}}, {{"kind": "absent", "question": "..."}}]}}"""

FIRST_NAMES = ["Aarav", "Diya", "Rohan", "Sneha", "Kabir", "Meera", "Arjun", "Fatima", "Vikram", "Ananya",
               "John", "Maria", "Chen", "Yuki", "Omar", "Elena", "Tunde", "Priya", "Rahul", "Neha"]
CITIES = ["Mumbai", "Chennai", "Delhi", "Bengaluru", "Pune", "Kolkata", "Hyderabad", "Jaipur", "Lucknow",
          "Kochi", "Singapore", "Dubai", "London", "Toronto", "Sydney"]
STYLES = ["formal and detailed", "short and dense", "with many numbers", "with several sections", "plain and simple"]


# ---------------------------------------------------------------- 1. documents

def make_document(job):
    doc_id, doc_type, split, rng_seed = job
    rng = random.Random(rng_seed)
    variety = f"main person {rng.choice(FIRST_NAMES)}, city {rng.choice(CITIES)}, style {rng.choice(STYLES)}"
    words = rng.choice(["300-450", "450-650", "650-900"])
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": DOC_PROMPT.format(doc_type=doc_type, variety=variety, words=words)}],
            response_format={"type": "json_object"},
            temperature=1.0,
        )
        data = json.loads(response.choices[0].message.content)
        text = str(data.get("text", "")).strip()
        title = str(data.get("title", "")).strip() or doc_type.split(" with ")[0]
    except Exception as e:
        print(f"   document failed ({doc_type[:40]}): {e}")
        return None
    if len(text.split()) < 150:
        return None
    return {"id": doc_id, "type": doc_type, "split": split, "title": title, "text": text, "ocr": False}


def _font(size):
    from PIL import ImageFont
    for path in [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\consola.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def ocr_document(text):
    """Draw the text as page images and read it back with the app's OCR, so it has real OCR mistakes."""
    from PIL import Image, ImageDraw
    from pdf_reader import _ocr_lines

    # Large text (like a page scanned at 200 dpi): with small text, OCR drops the spaces between words.
    font = _font(36)
    lines = []
    for line in text.splitlines():
        lines.extend(textwrap.wrap(line, 70) or [""])
    pages = [lines[i:i + 38] for i in range(0, len(lines), 38)]
    out = []
    for page_lines in pages:
        img = Image.new("RGB", (1700, 2200), "white")
        draw = ImageDraw.Draw(img)
        for i, line in enumerate(page_lines):
            draw.text((90, 100 + i * 52), line, fill="black", font=font)
        out.append(_ocr_lines(np.array(img)))
    return "\n".join(out)


def top_up_ocr(docs, path):
    """If fewer documents than planned went through OCR, OCR more of them. Returns the ids that changed."""
    from pdf_reader import _fix_spacing

    changed = set()
    # Documents OCR'd before the spacing fix existed: split their glued words now.
    for doc in docs:
        if doc["ocr"]:
            fixed = "\n".join(_fix_spacing(line) for line in doc["text"].splitlines())
            if fixed != doc["text"]:
                doc["text"] = fixed
                changed.add(doc["id"])

    target = round(OCR_SHARE * len(docs))
    have = sum(d["ocr"] for d in docs)
    if have >= target and not changed:
        return set()
    candidates = [d for d in docs if not d["ocr"]]
    random.Random(SEED + 1).shuffle(candidates)
    if have < target:
        print(f"Only {have} documents went through OCR; reading more with OCR to reach about {target} ...")
    for doc in candidates:
        if have >= target:
            break
        try:
            ocr_text = ocr_document(doc["text"])
        except Exception as e:
            print(f"   OCR failed for {doc['id']}: {e}")
            continue
        if len(ocr_text.split()) > 0.6 * len(doc["text"].split()):
            doc["text"], doc["ocr"] = ocr_text, True
            changed.add(doc["id"])
            have += 1
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"Now {have} documents went through OCR ({len(changed)} changed; their questions will be rewritten)")
    return changed


def load_or_make_documents(rng):
    path = os.path.join(OUT_DIR, "documents.jsonl")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            docs = [json.loads(line) for line in f]
        print(f"Reusing {len(docs)} documents from {path}")
        return docs, top_up_ocr(docs, path)

    jobs = []
    for t, doc_type in enumerate(TRAIN_TYPES):
        for k in range(DOCS_PER_TRAIN_TYPE):
            jobs.append((f"train-{t:02d}-{k}", doc_type, "train", rng.random()))
        for k in range(TEST_DOCS_PER_SEEN_TYPE):
            if t < 18:
                jobs.append((f"test-{t:02d}-{k}", doc_type, "test", rng.random()))
    for t, doc_type in enumerate(TEST_ONLY_TYPES):
        for k in range(TEST_DOCS_PER_UNSEEN_TYPE):
            jobs.append((f"test-new{t}-{k}", doc_type, "test", rng.random()))

    print(f"Writing {len(jobs)} invented documents with gpt-4o-mini ...")
    docs = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, fut in enumerate(as_completed([pool.submit(make_document, j) for j in jobs]), start=1):
            doc = fut.result()
            if doc:
                docs.append(doc)
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)} documents")
    docs.sort(key=lambda d: d["id"])

    print(f"Passing about {int(OCR_SHARE * 100)}% of documents through OCR (local, a few minutes) ...")
    for doc in docs:
        if rng.random() < OCR_SHARE:
            try:
                ocr_text = ocr_document(doc["text"])
            except Exception as e:
                print(f"   OCR failed for {doc['id']}: {e}")
                continue
            if len(ocr_text.split()) > 0.6 * len(doc["text"].split()):
                doc["text"], doc["ocr"] = ocr_text, True

    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"Saved {len(docs)} documents ({sum(d['ocr'] for d in docs)} through OCR)")
    return docs, top_up_ocr(docs, path)


# ---------------------------------------------------------------- 2. questions

def make_questions(job):
    key, doc, chunk = job
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": QA_PROMPT.format(document=doc["text"][:7000], chunk=chunk)}],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        data = json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"   questions failed for {doc['id']}: {e}")
        return key, []
    pairs = []
    for item in data.get("pairs", []):
        kind = str(item.get("kind", "")).strip()
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if kind in {"lookup", "reasoning", "summary"} and q and a:
            pairs.append((kind, q, a))
        elif kind == "absent" and q:
            pairs.append((kind, q, None))
    return key, pairs


# ---------------------------------------------------------------- 3. retrieval contexts

class ChunkPool:
    """All chunks of one split, embedded once, searched like the app does (vector search + rerank)."""

    def __init__(self, items):
        self.items = items  # (chunk_id, text, doc_title, doc_id)
        self.vectors = model.encode([t for _, t, _, _ in items], normalize_embeddings=True, show_progress_bar=False)

    def search(self, question, allowed=lambda item: True):
        q = model.encode([question], normalize_embeddings=True)[0]
        order = np.argsort(-(self.vectors @ q))
        candidates = [self.items[i] for i in order if allowed(self.items[i])][:CANDIDATES]
        if not candidates:
            return []
        scores = reranker.predict([[question, c[1]] for c in candidates])
        ranked = [c for c, _ in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)]
        return ranked[:CONTEXT_CHUNKS]


def build_example(context, question, target, doc, kind, gold, reference):
    labeled = [f"[{i}] (from {title})\n{text}" for i, (_, text, title, _) in enumerate(context, start=1)]
    user = "Context:\n" + "\n\n---\n\n".join(labeled) + f"\n\nQuestion: {question}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ],
        "paper": doc["id"],          # train_v2.py splits validation by this column
        "doc_type": doc["type"],
        "ocr": doc["ocr"],
        "type": kind,
        "gold": gold,
        "question": question,
        "reference": reference,
    }


def file_title(doc):
    """What the app would show as the source: a PDF file name."""
    safe = "".join(c if c.isalnum() or c in " -_" else "" for c in doc["title"])[:60].strip() or doc["id"]
    return f"{safe}.pdf"


def main():
    rng = random.Random(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    docs, changed_docs = load_or_make_documents(rng)

    # Chunk every document the same way the app does.
    chunk_items = {"train": [], "test": []}
    chunks_by_doc = {}
    for doc in docs:
        title = file_title(doc)
        chunks = [c for c in chunk_text(doc["text"]) if len(c.split()) >= 40]
        chunks_by_doc[doc["id"]] = [(f"{doc['id']}#{i}", c) for i, c in enumerate(chunks)]
        for cid, c in chunks_by_doc[doc["id"]]:
            chunk_items[doc["split"]].append((cid, c, title, doc["id"]))

    # Pick chunks to ask about.
    jobs = []
    for doc in docs:
        chunks = list(chunks_by_doc[doc["id"]])
        rng.shuffle(chunks)
        limit = CHUNKS_PER_TRAIN_DOC if doc["split"] == "train" else CHUNKS_PER_TEST_DOC
        for cid, c in chunks[:limit]:
            jobs.append((cid, doc, c))

    qa_path = os.path.join(OUT_DIR, "questions.jsonl")
    done = {}
    if os.path.exists(qa_path):
        with open(qa_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row["key"].split("#")[0] in changed_docs:
                    continue  # this document's text changed (OCR), so its old questions don't fit any more
                done[row["key"]] = [tuple(p) for p in row["pairs"]]
        print(f"Reusing questions for {len(done)} chunks")
    todo = [j for j in jobs if j[0] not in done]
    if todo:
        print(f"Writing questions for {len(todo)} chunks with gpt-4o-mini ...")
        with open(qa_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for i, fut in enumerate(as_completed([pool.submit(make_questions, j) for j in todo]), start=1):
                key, pairs = fut.result()
                if pairs:
                    done[key] = pairs
                    f.write(json.dumps({"key": key, "pairs": pairs}, ensure_ascii=False) + "\n")
                if i % 50 == 0:
                    print(f"  {i}/{len(todo)} chunks")

    print("\nBuilding retrieval contexts (local, no API cost) ...")
    pools = {split: ChunkPool(items) for split, items in chunk_items.items()}
    out = {"train": [], "test": []}
    stats = {"gold_retrieved": 0, "gold_inserted": 0}
    import time
    started = time.time()

    for n, (cid, doc, chunk) in enumerate(jobs, start=1):
        if n % 25 == 0:
            minutes = (time.time() - started) / 60
            left = minutes / n * (len(jobs) - n)
            print(f"  {n}/{len(jobs)} chunks  ({minutes:.1f} min so far, about {left:.0f} min left)")
        pairs = done.get(cid)
        if not pairs:
            continue
        split = doc["split"]
        pool = pools[split]
        gold_item = (cid, chunk, file_title(doc), doc["id"])

        for kind, question, answer in pairs:
            if kind == "absent":
                # Hard negative: the document's own chunk is shown, the answer still isn't there.
                # Other chunks come from OTHER documents, so the answer can't be hiding elsewhere in this one.
                context = pool.search(question, allowed=lambda it: it[3] != doc["id"])[: CONTEXT_CHUNKS - 1]
                context.insert(rng.randint(0, len(context)), gold_item)
                target = f"{NOT_FOUND}\n{USED_MARKER} NONE"
                out[split].append(build_example(context, question, target, doc, "absent", None, NOT_FOUND))
                continue

            context = pool.search(question)
            if cid in [c[0] for c in context]:
                stats["gold_retrieved"] += 1
            else:
                stats["gold_inserted"] += 1
                context = context[: CONTEXT_CHUNKS - 1]
                context.insert(rng.randint(0, len(context)), gold_item)
            gold = [c[0] for c in context].index(cid) + 1
            target = f"{answer}\n{USED_MARKER} {gold}"
            out[split].append(build_example(context, question, target, doc, kind, gold, answer))

        # Unanswerable: a lookup question about this document, but only other documents are retrieved.
        lookups = [p for p in pairs if p[0] == "lookup"]
        if lookups:
            question = lookups[0][1]
            context = pool.search(question, allowed=lambda it: it[3] != doc["id"])
            target = f"{NOT_FOUND}\n{USED_MARKER} NONE"
            out[split].append(build_example(context, question, target, doc, "unanswerable", None, NOT_FOUND))

    # Mix research-paper examples from v2 back in, so the model keeps what it learned there.
    v2 = []
    if os.path.exists(V2_TRAIN):
        with open(V2_TRAIN, encoding="utf-8") as f:
            v2 = [json.loads(line) for line in f]
        rng.shuffle(v2)
        v2 = v2[:V2_EXAMPLES]
        for ex in v2:
            ex.setdefault("doc_type", "research paper")
            ex.setdefault("ocr", False)
    else:
        print(f"WARNING: {V2_TRAIN} not found, so no research-paper examples are mixed in.")

    train = out["train"] + v2
    rng.shuffle(train)
    for split, rows in [("train", train), ("test", out["test"])]:
        with open(os.path.join(OUT_DIR, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for ex in rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    def count(rows, key):
        c = {}
        for r in rows:
            c[r[key]] = c.get(r[key], 0) + 1
        return c

    total = stats["gold_retrieved"] + stats["gold_inserted"]
    print(f"\nDone. Train: {len(train)} examples ({len(out['train'])} new documents + {len(v2)} research papers)")
    print(f"Test: {len(out['test'])} examples from {len({r['paper'] for r in out['test']})} unseen documents")
    print(f"Train question types: {count(out['train'], 'type')}")
    print(f"Test question types:  {count(out['test'], 'type')}")
    print(f"OCR examples: train {sum(r['ocr'] for r in out['train'])}, test {sum(r['ocr'] for r in out['test'])}")
    if total:
        print(f"Retrieval found the right chunk by itself for {stats['gold_retrieved']}/{total} answerable questions.")
    print(f"Saved in: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
