"""Inspect and clean the document database (no OpenAI calls, free to run).

Usage (from the src folder, venv active, backend STOPPED):
    python db_tools.py report
    python db_tools.py delete "test.pdf" "test2.pdf"

"delete" removes the document's chunks from ChromaDB and moves its PDF from
data/ to data/_removed/ (nothing is permanently deleted, so it can be undone).
"""

import os
import shutil
import sys

from embed_store import collection

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SRC_DIR, "..", "data"))
REMOVED_DIR = os.path.join(DATA_DIR, "_removed")

LOW_CHUNKS = 12          # fewer chunks than this is suspicious for a research paper
LOW_CHARS_PER_PAGE = 800  # less text than this per page suggests a bad or scanned PDF


def chunk_stats():
    data = collection.get(include=["metadatas", "documents"])
    per_doc = {}
    for meta, text in zip(data["metadatas"], data["documents"]):
        source = meta.get("source", "?")
        entry = per_doc.setdefault(source, {"chunks": 0, "words": 0})
        entry["chunks"] += 1
        entry["words"] += len(text.split())
    return per_doc


def pdf_info(name):
    """Returns (pages, characters) for a PDF in data/, or a short message."""
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        return "file not in data/"
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            pages = len(pdf.pages)
            chars = sum(len(page.extract_text() or "") for page in pdf.pages)
        return pages, chars
    except Exception as e:
        return f"could not read: {e}"


def looks_like_junk(name):
    lower = name.lower()
    return lower.startswith("test") or "(1)" in lower or "copy" in lower


def report():
    per_doc = chunk_stats()
    print(f"\nDocuments in database: {len(per_doc)}\n")
    print(f"{'Document':45} {'Chunks':>6} {'Words':>7}  Flag")
    print("-" * 75)

    flagged = []
    for name, s in sorted(per_doc.items(), key=lambda item: item[1]["chunks"]):
        flags = []
        if s["chunks"] < LOW_CHUNKS:
            flags.append("LOW CHUNKS")
        if looks_like_junk(name):
            flags.append("JUNK?")
        if flags:
            flagged.append(name)
        print(f"{name[:45]:45} {s['chunks']:>6} {s['words']:>7}  {', '.join(flags)}")

    if flagged:
        print("\nChecking the PDF files of flagged documents...")
        for name in flagged:
            info = pdf_info(name)
            if isinstance(info, tuple):
                pages, chars = info
                per_page = chars // max(pages, 1)
                note = "  <- very little text per page" if per_page < LOW_CHARS_PER_PAGE else ""
                print(f"  {name}: {pages} pages, {chars} characters ({per_page}/page){note}")
            else:
                print(f"  {name}: {info}")

    pdfs_on_disk = {f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")}
    missing_in_db = sorted(pdfs_on_disk - set(per_doc))
    missing_on_disk = sorted(set(per_doc) - pdfs_on_disk)
    if missing_in_db:
        print("\nPDFs in data/ but NOT in the database: " + ", ".join(missing_in_db))
    if missing_on_disk:
        print("\nIn the database but no PDF in data/: " + ", ".join(missing_on_disk))
    print()


def delete(names):
    per_doc = chunk_stats()
    os.makedirs(REMOVED_DIR, exist_ok=True)
    for name in names:
        if name not in per_doc:
            print(f"Not in database (check the exact name): {name}")
            continue
        collection.delete(where={"source": name})
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            shutil.move(path, os.path.join(REMOVED_DIR, name))
            print(f"Removed {name}: {per_doc[name]['chunks']} chunks deleted, PDF moved to data/_removed/")
        else:
            print(f"Removed {name}: {per_doc[name]['chunks']} chunks deleted (no PDF file found)")
    print(f"\nDocuments left in database: {len(chunk_stats())}")
    print("Restart the backend so search and the document list pick up the change.")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "report":
        report()
    elif len(sys.argv) >= 3 and sys.argv[1] == "delete":
        delete(sys.argv[2:])
    else:
        print(__doc__)