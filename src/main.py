import os
import re
import threading
import time
from collections import defaultdict, deque

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pdf_reader import read_pdf
from chunker import chunk_text
from embed_store import store_chunks
from rag_pipeline import generate_answer
from comparison import compare_documents_detailed, get_documents_in_store

app = FastAPI(title="AI Study Assistant API")

# Only our own frontend may call the API. For deployment, set FRONTEND_ORIGINS in .env,
# e.g. FRONTEND_ORIGINS=https://my-app.vercel.app,http://localhost:5173
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
MAX_UPLOAD_MB = 25
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
NOT_FOUND_PHRASE = "couldn't find this in the provided documents"

# ---------------------------------------------------------------- rate limiting
# Per user (IP address): list of (max requests, window in seconds).
RATE_LIMITS = {
    "ask": [(10, 60), (100, 86400)],
    "compare": [(3, 60), (20, 86400)],
    "upload": [(10, 3600)],
}
# For the whole app, per day. Caps the worst-case OpenAI bill.
GLOBAL_DAILY_LIMITS = {"ask": 1000, "compare": 100, "upload": 200}

_hits = defaultdict(deque)   # (action, ip or "*") -> timestamps of recent requests
_rate_lock = threading.Lock()


def _window_text(seconds):
    return {60: "minute", 3600: "hour", 86400: "day"}.get(seconds, f"{seconds} seconds")


def client_ip(request: Request):
    # Behind a hosting proxy the real address is in X-Forwarded-For (set TRUST_PROXY=1 there).
    if os.getenv("TRUST_PROXY") == "1":
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limit(request: Request, action: str):
    """Raises HTTP 429 if this user, or the whole app, has made too many requests."""
    now = time.time()
    ip = client_ip(request)
    checks = [
        ((action, ip), RATE_LIMITS[action], "you have"),
        ((action, "*"), [(GLOBAL_DAILY_LIMITS[action], 86400)], "the app has"),
    ]
    with _rate_lock:
        for key, rules, who in checks:
            timestamps = _hits[key]
            longest = max(window for _, window in rules)
            while timestamps and now - timestamps[0] > longest:
                timestamps.popleft()
            for limit, window in rules:
                recent = sum(1 for t in timestamps if now - t <= window)
                if recent >= limit:
                    print(f"[rate-limit] {action} blocked for {ip}: {limit} per {_window_text(window)}")
                    raise HTTPException(
                        status_code=429,
                        detail=(f"Too many {action} requests: {who} reached the limit of {limit} per "
                                f"{_window_text(window)}. Please wait and try again."),
                    )
        for key, _, _ in checks:
            _hits[key].append(now)


# ---------------------------------------------------------------- request models

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(8, ge=1, le=20)


class CompareRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    documents: list[str] = Field(default_factory=list, max_length=100)  # empty = all documents


def safe_pdf_name(filename):
    """Keep only the base name and safe characters, so a file can never be written outside data/."""
    name = os.path.basename(filename or "").strip()
    name = re.sub(r"[^A-Za-z0-9._() -]", "_", name)
    name = name.lstrip(".")
    if not name.lower().endswith(".pdf") or len(name) <= 4:
        return None
    return name[:150]


# ---------------------------------------------------------------- endpoints

@app.get("/")
def root():
    return {"message": "AI Study Assistant API is running"}


@app.post("/upload")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    """Upload a PDF, check it, process it, and store it in the vector database."""
    check_rate_limit(request, "upload")

    filename = safe_pdf_name(file.filename)
    if filename is None:
        raise HTTPException(status_code=400, detail="Only .pdf files can be uploaded.")

    os.makedirs(DATA_DIR, exist_ok=True)
    save_path = os.path.join(DATA_DIR, filename)

    # Read in pieces so an oversized file is rejected without loading it all into memory.
    size = 0
    first_bytes = b""
    try:
        with open(save_path, "wb") as buffer:
            while True:
                piece = await file.read(1024 * 1024)
                if not piece:
                    break
                if not first_bytes:
                    first_bytes = piece[:5]
                size += len(piece)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail=f"File is larger than {MAX_UPLOAD_MB} MB.")
                buffer.write(piece)
    except HTTPException:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise

    if not first_bytes.startswith(b"%PDF"):
        os.remove(save_path)
        raise HTTPException(status_code=400, detail="This file is not a valid PDF.")

    try:
        full_text = read_pdf(save_path)
    except Exception as e:
        os.remove(save_path)
        print(f"[upload] could not read {filename}: {e}")
        raise HTTPException(status_code=400, detail="The PDF could not be read.")

    chunks = chunk_text(full_text)
    if not chunks:
        os.remove(save_path)
        raise HTTPException(
            status_code=400,
            detail="No readable text was found in this PDF (it may be a scanned image).",
        )

    stored = store_chunks(chunks, doc_id=filename)
    return {"filename": filename, "chunks_stored": stored}


@app.post("/ask")
def ask_question(body: QuestionRequest, request: Request):
    """Ask a question and get a grounded answer with sources."""
    check_rate_limit(request, "ask")
    answer, sources = generate_answer(body.question, top_k=body.top_k)

    # If the model could not answer, retrieved documents are not real sources.
    normalized = answer.lower().replace("\u2019", "'")
    if NOT_FOUND_PHRASE in normalized:
        sources = []

    return {"answer": answer, "sources": list(sources)}


@app.post("/compare")
def compare(body: CompareRequest, request: Request):
    """Compare multiple documents on a given question."""
    check_rate_limit(request, "compare")
    documents = body.documents if body.documents else get_documents_in_store()
    result = compare_documents_detailed(body.question, documents)
    return {
        "answer": result["answer"],
        "documents_compared": result["documents_used"],
    }


@app.get("/documents")
def list_documents():
    """List all documents currently in the system."""
    return {"documents": get_documents_in_store()}