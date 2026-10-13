import os
import re

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pdf_reader import read_pdf
from chunker import chunk_text
from embed_store import store_chunks, collection
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


@app.get("/")
def root():
    return {"message": "AI Study Assistant API is running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, check it, process it, and store it in the vector database."""
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
def ask_question(request: QuestionRequest):
    """Ask a question and get a grounded answer with sources."""
    answer, sources = generate_answer(request.question, top_k=request.top_k)

    # If the model could not answer, retrieved documents are not real sources.
    normalized = answer.lower().replace("\u2019", "'")
    if NOT_FOUND_PHRASE in normalized:
        sources = []

    return {"answer": answer, "sources": list(sources)}


@app.post("/compare")
def compare(request: CompareRequest):
    """Compare multiple documents on a given question."""
    documents = request.documents if request.documents else get_documents_in_store()
    result = compare_documents_detailed(request.question, documents)
    return {
        "answer": result["answer"],
        "documents_compared": result["documents_used"],
    }


@app.get("/documents")
def list_documents():
    """List all documents currently in the system."""
    return {"documents": get_documents_in_store()}