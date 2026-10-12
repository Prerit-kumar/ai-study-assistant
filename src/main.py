from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os

from pdf_reader import read_pdf
from chunker import chunk_text
from embed_store import store_chunks, process_all_pdfs
from rag_pipeline import generate_answer
from comparison import compare_documents_detailed, get_documents_in_store

app = FastAPI(title="AI Study Assistant API")

# Allow requests from any frontend (we'll restrict this later for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

NOT_FOUND_PHRASE = "couldn't find this in the provided documents"


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5


class CompareRequest(BaseModel):
    question: str
    documents: list[str] = []  # empty means use all documents


@app.get("/")
def root():
    return {"message": "AI Study Assistant API is running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, process it, and store it in the vector database."""
    save_path = f"../data/{file.filename}"

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the newly uploaded PDF
    full_text = read_pdf(save_path)
    chunks = chunk_text(full_text)
    stored = store_chunks(chunks, doc_id=file.filename)

    return {"filename": file.filename, "chunks_stored": stored}


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