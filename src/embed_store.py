import os

import chromadb
from sentence_transformers import SentenceTransformer

from chunker import chunk_text
from pdf_reader import read_pdf

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="../chroma_db")
collection = client.get_or_create_collection("documents")

MIN_EXPECTED_CHUNKS = 5


def store_chunks(chunks, doc_id):
    chunks = [c for c in chunks if c and c.strip()]
    if not chunks:
        return 0

    # Remove any older version of this document so re-uploads don't clash.
    collection.delete(where={"source": doc_id})

    embeddings = model.encode(chunks).tolist()
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": doc_id} for _ in chunks]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(chunks)


def search(query, top_k=3):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    chunks = results["documents"][0]
    sources = [meta["source"] for meta in results["metadatas"][0]]
    return list(zip(chunks, sources))


def process_all_pdfs(data_folder="../data"):
    pdf_files = sorted(
        f for f in os.listdir(data_folder) if f.lower().endswith(".pdf")
    )
    print(f"Found {len(pdf_files)} PDFs\n")

    problems = []

    for filename in pdf_files:
        path = os.path.join(data_folder, filename)

        try:
            text = read_pdf(path)
            chunks = chunk_text(text)
            count = store_chunks(chunks, filename)
        except Exception as e:
            print(f"FAILED  {filename}: {e}")
            problems.append(filename)
            continue

        if count == 0:
            print(f"SKIPPED {filename}: no readable text found")
            problems.append(filename)
        elif count < MIN_EXPECTED_CHUNKS:
            print(f"Stored {count} chunks from {filename}   <-- WARNING: very few chunks")
            problems.append(filename)
        else:
            print(f"Stored {count} chunks from {filename}")

    print(f"\nTotal chunks in database: {collection.count()}")

    if problems:
        print("\nFiles that need a look:")
        for name in problems:
            print(f"  - {name}")


if __name__ == "__main__":
    process_all_pdfs()

    print("\n--- Quick search test ---")
    for chunk, source in search("What is entity resolution?", top_k=3):
        print(f"[{source}] {chunk[:150]}...\n")