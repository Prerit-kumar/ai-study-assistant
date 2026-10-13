def chunk_text(text, chunk_size=180, overlap=30):
    """Split text into overlapping chunks of about `chunk_size` words.

    180 words is roughly 250 tokens, which fits the 256-token limit of the
    embedding model (all-MiniLM-L6-v2), so every word of a chunk is embedded.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break  # the last chunk already reaches the end; no duplicate tail chunk
        start += step

    return chunks


if __name__ == "__main__":
    import sys

    from pdf_reader import read_pdf

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "../data/DistilBERT.pdf"
    chunks = chunk_text(read_pdf(pdf_path))

    print(f"Total chunks created: {len(chunks)}")
    print(f"Words per chunk: {[len(c.split()) for c in chunks]}")
    print("\n--- First chunk preview ---\n")
    print(chunks[0])