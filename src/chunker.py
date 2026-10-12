def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        start = end - overlap  # overlap so context isn't lost at boundaries

    return chunks


if __name__ == "__main__":
    from pdf_reader import read_pdf

    pdf_path = "../data/test.pdf"
    full_text = read_pdf(pdf_path)

    chunks = chunk_text(full_text)

    print(f"Total chunks created: {len(chunks)}")
    print("\n--- First chunk preview ---\n")
    print(chunks[0])