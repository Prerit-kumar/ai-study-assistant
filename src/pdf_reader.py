import re

import pdfplumber


def read_pdf(file_path):
    """Extract text from a PDF, keeping the spaces between words.

    x_tolerance=1 stops pdfplumber from gluing words together
    ("AsTransferLearningfrom..."), which broke keyword search and made
    chunks far larger than intended.
    """
    full_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text(x_tolerance=1)
            except Exception as e:
                print(f"[pdf_reader] could not read page {page_number} of {file_path}: {e}")
                continue
            if text:
                full_text += f"\n[Page {page_number}]\n" + text

    # Re-join words split by a hyphen at the end of a line: "general-\npurpose" -> "general-purpose"
    full_text = re.sub(r"(\w)-\n(\w)", r"\1-\2", full_text)
    return full_text


if __name__ == "__main__":
    import sys

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "../data/DistilBERT.pdf"
    result = read_pdf(pdf_path)
    print(f"chars: {len(result)}, words: {len(result.split())}\n")
    print(result[:1500])