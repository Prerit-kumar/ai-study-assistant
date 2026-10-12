import pdfplumber

def read_pdf(file_path):
    full_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                full_text += f"\n--- Page {page_number} ---\n"
                full_text += text
    return full_text

if __name__ == "__main__":
    pdf_path = "../data/test.pdf"
    result = read_pdf(pdf_path)
    print(result)