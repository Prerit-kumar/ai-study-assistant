import re

import pdfplumber

# ---------------------------------------------------------------- OCR settings
# Pages with fewer words than this are treated as scanned and read with OCR.
MIN_WORDS_PER_PAGE = 20
# OCR is slow (about 1-2 s per page on CPU), so a very long scanned PDF is capped.
MAX_OCR_PAGES = 30
# Pictures inside normal pages are read too if they are big enough to hold text
# (a screenshot, a photographed table), but not logos or icons.
MIN_IMAGE_SHARE = 0.12      # at least 12% of the page area
MIN_IMAGE_SIDE = 120        # and at least 120 points wide and high
MIN_IMAGE_WORDS = 8         # keep a picture's text only if OCR found at least this many words
OCR_DPI = 200

_ocr_engine = None
_ocr_failed = False


def _get_ocr():
    """Load RapidOCR once, on first use. Returns None if it isn't installed."""
    global _ocr_engine, _ocr_failed
    if _ocr_engine is not None or _ocr_failed:
        return _ocr_engine
    try:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = ("old", RapidOCR())
    except ImportError:
        try:
            from rapidocr import RapidOCR  # newer package name
            _ocr_engine = ("new", RapidOCR())
        except ImportError:
            print("[pdf_reader] OCR is off: run  pip install rapidocr-onnxruntime pymupdf")
            _ocr_failed = True
    return _ocr_engine


# RapidOCR's default model often drops the space between words ("CustomerName:MariaSantos",
# "01-Aug-2023to"). Glued words break keyword search, so the most common cases are split again.
_SPACING_RULES = [
    (re.compile(r"(?<=[a-z]{2})(?=[A-Z][a-z])"), " "),                 # CustomerName -> Customer Name (kWh, g/dL stay)
    (re.compile(r"(?<=[A-Za-z]):(?=[A-Za-z0-9])"), ": "),              # ID:INS-23 -> ID: INS-23 (10:30 stays)
    (re.compile(r":(?=[A-Za-z])"), ": "),
    (re.compile(r",(?=[A-Za-z])"), ", "),                               # (1,200 stays)
    (re.compile(r"(?<=\d)(?!(?:st|nd|rd|th)\b)(?=[a-z]{2,})"), " "),    # 2023to -> 2023 to (2nd stays)
    (re.compile(r"(?<=\d)(?=[A-Z][a-z])"), " "),                        # 3Months -> 3 Months
]


def _fix_spacing(line):
    for pattern, replacement in _SPACING_RULES:
        line = pattern.sub(replacement, line)
    return line


def _ocr_lines(image):
    """Run OCR on an RGB image array and return its text as lines, top to bottom."""
    engine = _get_ocr()
    if engine is None:
        return ""
    kind, ocr = engine
    bgr = image[:, :, ::-1].copy()
    if kind == "old":
        result, _ = ocr(bgr)
        items = [(box, text) for box, text, _score in (result or [])]
    else:
        result = ocr(bgr)
        items = list(zip(result.boxes, result.txts)) if result.txts else []
    if not items:
        return ""

    # Group the detected words into lines by their vertical position, then read each line left to right.
    boxes = []
    for box, text in items:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        boxes.append((sum(ys) / len(ys), min(xs), max(ys) - min(ys), text))
    boxes.sort()
    lines, current, current_y, current_h = [], [], None, None
    for y, x, h, text in boxes:
        if current and abs(y - current_y) > max(current_h, h) * 0.6:
            lines.append(" ".join(t for _, t in sorted(current)))
            current = []
        if not current:
            current_y, current_h = y, h
        current.append((x, text))
    if current:
        lines.append(" ".join(t for _, t in sorted(current)))
    return "\n".join(_fix_spacing(line) for line in lines)


def _pymupdf():
    try:
        import pymupdf
        return pymupdf
    except ImportError:
        import fitz  # older PyMuPDF versions
        return fitz


def _render(fitz_page, clip=None):
    """Render a page (or part of it) to an RGB image array."""
    import numpy as np

    pix = fitz_page.get_pixmap(dpi=OCR_DPI, clip=clip)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return image[:, :, :3]


def read_pdf(file_path, ocr=True):
    """Extract text from a PDF, keeping the spaces between words.

    x_tolerance=1 stops pdfplumber from gluing words together
    ("AsTransferLearningfrom..."), which broke keyword search and made
    chunks far larger than intended.

    With ocr=True, two more things are read with OCR:
      - pages with almost no text (scanned pages or photos saved as PDF)
      - large pictures on normal pages (screenshots, photographed tables)
    """
    fitz_doc = None
    if ocr and _get_ocr() is not None:
        try:
            fitz_doc = _pymupdf().open(file_path)  # PyMuPDF, used only to turn pages into images
        except Exception as e:
            print(f"[pdf_reader] OCR is off for {file_path}: {e}")

    full_text = ""
    ocr_pages = 0
    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text(x_tolerance=1) or ""
            except Exception as e:
                print(f"[pdf_reader] could not read page {page_number} of {file_path}: {e}")
                text = ""

            if fitz_doc is not None and ocr_pages < MAX_OCR_PAGES:
                fitz_page = fitz_doc[page_number - 1]

                if len(text.split()) < MIN_WORDS_PER_PAGE:
                    # Scanned page: read the whole page as an image.
                    ocr_pages += 1
                    try:
                        scanned = _ocr_lines(_render(fitz_page))
                    except Exception as e:
                        print(f"[pdf_reader] OCR failed on page {page_number}: {e}")
                        scanned = ""
                    if len(scanned.split()) > len(text.split()):
                        text = scanned
                else:
                    # Normal page: also read large pictures on it.
                    page_area = float(page.width * page.height)
                    seen = set()
                    for img in page.images:
                        x0, top, x1, bottom = img["x0"], img["top"], img["x1"], img["bottom"]
                        w, h = x1 - x0, bottom - top
                        key = (round(x0), round(top), round(x1), round(bottom))
                        if key in seen or w < MIN_IMAGE_SIDE or h < MIN_IMAGE_SIDE or w * h < MIN_IMAGE_SHARE * page_area:
                            continue
                        seen.add(key)
                        if ocr_pages >= MAX_OCR_PAGES:
                            break
                        ocr_pages += 1
                        try:
                            picture_text = _ocr_lines(_render(fitz_page, clip=_pymupdf().Rect(x0, top, x1, bottom)))
                        except Exception as e:
                            print(f"[pdf_reader] OCR failed on a picture on page {page_number}: {e}")
                            continue
                        if len(picture_text.split()) >= MIN_IMAGE_WORDS:
                            text += "\n[Text from a picture]\n" + picture_text

            if text.strip():
                full_text += f"\n[Page {page_number}]\n" + text

    if fitz_doc is not None:
        fitz_doc.close()
        if ocr_pages:
            print(f"[pdf_reader] used OCR {ocr_pages} time(s) on {file_path}")

    # Re-join words split by a hyphen at the end of a line: "general-\npurpose" -> "general-purpose"
    full_text = re.sub(r"(\w)-\n(\w)", r"\1-\2", full_text)
    return full_text


if __name__ == "__main__":
    import sys

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "../data/DistilBERT.pdf"
    result = read_pdf(pdf_path)
    print(f"chars: {len(result)}, words: {len(result.split())}\n")
    print(result[:1500])
