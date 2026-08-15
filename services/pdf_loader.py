import pypdf


def load_pdf_pages(file):
    reader = pypdf.PdfReader(file)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append((page_number, page_text.strip()))
    return pages


def load_pdf(file):
    return "\n\n".join(text for _, text in load_pdf_pages(file))
