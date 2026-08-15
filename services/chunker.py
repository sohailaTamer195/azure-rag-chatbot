import re


def chunk_text(text, chunk_size=800, overlap=200):
    text = re.sub(r"\r\n?", "\n", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundaries = (
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind("? ", start, end),
                text.rfind("! ", start, end),
            )
            boundary = max(boundaries)
            if boundary > start + chunk_size // 2:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def chunk_pages(pages, chunk_size=800, overlap=200):
    chunks = []
    for page_number, page_text in pages:
        chunks.extend(
            f"[Page {page_number}]\n{chunk}"
            for chunk in chunk_text(page_text, chunk_size, overlap)
        )
    return chunks
