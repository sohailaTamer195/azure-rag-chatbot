import re

from openai import RateLimitError

from services.embeddings import embed_chunks
from services.vector_store import search_index
from models.azure_client import client
from config.settings import CHAT_DEPLOYMENT

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "what", "when", "where", "which", "who", "why", "with",
}


def _terms(text):
    return {
        term for term in re.findall(r"\w+", text.casefold())
        if term not in STOP_WORDS and len(term) > 2
    }


def retrieve(query, index, chunks, k=4):
    query_normalized = " ".join(re.findall(r"\w+", query.casefold()))
    query_terms = _terms(query)
    semantic_indexes = []
    if index is not None:
        try:
            q_emb = embed_chunks([query])
            semantic_indexes = list(search_index(q_emb, index, k))
        except RateLimitError:
            pass

    lexical_scores = []
    for position, chunk in enumerate(chunks):
        chunk_normalized = " ".join(re.findall(r"\w+", chunk.casefold()))
        chunk_terms = _terms(chunk)
        score = len(chunk_terms.intersection(query_terms))
        if query_normalized and query_normalized in chunk_normalized:
            score += len(query_terms) + 2
        if score:
            lexical_scores.append((score, position))
    lexical_indexes = [
        position for _, position in sorted(lexical_scores, key=lambda item: (-item[0], item[1]))[:k]
    ]

    number_one_query = bool(
        query_terms.intersection({"1", "one", "first", "الأول", "الاول", "١"})
    )
    if number_one_query:
        markers = {"1", "١", "one", "first", "الأول", "الاول", "السؤال"}
        scored = []
        for position, chunk in enumerate(chunks):
            chunk_terms = _terms(chunk)
            score = len(chunk_terms.intersection(markers))
            if score:
                scored.append((score, position))
        lexical_indexes.extend(
            position for _, position in sorted(scored, key=lambda item: (-item[0], item[1]))[:k]
        )

    indexes = []
    for position in lexical_indexes + semantic_indexes:
        if 0 <= position < len(chunks) and position not in indexes:
            indexes.append(position)
    if not indexes:
        return chunks[:k]
    return [chunks[i] for i in indexes[: max(k, len(lexical_indexes))]]


def rag_answer(query, index, chunks):
    # For small documents, passing all text is more reliable than ranking
    # chunks, especially for vague references such as "number 1".
    document_size = sum(len(chunk) for chunk in chunks)
    context_chunks = chunks if document_size <= 20000 else retrieve(query, index, chunks)
    context = "\n\n".join(context_chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "Answer ONLY using the provided PDF context. "
                "Answer the user's question directly and briefly. "
                "If the answer is not in the PDF, say you don't know. "
                "Use the same language as the user's question when possible."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        },
    ]

    resp = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=messages,
    )

    return resp.choices[0].message.content
