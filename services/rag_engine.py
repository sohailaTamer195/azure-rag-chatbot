import re

from openai import RateLimitError

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


def retrieve(query, index, chunks, k=6):
    query_normalized = " ".join(re.findall(r"\w+", query.casefold()))
    query_terms = _terms(query)

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
    for position in lexical_indexes:
        if 0 <= position < len(chunks) and position not in indexes:
            indexes.append(position)
    if not indexes:
        return chunks[:k]

    return [chunks[position] for position in sorted(indexes[:k])]


def rag_answer(query, index, chunks):
    context_chunks = retrieve(query, index, chunks)
    context = "\n\n".join(context_chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "Answer immediately and directly. Do not describe internal steps "
                "or mention embedding, processing, loading, reading the document, "
                "latency, speed, background operations, or system processes. Use "
                "ONLY the retrieved document "
                "chunks provided in the user message as your source of truth. "
                "If the answer is found in the chunks, give a clear, concise answer. "
                "If the chunks do not contain the answer, reply exactly: "
                "'The document does not contain information about this.' "
                "Do not guess, assume, hallucinate, or add information from outside "
                "the chunks. Combine chunks when needed. Answer in the same language "
                "as the document, in Arabic or English. Keep the tone clean and "
                "helpful, and keep responses short, accurate, and focused unless the "
                "user requests a detailed explanation."
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
