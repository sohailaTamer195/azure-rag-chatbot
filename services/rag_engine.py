import re

from openai import RateLimitError

from models.azure_client import client
from config.settings import CHAT_DEPLOYMENT

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "what", "when", "where", "which", "who", "why", "with",
}
TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
ARABIC_TRANSLATION = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"})


def _normalize_token(token):
    token = re.sub(r"[\u064B-\u065F\u0670]", "", token.casefold())
    token = token.translate(ARABIC_TRANSLATION)
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _terms(text):
    return {
        term
        for raw_term in TOKEN_PATTERN.findall(text)
        if (term := _normalize_token(raw_term)) not in STOP_WORDS and len(term) > 2
    }


def retrieve(query, index, chunks, k=6):
    query_terms = _terms(query)
    query_phrase = " ".join(
        _normalize_token(term)
        for term in TOKEN_PATTERN.findall(query)
        if _normalize_token(term) not in STOP_WORDS
    )

    lexical_scores = []
    for position, chunk in enumerate(chunks):
        chunk_normalized = " ".join(
            _normalize_token(term) for term in TOKEN_PATTERN.findall(chunk)
        )
        chunk_terms = _terms(chunk)
        overlap = chunk_terms.intersection(query_terms)
        score = len(overlap) * 3
        if query_phrase and query_phrase in chunk_normalized:
            score += len(query_terms) * 4
        for query_term in query_terms - overlap:
            if len(query_term) >= 4 and any(
                token.startswith(query_term) or query_term.startswith(token)
                for token in chunk_terms
            ):
                score += 1
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
    if client is None or not CHAT_DEPLOYMENT:
        raise RuntimeError("Azure OpenAI is not configured.")

    context_chunks = retrieve(query, index, chunks)
    context = "\n\n".join(context_chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "Answer immediately and directly. Do not describe internal steps "
                "or mention embedding, processing, loading, reading the document, "
                "latency, speed, background operations, or system processes. Use "
                "ONLY the retrieved document chunks provided in the user message as "
                "your source of truth. Read and examine ALL retrieved chunks before "
                "answering, including indirect or split information. Combine relevant "
                "details from multiple chunks when needed. If the answer is present "
                "anywhere in the chunks, give a clear and concise answer. Only if the "
                "answer truly does not exist in any chunk, reply exactly: "
                "'The document does not contain information about this.' "
                "Do not guess, assume, hallucinate, or add information from outside "
                "the chunks. Answer in the same language as the document, in Arabic "
                "or English. Keep the tone clean and "
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
