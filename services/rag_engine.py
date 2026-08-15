import re

from services.embeddings import embed_chunks
from services.vector_store import search_index
from models.azure_client import client
from config.settings import CHAT_DEPLOYMENT


def retrieve(query, index, chunks, k=4):
    q_emb = embed_chunks([query])
    semantic_indexes = list(search_index(q_emb, index, k))

    query_terms = set(re.findall(r"\w+", query.casefold()))
    number_one_query = bool(
        query_terms.intersection({"1", "one", "first", "الأول", "الاول", "١"})
    )
    lexical_indexes = []
    if number_one_query:
        markers = {"1", "١", "one", "first", "الأول", "الاول", "السؤال"}
        scored = []
        for position, chunk in enumerate(chunks):
            chunk_terms = set(re.findall(r"\w+", chunk.casefold()))
            score = len(chunk_terms.intersection(markers))
            if score:
                scored.append((score, position))
        lexical_indexes = [position for _, position in sorted(scored, reverse=True)[:k]]

    indexes = []
    for position in lexical_indexes + semantic_indexes:
        if 0 <= position < len(chunks) and position not in indexes:
            indexes.append(position)
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
