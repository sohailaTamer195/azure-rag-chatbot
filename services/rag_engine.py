from services.embeddings import embed_chunks
from services.vector_store import search_index
from models.azure_client import client
from config.settings import CHAT_DEPLOYMENT

def retrieve(query, index, chunks, k=4):
    q_emb = embed_chunks([query])
    idxs = search_index(q_emb, index, k)
    return [chunks[i] for i in idxs]

def rag_answer(query, index, chunks):
    context_chunks = retrieve(query, index, chunks)
    context = "\n\n".join(context_chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "Answer ONLY using the provided PDF context. "
                "If the answer is not in the PDF, say you don't know."
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
