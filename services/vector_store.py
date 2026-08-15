import faiss

def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def search_index(query_embedding, index, k=4):
    query_norm = query_embedding / max(float((query_embedding**2).sum() ** 0.5), 1e-12)
    _, indexes = index.search(query_norm, min(k, index.ntotal))
    return indexes[0]
