import faiss

def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search_index(query_embedding, index, k=4):
    D, I = index.search(query_embedding, k)
    return I[0]
