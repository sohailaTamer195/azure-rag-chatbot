import numpy as np
from models.azure_client import client
from config.settings import EMBED_DEPLOYMENT

def embed_chunks(chunks):
    resp = client.embeddings.create(
        model=EMBED_DEPLOYMENT,
        input=chunks,
    )
    embeddings = np.array([e.embedding for e in resp.data], dtype="float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)
    
