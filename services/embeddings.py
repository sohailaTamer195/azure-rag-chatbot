import numpy as np
from models.azure_client import client
from config.settings import EMBED_DEPLOYMENT

def embed_chunks(chunks):
    resp = client.embeddings.create(
    model=EMBED_DEPLOYMENT,
    input=chunks
)
    return np.array([e.embedding for e in resp.data], dtype="float32")
    
