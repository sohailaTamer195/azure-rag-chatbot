import numpy as np
import time
from openai import RateLimitError

from models.azure_client import client
from config.settings import EMBED_DEPLOYMENT

def embed_chunks(chunks, batch_size=16, max_retries=5):
    embeddings = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        for attempt in range(max_retries + 1):
            try:
                resp = client.embeddings.create(
                    model=EMBED_DEPLOYMENT,
                    input=batch,
                )
                embeddings.extend(item.embedding for item in resp.data)
                break
            except RateLimitError:
                if attempt == max_retries:
                    raise
                time.sleep(min(2 ** attempt, 16))

    embeddings = np.array(embeddings, dtype="float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)
    
