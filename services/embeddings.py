import numpy as np
import time
from openai import RateLimitError

from models.azure_client import client
from config.settings import EMBED_DEPLOYMENT

def _retry_delay(error, attempt):
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {})
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after:
        try:
            return max(float(retry_after), 60.0)
        except (TypeError, ValueError):
            pass
    return min(2 ** attempt, 16)


def embed_chunks(chunks, batch_size=32, batch_delay=0.0, max_retries=5):
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
            except RateLimitError as error:
                if attempt == max_retries:
                    raise
                time.sleep(_retry_delay(error, attempt))

        if batch_delay and start + batch_size < len(chunks):
            time.sleep(batch_delay)

    embeddings = np.array(embeddings, dtype="float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)
    
