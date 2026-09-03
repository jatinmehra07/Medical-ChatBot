import os
import sys
from typing import List
import numpy as np
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings
from app.common.logger import get_logger

logger = get_logger(__name__)

class DirectHuggingFaceEndpointEmbeddings(Embeddings):
    """Hugging Face InferenceClient embeddings targeting feature-extraction directly."""
    def __init__(self, token: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.client = InferenceClient(token=token.strip())
        self.model_name = model_name

    def _extract(self, text: str) -> List[float]:
        # Explicitly requests feature extraction vectors
        res = self.client.feature_extraction(text, model=self.model_name)
        if hasattr(res, "tolist"):
            res = res.tolist()
        
        # Squeeze batch / sequence dimensions if returned as 2D/3D array
        arr = np.array(res)
        if arr.ndim > 1:
            arr = arr.mean(axis=0) if arr.ndim == 2 and arr.shape[0] > 1 else arr.squeeze()
        return arr.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._extract(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._extract(text)

def get_embedding_model():
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN) is not set in Render Environment Variables.")

    print(f"[DEBUG] HF Token loaded (starts with: {token[:6]}...)", file=sys.stderr, flush=True)
    return DirectHuggingFaceEndpointEmbeddings(token=token)