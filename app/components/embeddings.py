import os
import sys
import requests
from typing import List
from langchain_core.embeddings import Embeddings
from app.common.logger import get_logger

logger = get_logger(__name__)

class DirectHuggingFaceEndpointEmbeddings(Embeddings):
    """Zero-memory embeddings using direct HTTP requests to Hugging Face Serverless API."""
    def __init__(self, token: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
        self.headers = {"Authorization": f"Bearer {token.strip()}"}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=60
        )
        if response.status_code != 200:
            raise RuntimeError(f"Hugging Face API error ({response.status_code}): {response.text}")
        return response.json()

    def embed_query(self, text: str) -> List[float]:
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": text, "options": {"wait_for_model": True}},
            timeout=60
        )
        if response.status_code != 200:
            raise RuntimeError(f"Hugging Face API error ({response.status_code}): {response.text}")
        data = response.json()
        
        # Normalize nested return structures from feature extraction
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            return data[0]
        return data

def get_embedding_model():
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN) is not set in Render Environment Variables.")

    print(f"[DEBUG] HF Token loaded (starts with: {token[:6]}...)", file=sys.stderr, flush=True)
    return DirectHuggingFaceEndpointEmbeddings(token=token)