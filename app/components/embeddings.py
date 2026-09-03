import os
import sys
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List
from langchain_core.embeddings import Embeddings
from app.common.logger import get_logger

logger = get_logger(__name__)

class DirectHuggingFaceEndpointEmbeddings(Embeddings):
    """Direct HTTP requests to Hugging Face's updated Inference Router API."""
    def __init__(self, token: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Updated active inference router host
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model_name}"
        self.headers = {
            "Authorization": f"Bearer {token.strip()}",
            "Content-Type": "application/json"
        }
        
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[502, 503, 504],
            raise_on_status=False
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def _call_api(self, text_payload):
        response = self.session.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": text_payload, "options": {"wait_for_model": True}},
            timeout=45
        )
        if response.status_code != 200:
            raise RuntimeError(f"Hugging Face API error ({response.status_code}): {response.text}")
        return response.json()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._call_api(texts)

    def embed_query(self, text: str) -> List[float]:
        data = self._call_api(text)
        
        # Flatten embedding if wrapped in outer batch dimension
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            return data[0]
        return data

def get_embedding_model():
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN) is not set in Render Environment Variables.")

    print(f"[DEBUG] HF Token loaded (starts with: {token[:6]}...)", file=sys.stderr, flush=True)
    return DirectHuggingFaceEndpointEmbeddings(token=token)