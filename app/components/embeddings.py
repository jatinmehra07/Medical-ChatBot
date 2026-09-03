import os
import sys
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from app.common.logger import get_logger

logger = get_logger(__name__)

def get_embedding_model():
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    if not token:
        raise ValueError(
            "Hugging Face token is missing! Please set 'HF_TOKEN' in Render Environment Variables."
        )

    print(f"[DEBUG] HF Token found (starts with: {token[:6]}...)", file=sys.stderr, flush=True)

    try:
        # Remote inference via Hugging Face Serverless API (Zero local RAM overhead)
        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction",
            huggingfacehub_api_token=token.strip(),
        )
        return embeddings
    except TypeError:
        # Fallback for versions accepting 'api_key' instead of 'huggingfacehub_api_token'
        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction",
            api_key=token.strip(),
        )
        return embeddings
    except Exception as e:
        print(f"[DEBUG ERROR IN EMBEDDINGS]: {str(e)}", file=sys.stderr, flush=True)
        raise e