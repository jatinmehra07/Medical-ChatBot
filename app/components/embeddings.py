import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def get_embedding_model():
    try:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        logger.info("Initializing HuggingFace Endpoint Embeddings (Remote API)...")
        
        # Calls HuggingFace's cloud API for the same all-MiniLM-L6-v2 model (Zero RAM usage)
        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction",
            huggingfacehub_api_token=token,
        )
        return embeddings
    except Exception as e:
        error = CustomException("Failed to initialize HuggingFace embeddings", e)
        logger.error(str(error))
        return None