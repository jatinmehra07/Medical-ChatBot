import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

load_dotenv()
logger = get_logger(__name__)

def get_active_model(api_key):
    """Fetch all active chat models for this API key and pick the first match."""
    preferred_order = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "gemma2-9b-it",
        "llama3-70b-8192",
        "llama3-8b-8192",
    ]
    try:
        url = "https://api.groq.com/openai/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()

        available_ids = [
            m["id"] for m in data.get("data", [])
            if "whisper" not in m["id"].lower() and "guard" not in m["id"].lower()
        ]
        logger.info(f"Available Groq models on account: {available_ids}")

        # Pick highest preference matching available models
        for model in preferred_order:
            if model in available_ids:
                return model

        # Fallback to the first available text model if none of the preferred match
        if available_ids:
            return available_ids[0]

    except Exception as e:
        logger.warning(f"Could not fetch dynamic model list: {e}")

    # Default fallback
    return "gemma2-9b-it"

def load_llm():
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise CustomException("GROQ_API_KEY environment variable is not set in .env")

        selected_model = get_active_model(api_key)
        logger.info(f"Loading LLM from Groq with model: {selected_model}")

        llm = ChatGroq(
            model_name=selected_model,
            groq_api_key=api_key,
            temperature=0.2
        )
        logger.info("LLM loaded successfully from Groq.")
        return llm

    except Exception as e:
        error_message = CustomException("Failed to load an LLM from Groq", e)
        logger.error(str(error_message))
        return None