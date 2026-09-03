import os
from dotenv import load_dotenv

load_dotenv()

# Base project directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Directory paths
DATA_PATH = os.path.join(BASE_DIR, "data")
DB_FAISS_PATH = os.path.join(BASE_DIR, "vectorstore")

# Text splitter parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# HuggingFace & Model settings
HUGGINGFACE_REPO_ID = os.getenv("HUGGINGFACE_REPO_ID", "sentence-transformers/all-MiniLM-L6-v2")
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")