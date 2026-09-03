import os
import sys
from langchain_community.vectorstores import FAISS
from app.components.embeddings import get_embedding_model
from app.common.logger import get_logger

logger = get_logger(__name__)

def get_faiss_directory():
    # 1. Project root relative to this file
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    # Check multiple candidate directories where index.faiss might live
    candidate_paths = [
        os.path.join(base_dir, "vectorstore"),
        os.path.join(base_dir, "vectorstore", "db_faiss"),
        os.path.join(os.getcwd(), "vectorstore"),
        os.path.join(os.getcwd(), "vectorstore", "db_faiss"),
        "vectorstore",
        "vectorstore/db_faiss",
    ]

    for p in candidate_paths:
        target_file = os.path.join(p, "index.faiss")
        exists = os.path.exists(target_file)
        print(f"[DEBUG PATH CHECK] Checking '{target_file}' -> Exists: {exists}", file=sys.stderr, flush=True)
        if exists:
            return os.path.abspath(p)

    # If not found, walk the project tree to locate any .faiss file on Render
    print("[DEBUG PATH CHECK] Searching all files for .faiss...", file=sys.stderr, flush=True)
    for root, _, files in os.walk(base_dir):
        if "index.faiss" in files:
            print(f"[DEBUG PATH CHECK] Found index.faiss in: {root}", file=sys.stderr, flush=True)
            return root

    return None

def load_vector_store():
    embedding_model = get_embedding_model()
    if embedding_model is None:
        raise ValueError("Embedding model could not be initialized.")

    index_dir = get_faiss_directory()
    if not index_dir:
        raise FileNotFoundError(
            "index.faiss was NOT found anywhere in the repository on Render. "
            "Make sure 'vectorstore/index.faiss' and 'vectorstore/index.pkl' are committed to Git."
        )

    print(f"[DEBUG] Loading FAISS index locally from: {index_dir}", file=sys.stderr, flush=True)
    
    # Load the index with dangerous deserialization permitted for local pkl files
    vector_store = FAISS.load_local(
        folder_path=index_dir,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    print("[DEBUG] Vector store loaded successfully!", file=sys.stderr, flush=True)
    return vector_store