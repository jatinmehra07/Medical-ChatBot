import os
import sys
from langchain_community.vectorstores import FAISS
from app.components.embeddings import get_embedding_model
from app.common.logger import get_logger

logger = get_logger(__name__)

def get_faiss_directory():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
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
        if os.path.exists(target_file):
            return os.path.abspath(p)

    for root, _, files in os.walk(base_dir):
        if "index.faiss" in files:
            return root

    return None

def load_vector_store():
    embedding_model = get_embedding_model()
    if embedding_model is None:
        raise ValueError("Embedding model could not be initialized.")

    index_dir = get_faiss_directory()
    if not index_dir:
        raise FileNotFoundError("index.faiss was not found in the project directory.")

    print(f"[DEBUG] Loading FAISS index locally from: {index_dir}", file=sys.stderr, flush=True)

    try:
        vector_store = FAISS.load_local(
            folder_path=index_dir,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
    except KeyError as e:
        if "__fields_set__" in str(e):
            print("[DEBUG] Caught Pydantic v1/v2 unpickling mismatch. Re-reading with compatibility fix...", file=sys.stderr, flush=True)
            # Re-read index directly using FAISS read_index if pickle docstore has schema mismatch
            import faiss
            import pickle
            
            index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
            with open(os.path.join(index_dir, "index.pkl"), "rb") as f:
                docstore, index_to_docstore_id = pickle.load(f)
            
            vector_store = FAISS(
                embedding_function=embedding_model,
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id
            )
        else:
            raise e

    print("[DEBUG] Vector store loaded successfully!", file=sys.stderr, flush=True)
    return vector_store