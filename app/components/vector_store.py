import os
import sys
import pickle
import io
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from app.components.embeddings import get_embedding_model
from app.common.logger import get_logger

logger = get_logger(__name__)

# Custom unpickler to ignore missing Pydantic private attributes like '__fields_set__'
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(module, name)

def safe_load_pickle(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    
    # Patch __setstate__ for Pydantic objects during deserialization
    try:
        return pickle.loads(data)
    except (KeyError, AttributeError):
        print("[DEBUG] Using fallback dict unpickling for Pydantic mismatch...", file=sys.stderr, flush=True)
        
        # Monkeypatch Document.__setstate__ temporarily if Pydantic fails
        from langchain_core.documents import Document
        orig_setstate = getattr(Document, "__setstate__", None)
        
        def lenient_setstate(self, state):
            if isinstance(state, dict):
                # Remove internal pydantic fields if present
                state.pop("__fields_set__", None)
                state.pop("__pydantic_extra__", None)
                state.pop("__pydantic_fields_set__", None)
                if hasattr(self, "__dict__"):
                    self.__dict__.update(state)
            elif isinstance(state, tuple) and len(state) == 2:
                self.__dict__.update(state[1])
                
        Document.__setstate__ = lenient_setstate
        try:
            res = pickle.loads(data)
            return res
        finally:
            if orig_setstate:
                Document.__setstate__ = orig_setstate

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

    faiss_file = os.path.join(index_dir, "index.faiss")
    pkl_file = os.path.join(index_dir, "index.pkl")

    # Load FAISS index file directly
    raw_index = faiss.read_index(faiss_file)

    # Safely load the docstore pickle
    docstore, index_to_docstore_id = safe_load_pickle(pkl_file)

    # Reconstruct the LangChain FAISS instance
    vector_store = FAISS(
        embedding_function=embedding_model,
        index=raw_index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    print("[DEBUG] Vector store loaded successfully!", file=sys.stderr, flush=True)
    return vector_store