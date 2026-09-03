import os
import sys
import pickle
import io
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from app.components.embeddings import get_embedding_model
from app.common.logger import get_logger

logger = get_logger(__name__)

# Custom unpickler class to intercept Document objects before Pydantic fails
class LenientDocument:
    def __init__(self, *args, **kwargs):
        self.page_content = ""
        self.metadata = {}

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.page_content = state.get("page_content", "")
            self.metadata = state.get("metadata", {})
            # Read __dict__ if nested
            if not self.page_content and "__dict__" in state:
                self.page_content = state["__dict__"].get("page_content", "")
                self.metadata = state["__dict__"].get("metadata", {})
        elif isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            self.page_content = state[1].get("page_content", "")
            self.metadata = state[1].get("metadata", {})

class SafeDocstoreUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Intercept any Document class and substitute with LenientDocument
        if name == "Document" and ("langchain" in module or "schema" in module or "pydantic" in module):
            return LenientDocument
        return super().find_class(module, name)

def load_clean_docstore(file_path):
    with open(file_path, "rb") as f:
        unpickler = SafeDocstoreUnpickler(f)
        raw = unpickler.load()

    docstore, index_to_docstore_id = raw

    cleaned_dict = {}
    underlying_dict = getattr(docstore, "_dict", docstore)

    for doc_id, doc in underlying_dict.items():
        content = getattr(doc, "page_content", "")
        meta = getattr(doc, "metadata", {})
        cleaned_dict[doc_id] = Document(page_content=str(content), metadata=dict(meta) if meta else {})

    new_docstore = InMemoryDocstore(cleaned_dict)
    return new_docstore, index_to_docstore_id

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

    # Safely load docstore using custom unpickler
    docstore, index_to_docstore_id = load_clean_docstore(pkl_file)

    # Reconstruct the LangChain FAISS instance
    vector_store = FAISS(
        embedding_function=embedding_model,
        index=raw_index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    print("[DEBUG] Vector store loaded successfully with clean Document objects!", file=sys.stderr, flush=True)
    return vector_store