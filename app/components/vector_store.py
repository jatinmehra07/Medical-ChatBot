import os
import sys
import pickle
import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from app.components.embeddings import get_embedding_model
from app.common.logger import get_logger

logger = get_logger(__name__)

def load_clean_docstore(file_path):
    with open(file_path, "rb") as f:
        raw = pickle.load(f)
    
    docstore, index_to_docstore_id = raw

    # Ensure all items inside docstore._dict are fully formed Document objects with page_content
    cleaned_dict = {}
    underlying_dict = getattr(docstore, "_dict", docstore)

    for doc_id, doc in underlying_dict.items():
        if isinstance(doc, Document):
            # If page_content was lost during pickle state mismatch, recover it from instance attributes
            content = getattr(doc, "page_content", None) or getattr(doc, "__dict__", {}).get("page_content", "")
            meta = getattr(doc, "metadata", None) or getattr(doc, "__dict__", {}).get("metadata", {})
            cleaned_dict[doc_id] = Document(page_content=str(content), metadata=dict(meta) if meta else {})
        elif isinstance(doc, dict):
            cleaned_dict[doc_id] = Document(
                page_content=str(doc.get("page_content", "")),
                metadata=doc.get("metadata", {})
            )
        else:
            # Fallback for raw text strings or tuple objects
            content = getattr(doc, "page_content", str(doc))
            cleaned_dict[doc_id] = Document(page_content=str(content))

    if hasattr(docstore, "_dict"):
        docstore._dict = cleaned_dict
        return docstore, index_to_docstore_id
    else:
        from langchain_community.docstore.in_memory import InMemoryDocstore
        return InMemoryDocstore(cleaned_dict), index_to_docstore_id

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

    # Safely load and convert docstore items into genuine Document objects
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