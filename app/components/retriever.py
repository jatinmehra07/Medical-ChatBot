import sys
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from app.components.llm import load_llm
from app.components.vector_store import load_vector_store
from app.common.logger import get_logger

logger = get_logger(__name__)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.\n\n"
    "{context}"
)

def create_qa_chain():
    try:
        print("[DEBUG] Attempting to load LLM...", file=sys.stderr, flush=True)
        llm = load_llm()
        if llm is None:
            raise RuntimeError("load_llm() returned None. Check GROQ_API_KEY.")

        print("[DEBUG] Attempting to load Vector Store...", file=sys.stderr, flush=True)
        vector_store = load_vector_store()
        if vector_store is None:
            raise RuntimeError("load_vector_store() returned None. Check FAISS index path and HF_TOKEN.")

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        print("[DEBUG] QA chain built successfully!", file=sys.stderr, flush=True)
        return rag_chain

    except Exception as e:
        print(f"[DEBUG ERROR IN RETRIEVER]: {str(e)}", file=sys.stderr, flush=True)
        # Raise the actual error instead of returning None so application.py catches it
        raise e