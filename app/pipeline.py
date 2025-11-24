import os
import uuid

from app.ingest import ingest_document
from app.chunking import chunk_text
from app.embeddings import embed_texts
from app.vectordb import FaissStore
from app.retriever import retrieve_top_k, build_context, build_prompt
from app.llm_client import get_llm_client, ask_llm

BASE_PATH = "data/documents"

def generate_doc_id():
    return str(uuid.uuid4())


# BUILD DOCUMENT INDEX

def build_document_index(file_path):

    # 1. Ingest document
    text = ingest_document(file_path)

    # 2. Chunk text
    chunks = chunk_text(text)
    chunks_text = [c["text"] for c in chunks]

    # 3. Generate embeddings using HuggingFace
    vectors = embed_texts(chunks_text)

    # 4. Determine embedding dimension (should be 768)
    dim = len(vectors[0])

    # 5. Create FAISS store
    store = FaissStore(dim=dim)
    store.add(vectors, chunks)

    # 6. Generate unique document ID
    doc_id = generate_doc_id()

    # 7. Save FAISS files
    save_path = os.path.join(BASE_PATH, doc_id)
    os.makedirs(save_path, exist_ok=True)
    store.save(save_path)

    return doc_id


# ANSWER QUESTION

def answer_question(doc_id: str, question: str, k: int = 5) -> str:

    # 1. Load FAISS store
    load_path = os.path.join(BASE_PATH, doc_id)
    store = FaissStore.load(load_path, dim=768)   # HuggingFace dimension

    # 2. Embed the question using the SAME HF model
    query_vector = embed_texts([question])[0]     # returns list → take first

    # 3. Retrieve top-k chunks
    results = retrieve_top_k(store, query_vector, k)

    # 4. Build context
    context = build_context(results)

    # 5. Create final prompt
    prompt = build_prompt(question, context)

    # 6. Ask Groq LLM
    llm_client = get_llm_client()
    answer = ask_llm(llm_client, prompt)

    return answer
