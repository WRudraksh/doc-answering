from app.embeddings import embed

def embed_question(client, question: str, model="text-embedding-3-small"):
    vector = embed(client, [question], model=model)[0]
    return vector

def retriever_top_k(store, query_vector, k=5):
    results = store.search(query_vector, k)
    return results

def build_context(results):
    contex_parts = []
    for item in results:
        metadata = item["metadata"]
        text = metadata.get("text", "")
        contex_parts.append(text)

    return "\n---\n".join(contex_parts)

def build_prompt(context: str, question: str) -> str:
    prompt = f"""Use the following context to answer the question.
If context does not contain the answer, say "I don't know

Context:
{context}

Question: {question}
Answer:"""
    return prompt
