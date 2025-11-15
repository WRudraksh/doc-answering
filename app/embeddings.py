from openai import OpenAI
from app.config import OPENAI_API_KEY

def get_embedding_client():
    api_key = OPENAI_API_KEY
    return OpenAI(api_key=api_key)

def embed(client, texts:list, model: str="text-embedding-3-small"):
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    embeddings = [data_point.embedding for data_point in response.data]
    return embeddings