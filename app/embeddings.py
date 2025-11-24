import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def embed_texts(text: list):
    vectors = model.encode(text, normalize_embeddings=True)
    return vectors.tolist()
