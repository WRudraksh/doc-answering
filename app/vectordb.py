import faiss
import numpy as np
import pickle
from typing import List, Dict

class FaissStore:
    def __init__(self, dim:int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.metadatas: List[Dict] = []

    def save(self, path_prefix: str):
        faiss.write_index(self.index, f"{path_prefix}_index.index")
        with open(f"{path_prefix}_meta.pkl", "wb") as f:
            pickle.dump(self.metadatas, f)


    @classmethod
    def load(cls, path_prefix: str, dim: int):
        inst = cls(dim)
        inst.index = faiss.read_index(f"{path_prefix}_index.index")
        with open(f"{path_prefix}_meta.pkl", "rb") as f:
            inst.metadatas = pickle.load(f)
        return inst
    

    def add(self, vectors: List[List[float]], metadatas: List[Dict]):
        if len(vectors) != len(metadatas):
            raise ValueError("Length of vectors and metadatas must be the same.")
        np_vectors = np.array(vectors).astype("float32")
        self.index.add(np_vectors)
        self.metadatas.extend(metadatas)

    
