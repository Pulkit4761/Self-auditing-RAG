import numpy as np
from sentence_transformers import SentenceTransformer
from . import config

class Embedder:
    def __init__(self):
        self._model = SentenceTransformer(config.EMBEDDING_MODEL)

    def embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return np.array(embeddings, dtype="float32")
