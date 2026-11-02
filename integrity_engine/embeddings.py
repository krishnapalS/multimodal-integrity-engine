"""Sentence embedding wrapper.

all-MiniLM-L6-v2 is chosen deliberately: 384-dim, ~90MB, strong
paraphrase retrieval on CPU, and explicitly suggested by the brief.
Vectors are L2-normalized so inner product == cosine similarity.
"""
import numpy as np


class Embedder:
    def __init__(self, model_name: str, batch_size: int = 16):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size

    def encode(self, texts: list) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32")

    def unload(self):
        """Free the model; the pipeline's stages run sequentially and the
        container has limited RAM."""
        import gc
        del self._model
        self._model = None
        gc.collect()
