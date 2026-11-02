"""FAISS vector index over chunk embeddings.

IndexFlatIP over L2-normalized vectors gives exact cosine similarity.
Flat (not HNSW/IVF) is the right call at prototype scale (~10^2-10^3
vectors): exact results, zero tuning. The README scaling brief covers the
migration path to approximate indexes for 10k+ documents.
"""
import faiss
import numpy as np


class VectorIndex:
    def __init__(self, dim: int):
        self._index = faiss.IndexFlatIP(dim)
        self.records = []  # parallel list of metadata dicts per vector

    def add(self, vectors: np.ndarray, records: list):
        assert len(vectors) == len(records)
        self._index.add(vectors)
        self.records.extend(records)

    def search(self, vectors: np.ndarray, k: int) -> tuple:
        """Return (similarities, indices) of the k nearest neighbours."""
        k = min(k, len(self.records))
        if k == 0:
            empty = np.empty((len(vectors), 0), dtype="float32")
            return empty, np.full((len(vectors), 0), -1)
        return self._index.search(vectors, k)

    def __len__(self):
        return len(self.records)
