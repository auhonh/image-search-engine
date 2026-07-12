"""Abstract ANN index interface. Both LSH and HNSW implement this,
so the API and benchmarks can swap implementations with one line."""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Index(ABC):
    """Vectors are assumed L2-normalized, so dot product == cosine sim."""

    @abstractmethod
    def add(self, vec: np.ndarray, item_id: int) -> None:
        """Insert one vector with its external item id."""

    @abstractmethod
    def query(self, vec: np.ndarray, k: int = 20) -> list[int]:
        """Return up to k item ids, most similar first."""

    def add_batch(self, vecs: np.ndarray, ids: list[int]) -> None:
        for v, i in zip(vecs, ids):
            self.add(v, i)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "Index":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, Index):
            raise TypeError(f"{path} did not contain an Index")
        return obj


class BruteForceIndex(Index):
    """Exact search baseline — the ground truth for recall benchmarks."""

    def __init__(self):
        self._vecs: list[np.ndarray] = []
        self._ids: list[int] = []
        self._matrix: np.ndarray | None = None

    def add(self, vec: np.ndarray, item_id: int) -> None:
        self._vecs.append(vec.astype(np.float32))
        self._ids.append(item_id)
        self._matrix = None  # invalidate cache

    def query(self, vec: np.ndarray, k: int = 20) -> list[int]:
        if not self._vecs:
            return []
        if self._matrix is None:
            self._matrix = np.stack(self._vecs)
        sims = self._matrix @ vec.astype(np.float32)
        top = np.argsort(sims)[::-1][:k]
        return [self._ids[i] for i in top]
