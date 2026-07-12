"""Locality-Sensitive Hashing via random hyperplane projections.

Each table hashes a vector to an n_bits binary code by taking the sign of
n_bits random projections. Similar vectors (small angle) agree on most signs,
so they land in the same bucket with high probability.

  n_tables ↑  ->  recall ↑  (more chances to collide with true neighbors)
  n_bits   ↑  ->  precision ↑, bucket size ↓ (fewer false candidates)
"""

import numpy as np

from .base import Index


class LSHIndex(Index):
    def __init__(
        self,
        dim: int,
        n_tables: int = 12,
        n_bits: int = 10,
        seed: int = 42,
    ):
        self.dim = dim
        self.n_tables = n_tables
        self.n_bits = n_bits
        rng = np.random.default_rng(seed)
        # (n_tables, n_bits, dim) random hyperplane normals
        self.planes = rng.standard_normal(
            (n_tables, n_bits, dim), dtype=np.float32
        )
        # Precompute powers of 2 for fast bit packing
        self._pow2 = (1 << np.arange(n_bits)).astype(np.int64)
        self.tables: list[dict[int, list[int]]] = [dict() for _ in range(n_tables)]
        self._vecs: list[np.ndarray] = []
        self._ids: list[int] = []

    def _hashes(self, vec: np.ndarray) -> np.ndarray:
        """Return the bucket key for this vector in every table. Shape (n_tables,)."""
        projections = self.planes @ vec.astype(np.float32)  # (n_tables, n_bits)
        bits = (projections > 0).astype(np.int64)
        return bits @ self._pow2

    def add(self, vec: np.ndarray, item_id: int) -> None:
        internal = len(self._vecs)
        self._vecs.append(vec.astype(np.float32))
        self._ids.append(item_id)
        for t, h in enumerate(self._hashes(vec)):
            self.tables[t].setdefault(int(h), []).append(internal)

    def query(self, vec: np.ndarray, k: int = 20,
              multiprobe: bool = True) -> list[int]:
        candidates: set[int] = set()
        hashes = self._hashes(vec)
        for t, h in enumerate(hashes):
            candidates.update(self.tables[t].get(int(h), []))

        # Multi-probe: if the exact buckets are too sparse, also probe every
        # Hamming-distance-1 bucket (flip one bit of each table's key).
        # Vectors near a hyperplane boundary often differ by exactly one bit.
        if multiprobe and len(candidates) < 3 * k:
            for t, h in enumerate(hashes):
                for bit in range(self.n_bits):
                    candidates.update(
                        self.tables[t].get(int(h) ^ (1 << bit), [])
                    )
                if len(candidates) >= 10 * k:
                    break

        if not candidates:
            return []
        cands = list(candidates)
        mat = np.stack([self._vecs[c] for c in cands])
        sims = mat @ vec.astype(np.float32)
        top = np.argsort(sims)[::-1][:k]
        return [self._ids[cands[i]] for i in top]

    def stats(self) -> dict:
        """Diagnostics for the write-up: bucket occupancy per table."""
        sizes = [len(bucket) for tbl in self.tables for bucket in tbl.values()]
        return {
            "n_items": len(self._vecs),
            "n_tables": self.n_tables,
            "n_bits": self.n_bits,
            "avg_bucket_size": float(np.mean(sizes)) if sizes else 0.0,
            "max_bucket_size": int(np.max(sizes)) if sizes else 0,
        }
