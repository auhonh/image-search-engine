"""Navigable Small World graph — HNSW's layer 0 in isolation.

Each inserted node is connected (bidirectionally) to its M approximate
nearest neighbors found by greedy beam search over the existing graph.
Queries greedily walk the graph toward the query vector.
"""

import heapq

import numpy as np

from .base import Index


class NSWIndex(Index):
    def __init__(self, M: int = 16, ef: int = 50, seed: int = 42):
        self.M = M            # max connections per node
        self.ef = ef          # beam width during search
        self.graph: dict[int, list[int]] = {}
        self.vectors: dict[int, np.ndarray] = {}
        self._entry: int | None = None
        self._rng = np.random.default_rng(seed)

    def _sim(self, node: int, vec: np.ndarray) -> float:
        return float(self.vectors[node] @ vec)

    def _beam_search(self, query: np.ndarray, ef: int) -> list[int]:
        """Best-first search with a bounded candidate beam.
        Returns visited nodes sorted by similarity (best first)."""
        if self._entry is None:
            return []
        entry = self._entry
        visited = {entry}
        # Min-heap on negative similarity = max-heap on similarity
        candidates = [(-self._sim(entry, query), entry)]
        # Results: min-heap on similarity so we can evict the worst
        results = [(self._sim(entry, query), entry)]

        while candidates:
            neg_sim, node = heapq.heappop(candidates)
            # Stop when best remaining candidate is worse than the worst result
            if len(results) >= ef and -neg_sim < results[0][0]:
                break
            for nb in self.graph.get(node, []):
                if nb in visited:
                    continue
                visited.add(nb)
                s = self._sim(nb, query)
                if len(results) < ef or s > results[0][0]:
                    heapq.heappush(candidates, (-s, nb))
                    heapq.heappush(results, (s, nb))
                    if len(results) > ef:
                        heapq.heappop(results)

        return [n for _, n in sorted(results, reverse=True)]

    def _select_neighbors(self, vec: np.ndarray, candidates: list[int],
                          m: int) -> list[int]:
        """HNSW paper heuristic: keep a candidate only if it is more similar
        to the new node than to any already-selected neighbor. This preserves
        diverse, long-range links instead of clustering all edges locally —
        without it, tightly clustered data fragments the graph and greedy
        search can't cross between clusters."""
        ordered = sorted(candidates, key=lambda c: self._sim(c, vec), reverse=True)
        selected: list[int] = []
        for c in ordered:
            if len(selected) >= m:
                break
            sim_to_new = self._sim(c, vec)
            if all(
                sim_to_new >= float(self.vectors[c] @ self.vectors[s])
                for s in selected
            ):
                selected.append(c)
        # Backfill with nearest remaining if heuristic was too strict
        for c in ordered:
            if len(selected) >= m:
                break
            if c not in selected:
                selected.append(c)
        return selected

    def add(self, vec: np.ndarray, item_id: int) -> None:
        vec = vec.astype(np.float32)
        self.vectors[item_id] = vec
        self.graph[item_id] = []
        if self._entry is None:
            self._entry = item_id
            return

        found = self._beam_search(vec, ef=max(self.ef, self.M))
        neighbors = self._select_neighbors(vec, found, self.M)
        self.graph[item_id] = list(neighbors)
        for nb in neighbors:
            self.graph[nb].append(item_id)
            if len(self.graph[nb]) > self.M:
                self.graph[nb] = self._select_neighbors(
                    self.vectors[nb], self.graph[nb], self.M
                )

    def query(self, vec: np.ndarray, k: int = 20) -> list[int]:
        return self._beam_search(vec.astype(np.float32), ef=max(self.ef, k))[:k]
