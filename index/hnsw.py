"""Hierarchical Navigable Small World (simplified but faithful).

Structure: a stack of NSW graphs. Every node exists at layer 0; each node is
also promoted to higher layers with exponentially decaying probability
(P(level >= l) = e^(-l/mL)). Queries start at the sparse top layer, greedily
descend to a good entry point, then run a wider beam search at layer 0.

This gives O(log N)-ish hop counts vs O(sqrt N) for a flat NSW.
"""

import heapq
import math

import numpy as np

from .base import Index


class HNSWIndex(Index):
    def __init__(self, M: int = 16, ef_construction: int = 100,
                 ef_search: int = 50, seed: int = 42):
        self.M = M
        self.M0 = 2 * M                     # layer 0 gets double connectivity
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.mL = 1.0 / math.log(M)
        self._rng = np.random.default_rng(seed)

        self.vectors: dict[int, np.ndarray] = {}
        # layers[l][node] -> neighbor list; layer 0 contains every node
        self.layers: list[dict[int, list[int]]] = [dict()]
        self._entry: int | None = None
        self._entry_level: int = -1

    # ---------- internals ----------

    def _sim(self, node: int, vec: np.ndarray) -> float:
        return float(self.vectors[node] @ vec)

    def _random_level(self) -> int:
        return int(-math.log(self._rng.uniform(1e-12, 1.0)) * self.mL)

    def _greedy_step(self, query: np.ndarray, entry: int, layer: int) -> int:
        """Hill-climb to the local best node at this layer (used while descending)."""
        current = entry
        current_sim = self._sim(current, query)
        improved = True
        while improved:
            improved = False
            for nb in self.layers[layer].get(current, []):
                s = self._sim(nb, query)
                if s > current_sim:
                    current, current_sim = nb, s
                    improved = True
        return current

    def _beam_search(self, query: np.ndarray, entry: int,
                     layer: int, ef: int) -> list[int]:
        visited = {entry}
        candidates = [(-self._sim(entry, query), entry)]
        results = [(self._sim(entry, query), entry)]
        graph = self.layers[layer]

        while candidates:
            neg_sim, node = heapq.heappop(candidates)
            if len(results) >= ef and -neg_sim < results[0][0]:
                break
            for nb in graph.get(node, []):
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
        """Diversity heuristic from the HNSW paper: keep a candidate only if
        it is more similar to the target than to any already-kept neighbor.
        Preserves long-range links so clusters stay mutually reachable."""
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
        for c in ordered:
            if len(selected) >= m:
                break
            if c not in selected:
                selected.append(c)
        return selected

    def _connect(self, node: int, neighbors: list[int], layer: int) -> None:
        max_conn = self.M0 if layer == 0 else self.M
        graph = self.layers[layer]
        chosen = self._select_neighbors(self.vectors[node], neighbors, max_conn)
        graph[node] = list(chosen)
        for nb in chosen:
            graph.setdefault(nb, []).append(node)
            if len(graph[nb]) > max_conn:
                graph[nb] = self._select_neighbors(
                    self.vectors[nb], graph[nb], max_conn
                )

    # ---------- public API ----------

    def add(self, vec: np.ndarray, item_id: int) -> None:
        vec = vec.astype(np.float32)
        self.vectors[item_id] = vec
        level = self._random_level()

        while len(self.layers) <= level:
            self.layers.append(dict())

        if self._entry is None:
            for l in range(level + 1):
                self.layers[l][item_id] = []
            self._entry, self._entry_level = item_id, level
            return

        entry = self._entry
        # Descend from the top of the structure to level+1 greedily
        for l in range(self._entry_level, level, -1):
            if l < len(self.layers) and entry in self.layers[l]:
                entry = self._greedy_step(vec, entry, l)

        # Insert with beam search at each layer from min(level, entry_level) down
        for l in range(min(level, self._entry_level), -1, -1):
            found = self._beam_search(vec, entry, l, self.ef_construction)
            self._connect(item_id, found, l)
            entry = found[0] if found else entry

        # Ensure node exists on all its layers even if search found nothing
        for l in range(level + 1):
            self.layers[l].setdefault(item_id, [])

        if level > self._entry_level:
            self._entry, self._entry_level = item_id, level

    def query(self, vec: np.ndarray, k: int = 20) -> list[int]:
        if self._entry is None:
            return []
        vec = vec.astype(np.float32)
        entry = self._entry
        for l in range(self._entry_level, 0, -1):
            entry = self._greedy_step(vec, entry, l)
        results = self._beam_search(vec, entry, 0, max(self.ef_search, k))
        return results[:k]

    def stats(self) -> dict:
        return {
            "n_items": len(self.vectors),
            "n_layers": len(self.layers),
            "nodes_per_layer": [len(g) for g in self.layers],
            "M": self.M,
        }
