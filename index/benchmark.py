"""Benchmark ANN indexes against exact brute-force search.

Reports recall@K (fraction of true top-K neighbors the ANN index finds)
and mean query latency, per index. These numbers back the README table.
"""

import time
from dataclasses import dataclass, asdict

import numpy as np

from .base import BruteForceIndex, Index


@dataclass
class BenchmarkResult:
    index_name: str
    n_corpus: int
    n_queries: int
    k: int
    recall_at_k: float
    mean_query_ms: float
    speedup_vs_brute: float

    def to_dict(self) -> dict:
        return asdict(self)


def _timed_queries(index: Index, queries: np.ndarray, k: int):
    results, times = [], []
    for q in queries:
        t0 = time.perf_counter()
        results.append(index.query(q, k=k))
        times.append(time.perf_counter() - t0)
    return results, float(np.mean(times)) * 1000.0


def run_benchmark(
    indexes: dict[str, Index],
    corpus_vecs: np.ndarray,
    corpus_ids: list[int],
    query_vecs: np.ndarray,
    k: int = 10,
) -> list[BenchmarkResult]:
    brute = BruteForceIndex()
    brute.add_batch(corpus_vecs, corpus_ids)
    truth, brute_ms = _timed_queries(brute, query_vecs, k)

    out = []
    for name, idx in indexes.items():
        preds, ann_ms = _timed_queries(idx, query_vecs, k)
        hits = sum(
            len(set(p) & set(t)) for p, t in zip(preds, truth)
        )
        recall = hits / (len(query_vecs) * k)
        out.append(BenchmarkResult(
            index_name=name,
            n_corpus=len(corpus_ids),
            n_queries=len(query_vecs),
            k=k,
            recall_at_k=round(recall, 4),
            mean_query_ms=round(ann_ms, 3),
            speedup_vs_brute=round(brute_ms / ann_ms, 2) if ann_ms > 0 else 0.0,
        ))
    return out
