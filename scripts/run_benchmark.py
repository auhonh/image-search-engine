"""Benchmark LSH and HNSW against brute force on held-out query vectors.

    python -m scripts.run_benchmark --n-queries 100 --k 10
"""

import argparse
import json
from pathlib import Path

import numpy as np

from index import HNSWIndex, Index, LSHIndex, run_benchmark

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "data" / "embeddings"
CACHE = ROOT / "data" / "index_cache"
RESULTS = ROOT / "results"


def main(n_queries: int, k: int, seed: int) -> None:
    vecs = np.load(EMB / "corpus_vecs.npy")
    with open(EMB / "id_map.json") as f:
        ids = [int(i) for i in json.load(f).keys()]

    # Hold out query vectors; index the rest
    rng = np.random.default_rng(seed)
    q_idx = rng.choice(len(ids), size=min(n_queries, len(ids) // 5), replace=False)
    q_mask = np.zeros(len(ids), dtype=bool)
    q_mask[q_idx] = True
    corpus_vecs, corpus_ids = vecs[~q_mask], [i for i, m in zip(ids, q_mask) if not m]
    query_vecs = vecs[q_mask]

    indexes: dict[str, Index] = {
        "lsh": LSHIndex(dim=vecs.shape[1]),
        "hnsw": HNSWIndex(),
    }
    for idx in indexes.values():
        idx.add_batch(corpus_vecs, corpus_ids)

    results = run_benchmark(indexes, corpus_vecs, corpus_ids, query_vecs, k=k)

    RESULTS.mkdir(exist_ok=True)
    out = [r.to_dict() for r in results]
    with open(RESULTS / "benchmark_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'index':<8}{'recall@' + str(k):<12}{'mean ms':<10}{'speedup':<8}")
    for r in results:
        print(f"{r.index_name:<8}{r.recall_at_k:<12}{r.mean_query_ms:<10}{r.speedup_vs_brute}x")
    print(f"\nWritten to {RESULTS / 'benchmark_results.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-queries", type=int, default=100)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    main(a.n_queries, a.k, a.seed)
