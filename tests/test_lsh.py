import numpy as np
import pytest

from index import BruteForceIndex, HNSWIndex, LSHIndex, NSWIndex


def make_clusters(n_clusters=5, per_cluster=40, dim=64, seed=0):
    """Synthetic unit vectors in tight clusters — items in the same cluster
    are near neighbors by construction."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    vecs, labels = [], []
    for c in range(n_clusters):
        pts = centers[c] + 0.05 * rng.standard_normal((per_cluster, dim))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        vecs.append(pts.astype(np.float32))
        labels += [c] * per_cluster
    return np.vstack(vecs), np.array(labels)


@pytest.fixture(scope="module")
def data():
    return make_clusters()


@pytest.mark.parametrize("index_cls,kwargs", [
    (LSHIndex, {"dim": 64, "n_tables": 12, "n_bits": 8}),
    (NSWIndex, {"M": 8, "ef": 40}),
    (HNSWIndex, {"M": 8, "ef_construction": 60, "ef_search": 40}),
])
def test_same_cluster_retrieved(data, index_cls, kwargs):
    vecs, labels = data
    idx = index_cls(**kwargs)
    idx.add_batch(vecs, list(range(len(vecs))))

    hits = total = 0
    for q in range(0, len(vecs), 20):
        results = idx.query(vecs[q], k=10)
        assert results, "index returned no candidates"
        same = sum(labels[r] == labels[q] for r in results)
        hits += same
        total += len(results)
    # Clustered data: the vast majority of neighbors should share the label
    assert hits / total > 0.8


def test_recall_vs_bruteforce(data):
    vecs, _ = data
    ids = list(range(len(vecs)))
    brute, lsh = BruteForceIndex(), LSHIndex(dim=64, n_tables=16, n_bits=8)
    brute.add_batch(vecs, ids)
    lsh.add_batch(vecs, ids)

    k, hits = 10, 0
    queries = range(0, len(vecs), 10)
    for q in queries:
        truth = set(brute.query(vecs[q], k=k))
        pred = set(lsh.query(vecs[q], k=k))
        hits += len(truth & pred)
    recall = hits / (len(list(queries)) * k)
    assert recall > 0.7, f"LSH recall too low: {recall:.2f}"


def test_query_empty_index():
    assert LSHIndex(dim=8).query(np.ones(8, dtype=np.float32)) == []
    assert HNSWIndex().query(np.ones(8, dtype=np.float32)) == []
