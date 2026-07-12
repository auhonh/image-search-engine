"""Embed the corpus and build serialized indexes.

    python -m scripts.build_index            # embeds + builds LSH and HNSW
    python -m scripts.build_index --skip-embed   # reuse existing embeddings
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from embeddings.build_corpus import build_corpus
from index import HNSWIndex, LSHIndex

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
EMB = ROOT / "data" / "embeddings"
CACHE = ROOT / "data" / "index_cache"


def main(skip_embed: bool) -> None:
    if skip_embed and (EMB / "corpus_vecs.npy").exists():
        vecs = np.load(EMB / "corpus_vecs.npy")
        with open(EMB / "id_map.json") as f:
            id_map = {int(k): v for k, v in json.load(f).items()}
        print(f"Reusing embeddings: {vecs.shape}")
    else:
        vecs, id_map = build_corpus(CORPUS, EMB)

    ids = list(id_map.keys())
    CACHE.mkdir(parents=True, exist_ok=True)

    for name, idx in [
        ("lsh", LSHIndex(dim=vecs.shape[1])),
        ("hnsw", HNSWIndex()),
    ]:
        t0 = time.perf_counter()
        idx.add_batch(vecs, ids)
        idx.save(CACHE / f"{name}.pkl")
        print(f"Built {name} in {time.perf_counter() - t0:.1f}s -> {CACHE / (name + '.pkl')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip-embed", action="store_true")
    main(p.parse_args().skip_embed)
