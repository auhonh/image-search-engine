"""Singleton application state. ResNet-50 + the ANN index load once at
startup (several seconds), not per-request."""

import json
from pathlib import Path

import numpy as np

from embeddings import ImageEncoder
from index import Index, LSHIndex
from ranking import Ranker

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EMB_DIR = DATA / "embeddings"
CACHE_DIR = DATA / "index_cache"
CORPUS_DIR = DATA / "corpus"


class AppState:
    encoder: ImageEncoder
    index: Index
    ranker: Ranker
    corpus_vecs: np.ndarray
    corpus_paths: dict[int, str]

    def load(self, index_name: str = "lsh") -> None:
        self.encoder = ImageEncoder()
        self.corpus_vecs = np.load(EMB_DIR / "corpus_vecs.npy")
        with open(EMB_DIR / "id_map.json") as f:
            raw = json.load(f)
        self.corpus_paths = {int(k): str(CORPUS_DIR / v) for k, v in raw.items()}

        cache = CACHE_DIR / f"{index_name}.pkl"
        if cache.exists():
            self.index = Index.load(cache)
            print(f"Loaded cached {index_name} index ({cache})")
        else:
            print(f"No cached index at {cache}; building LSH from vectors ...")
            idx = LSHIndex(dim=self.corpus_vecs.shape[1])
            idx.add_batch(self.corpus_vecs, list(self.corpus_paths.keys()))
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            idx.save(cache)
            self.index = idx

        self.ranker = Ranker()
        print(f"State ready: {len(self.corpus_paths)} images indexed.")


state = AppState()
