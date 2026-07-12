"""Batch-embed a directory of images into data/embeddings/.

Outputs:
  corpus_vecs.npy  -- (N, 2048) float32, row i is item i
  id_map.json      -- {item_id (str): relative image path}
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .encoder import ImageEncoder
from .preprocess import preprocess

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ImageFolderDataset(Dataset):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.paths = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in VALID_EXTS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found under {self.root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return preprocess(img), i


def build_corpus(
    corpus_dir: str | Path,
    out_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 2,
) -> tuple[np.ndarray, dict[int, str]]:
    corpus_dir, out_dir = Path(corpus_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = ImageFolderDataset(corpus_dir)
    # MPS doesn't play well with multiprocess workers in some setups;
    # fall back to 0 workers if issues arise.
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    encoder = ImageEncoder()

    vecs = np.zeros((len(ds), ImageEncoder.DIM), dtype=np.float32)
    print(f"Embedding {len(ds)} images on {encoder.device} ...")
    for batch, idxs in loader:
        out = encoder.encode_batch(batch).numpy()
        vecs[idxs.numpy()] = out
        done = int(idxs.max()) + 1
        if done % (batch_size * 10) < batch_size:
            print(f"  {done}/{len(ds)}")

    id_map = {i: str(p.relative_to(corpus_dir)) for i, p in enumerate(ds.paths)}

    np.save(out_dir / "corpus_vecs.npy", vecs)
    with open(out_dir / "id_map.json", "w") as f:
        json.dump(id_map, f)
    print(f"Saved {vecs.shape} vectors -> {out_dir}")
    return vecs, id_map
