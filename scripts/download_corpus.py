"""Download a demo corpus into data/corpus/.

Default: a class-balanced subset of CIFAR-10 (via torchvision, no account
needed), upscaled and saved as JPEGs organized by class folder. Small enough
to embed in minutes on an M-series Mac, varied enough for interesting results.

    python -m scripts.download_corpus --per-class 200
"""

import argparse
from collections import defaultdict
from pathlib import Path

from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"


def main(per_class: int) -> None:
    CORPUS.mkdir(parents=True, exist_ok=True)
    ds = CIFAR10(root=str(ROOT / "data" / "_raw"), train=True, download=True)
    counts: dict[int, int] = defaultdict(int)
    total = 0
    for img, label in ds:
        if counts[label] >= per_class:
            continue
        cls = ds.classes[label]
        out_dir = CORPUS / cls
        out_dir.mkdir(exist_ok=True)
        # Upscale 32x32 -> 224x224 so ResNet preprocessing has real pixels
        img.resize((224, 224)).save(out_dir / f"{cls}_{counts[label]:04d}.jpg", "JPEG", quality=90)
        counts[label] += 1
        total += 1
        if all(c >= per_class for c in counts.values()) and len(counts) == 10:
            break
    print(f"Saved {total} images ({per_class} per class) -> {CORPUS}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--per-class", type=int, default=200)
    main(p.parse_args().per_class)
