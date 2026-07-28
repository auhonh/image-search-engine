"""Download a demo corpus into data/corpus/.

Default: a class-balanced subset of STL-10 (via torchvision, no account
needed), saved as JPEGs organized by class folder. STL-10 images are natively
96x96 (vs. CIFAR-10's 32x32) -- 9x the source pixels, so results actually
look sharp instead of blocky. Pulls from the train+test splits combined
(up to 1,300 images/class) since STL-10's train split alone only has 500/class.

    python -m scripts.download_corpus --per-class 200
"""

import argparse
from collections import defaultdict
from itertools import chain
from pathlib import Path

from PIL import Image
from torchvision.datasets import STL10

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"


def main(per_class: int) -> None:
    CORPUS.mkdir(parents=True, exist_ok=True)
    raw_root = str(ROOT / "data" / "_raw")
    splits = chain(
        STL10(root=raw_root, split="train", download=True),
        STL10(root=raw_root, split="test", download=True),
    )
    counts: dict[int, int] = defaultdict(int)
    total = 0
    classes = STL10(root=raw_root, split="train").classes
    for img, label in splits:
        if counts[label] >= per_class:
            continue
        cls = classes[label]
        out_dir = CORPUS / cls
        out_dir.mkdir(exist_ok=True)
        # Mild upscale 96x96 -> 224x224 with a high-quality filter for display
        # consistency; ResNet preprocessing would resize regardless, this is
        # just so gallery thumbnails aren't tiny.
        img.resize((224, 224), Image.Resampling.LANCZOS).save(
            out_dir / f"{cls}_{counts[label]:04d}.jpg", "JPEG", quality=95
        )
        counts[label] += 1
        total += 1
        if all(c >= per_class for c in counts.values()) and len(counts) == len(classes):
            break
    print(f"Saved {total} images ({per_class} per class) -> {CORPUS}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--per-class", type=int, default=200)
    main(p.parse_args().per_class)
