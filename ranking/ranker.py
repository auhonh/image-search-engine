"""Re-rank ANN candidates with exact cosine + auxiliary visual signals."""

from dataclasses import dataclass, field

import numpy as np

from .signals import aspect_similarity, color_similarity


@dataclass
class RankedResult:
    item_id: int
    final_score: float
    cosine_sim: float
    color_sim: float
    aspect_sim: float
    image_path: str = ""

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "final_score": round(self.final_score, 4),
            "cosine_sim": round(self.cosine_sim, 4),
            "color_sim": round(self.color_sim, 4),
            "aspect_sim": round(self.aspect_sim, 4),
            "image_path": self.image_path,
        }


@dataclass
class Ranker:
    """Weighted linear fusion. Weights are normalized to sum to 1 so the
    final score stays comparable as you experiment with weight settings."""

    w_cosine: float = 0.75
    w_color: float = 0.20
    w_aspect: float = 0.05
    _norm: float = field(init=False, default=1.0)

    def __post_init__(self):
        total = self.w_cosine + self.w_color + self.w_aspect
        if total <= 0:
            raise ValueError("At least one weight must be positive")
        self._norm = total

    def rank(
        self,
        query_vec: np.ndarray,
        query_img_path: str,
        candidate_ids: list[int],
        corpus_vecs: np.ndarray,
        corpus_paths: dict[int, str],
        top_k: int = 10,
    ) -> list[RankedResult]:
        query_vec = query_vec.astype(np.float32)
        results = []
        for item_id in candidate_ids:
            cand_path = corpus_paths[item_id]
            cosine = float(corpus_vecs[item_id] @ query_vec)
            color = color_similarity(query_img_path, cand_path)
            aspect = aspect_similarity(query_img_path, cand_path)
            final = (
                self.w_cosine * cosine
                + self.w_color * color
                + self.w_aspect * aspect
            ) / self._norm
            results.append(RankedResult(
                item_id=item_id,
                final_score=final,
                cosine_sim=cosine,
                color_sim=color,
                aspect_sim=aspect,
                image_path=cand_path,
            ))
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:top_k]
