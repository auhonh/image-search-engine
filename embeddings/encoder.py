"""ResNet-50 image encoder. Strips the classifier head and returns
L2-normalized 2048-dim feature vectors, so dot product == cosine similarity.
"""

import torch
import torch.nn as nn
from torchvision import models


def best_device() -> torch.device:
    """Pick the best available device. On Apple Silicon (M-series) this
    selects MPS, which is significantly faster than CPU for ResNet inference."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ImageEncoder(nn.Module):
    """Wraps torchvision ResNet-50, dropping the final fc layer.

    Output: (batch, 2048) unit-norm float32 vectors.
    """

    DIM = 2048

    def __init__(self, device: torch.device | None = None):
        super().__init__()
        self.device = device or best_device()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Everything up to (and including) avgpool; drop the fc classifier.
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.backbone.eval()
        self.to(self.device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        feat = self.backbone(x)                # (B, 2048, 1, 1)
        feat = feat.flatten(1)                 # (B, 2048)
        return nn.functional.normalize(feat, p=2, dim=1)

    @torch.no_grad()
    def encode_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Encode a preprocessed batch, returning CPU float32 tensors."""
        return self.forward(batch).cpu().float()
