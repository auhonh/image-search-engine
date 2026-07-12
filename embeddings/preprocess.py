"""Image preprocessing shared by indexing and query paths.
ImageNet normalization constants must match ResNet-50's pretraining."""

import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_image(path: str) -> torch.Tensor:
    """Load a single image file -> preprocessed (1, 3, 224, 224) tensor."""
    img = Image.open(path).convert("RGB")
    return preprocess(img).unsqueeze(0)


def embed_image(path: str, encoder) -> torch.Tensor:
    """Convenience: path -> unit-norm (2048,) CPU tensor."""
    return encoder.encode_batch(load_image(path)).squeeze(0)
