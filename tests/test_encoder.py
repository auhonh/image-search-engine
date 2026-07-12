import numpy as np
import pytest
import torch

from embeddings import ImageEncoder


@pytest.fixture(scope="module")
def encoder():
    return ImageEncoder(device=torch.device("cpu"))


def test_output_shape(encoder):
    x = torch.randn(2, 3, 224, 224)
    out = encoder.encode_batch(x)
    assert out.shape == (2, 2048)


def test_unit_norm(encoder):
    x = torch.randn(3, 3, 224, 224)
    out = encoder.encode_batch(x).numpy()
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_deterministic(encoder):
    x = torch.randn(1, 3, 224, 224)
    a = encoder.encode_batch(x.clone()).numpy()
    b = encoder.encode_batch(x.clone()).numpy()
    assert np.allclose(a, b, atol=1e-6)
