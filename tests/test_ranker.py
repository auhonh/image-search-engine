import numpy as np
import pytest
from PIL import Image

from ranking import Ranker, color_similarity


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    """Three solid-color images: red query, red candidate, blue candidate."""
    d = tmp_path_factory.mktemp("imgs")
    paths = {}
    for name, rgb in [("red_q", (200, 30, 30)), ("red_c", (210, 40, 35)), ("blue_c", (30, 30, 200))]:
        p = d / f"{name}.jpg"
        Image.new("RGB", (224, 224), rgb).save(p)
        paths[name] = str(p)
    return paths


def test_color_similarity_orders_correctly(images):
    same = color_similarity(images["red_q"], images["red_c"])
    diff = color_similarity(images["red_q"], images["blue_c"])
    assert same > diff


def test_weights_normalized():
    r = Ranker(w_cosine=2.0, w_color=1.0, w_aspect=1.0)
    assert r._norm == pytest.approx(4.0)


def test_rank_order_and_scores(images):
    rng = np.random.default_rng(0)
    q = rng.standard_normal(16).astype(np.float32)
    q /= np.linalg.norm(q)

    # Candidate 0: identical embedding + matching color -> should win
    v_close = q.copy()
    v_far = -q
    corpus = np.stack([v_close, v_far])
    paths = {0: images["red_c"], 1: images["blue_c"]}

    ranked = Ranker().rank(q, images["red_q"], [0, 1], corpus, paths, top_k=2)
    assert [r.item_id for r in ranked] == [0, 1]
    assert ranked[0].final_score > ranked[1].final_score
    assert all(-1.0 <= r.final_score <= 1.0 for r in ranked)


def test_invalid_weights():
    with pytest.raises(ValueError):
        Ranker(w_cosine=0, w_color=0, w_aspect=0)
