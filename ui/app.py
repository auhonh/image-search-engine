"""Gradio UI. Run from the project root:

    python -m ui.app

Loads the model + index in-process (no need to run the API separately).
"""

import tempfile

import gradio as gr
from PIL import Image

from api.state import state
from embeddings import embed_image


def search(query_img: Image.Image, k: int, n_candidates: int):
    if query_img is None:
        return [], "Upload an image to search."
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        query_img.convert("RGB").save(tmp.name, "JPEG")
        q_vec = embed_image(tmp.name, state.encoder).numpy()
        candidates = state.index.query(q_vec, k=int(n_candidates))
        ranked = state.ranker.rank(
            q_vec, tmp.name, candidates,
            state.corpus_vecs, state.corpus_paths, top_k=int(k),
        )
    gallery = [
        (r.image_path, f"#{i+1}  score {r.final_score:.3f}  (cos {r.cosine_sim:.3f})")
        for i, r in enumerate(ranked)
    ]
    status = f"{len(candidates)} ANN candidates -> top {len(ranked)} after ranking."
    return gallery, status


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Image similarity search") as demo:
        gr.Markdown("# Image similarity search\nResNet-50 embeddings + custom LSH index + weighted ranking.")
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="Query image")
                k = gr.Slider(1, 30, value=9, step=1, label="Results (k)")
                cands = gr.Slider(10, 200, value=50, step=10, label="ANN candidates")
                btn = gr.Button("Search", variant="primary")
                status = gr.Markdown()
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Results", columns=3, height="auto")
        btn.click(search, inputs=[inp, k, cands], outputs=[gallery, status])
    return demo


if __name__ == "__main__":
    state.load(index_name="lsh")
    build_app().launch()
