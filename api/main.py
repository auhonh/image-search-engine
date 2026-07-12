"""FastAPI service. Run from the project root:

    uvicorn api.main:app --reload
"""

import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

from api.schemas import HealthResponse, SearchResponse, SearchResult
from api.state import state
from embeddings import embed_image


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.load(index_name="lsh")
    yield


app = FastAPI(title="Image Similarity Search", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        n_indexed=len(state.corpus_paths),
        index_type=type(state.index).__name__,
        device=str(state.encoder.device),
    )


@app.post("/search", response_model=SearchResponse)
async def search(file: UploadFile = File(...), k: int = 10, n_candidates: int = 50):
    # Persist upload to a temp file (signals layer reads from disk via OpenCV)
    suffix = Path(file.filename or "query.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        q_vec = embed_image(tmp_path, state.encoder).numpy()
        candidates = state.index.query(q_vec, k=n_candidates)
        ranked = state.ranker.rank(
            q_vec, tmp_path, candidates,
            state.corpus_vecs, state.corpus_paths, top_k=k,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return SearchResponse(
        query_filename=file.filename or "upload",
        n_candidates=len(candidates),
        results=[SearchResult(**r.to_dict()) for r in ranked],
    )


@app.get("/image/{item_id}")
def image(item_id: int):
    """Serve a corpus image so the UI can display results."""
    path = state.corpus_paths.get(item_id)
    if path is None or not Path(path).exists():
        return {"error": f"unknown item_id {item_id}"}
    return FileResponse(path)
