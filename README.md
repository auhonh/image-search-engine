# Image similarity search engine

A visual search engine built from scratch: CNN embeddings, a
custom approximate-nearest-neighbor index (LSH and HNSW, no FAISS), and a
weighted ranking layer — served through FastAPI with a Gradio demo UI.

## Architecture

```
query image
    │
    ▼
[ Layer 1: embedding ]   ResNet-50 (classifier head removed) → L2-normalized
    │                    2048-dim vector, so dot product == cosine similarity
    ▼
[ Layer 2: ANN index ]   Custom LSH (random hyperplane projections, multi-table)
    │                    and HNSW (hierarchical navigable small world graph)
    ▼                    → ~50 candidates in sublinear time
[ Layer 3: ranking ]     Exact cosine re-score + color histogram intersection
    │                    + aspect-ratio signal, fused with normalized weights
    ▼
top-K ranked results
```

## Benchmark results

Run `python -m scripts.run_benchmark` to regenerate. Numbers below are from a
real 2,000-image corpus (STL-10 subset, ResNet-50 embeddings), 100 held-out
queries, k=10:

| Index | Recall@10 | Mean query (ms) | Speedup vs brute force |
|-------|-----------|-----------------|------------------------|
| LSH   | 0.467     | 0.10            | 1.81x                  |
| HNSW  | 1.000     | 0.39            | 0.47x                  |

An honest finding worth knowing: at only ~2k vectors, brute force (one
vectorized matmul) is hard to beat from pure-Python index code — HNSW is
actually slower than brute force here despite perfect recall, and LSH's
default `n_bits`/`n_tables` trade noticeably more recall than the earlier
synthetic-vector benchmark suggested, since real ResNet embeddings cluster
differently than the synthetic ones. The ANN advantage grows with corpus
size — rerun the benchmark at 10k+ images to see the gap widen, and tune
LSH's `n_tables`/`n_bits` if you need higher recall at this corpus size.

Key tradeoff explored: LSH `n_tables` controls recall, `n_bits` controls
bucket precision. HNSW `ef_search` trades latency for recall smoothly.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m scripts.download_corpus --per-class 200   # 2,000 demo images
python -m scripts.build_index                        # embed + build LSH & HNSW
python -m scripts.run_benchmark                      # recall/latency table
python -m ui.app                                     # Gradio demo at :7860
```

Or run the API instead of the UI:

```bash
uvicorn api.main:app --reload
# POST an image to http://127.0.0.1:8000/search  (docs at /docs)
```

On Apple Silicon, the encoder automatically uses the MPS backend.

### Using your own photos

`data/corpus/` just needs to contain image files somewhere under it —
`build_corpus` recursively globs for `.jpg/.jpeg/.png/.webp/.bmp` and doesn't
care about folder names or nesting; the per-class subfolders from
`download_corpus.py` are purely organizational.

```bash
# copy your own photos in (flat, or in whatever folders you like)
cp ~/Pictures/vacation/*.jpg data/corpus/my_photos/

python -m scripts.build_index      # re-embeds everything under data/corpus/ + rebuilds LSH/HNSW
python -m ui.app                   # or uvicorn api.main:app --reload
```

To mix in the STL-10 demo set alongside your own photos, just leave the
downloaded class folders in place — `build_index` embeds the whole tree.
To replace the demo set entirely, delete `data/corpus/{airplane,bird,car,...}`
before adding your photos, and delete `data/embeddings/` and
`data/index_cache/` so stale vectors/indexes aren't reused (`build_index`
regenerates both).

## Project layout

- `embeddings/` — ResNet-50 encoder, preprocessing, batch corpus embedding
- `index/` — `Index` abstract base, `LSHIndex`, `NSWIndex`, `HNSWIndex`,
  brute-force baseline, recall/latency benchmark harness
- `ranking/` — auxiliary visual signals and weighted score fusion
- `api/` — FastAPI service (singleton state loads model/index once at startup)
- `ui/` — Gradio demo
- `scripts/` — corpus download, index build, benchmark CLI
- `tests/` — unit tests (`pytest`)

## Tests

```bash
pytest tests/ -v
```

## Design notes / honest limitations

- The LSH candidate set can be empty for out-of-distribution queries when
  `n_bits` is high; production systems probe neighboring buckets (multi-probe
  LSH) — a natural extension.
- HNSW insertion here prunes neighbors by raw similarity; the paper's
  heuristic pruning (keeping diverse neighbors) improves graph quality.
- Ranking weights are hand-tuned; learning them from click data is the
  obvious next step (learning-to-rank).
