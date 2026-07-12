# Image similarity search engine

A Pinterest-style visual search engine built from scratch: CNN embeddings, a
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
2,000-image corpus (CIFAR-10 subset), 100 held-out queries, k=10:

| Index | Recall@10 | Mean query (ms) | Speedup vs brute force |
|-------|-----------|-----------------|------------------------|
| LSH   | 0.878     | 0.38            | 2.5x                   |
| HNSW  | 1.000     | 0.72            | 1.3x                   |

*(Numbers above are from 2,000 synthetic clustered 2048-dim vectors as a
sanity baseline — regenerate with real image embeddings via the command
above and replace this table.)*

An honest finding worth knowing: at only ~2k vectors, brute force (one
vectorized matmul) is hard to beat from pure-Python index code. The ANN
advantage grows with corpus size — rerun the benchmark at 10k+ images to
see the gap widen.

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
