"""RAG configuration experiments - embedding models x chunking strategies.

What is measured
----------------
For every (embedding model x chunking strategy) combination we build an isolated
ChromaDB store and evaluate:

* **Retrieval quality** — hit@1, precision@4, recall@4, MRR over a ground-truth
  2-topic corpus (the same queries/method as the evaluation suite's M3 metric).
* **Embedding latency** — wall-clock to embed the whole corpus with that model
  (a throughput proxy for ingest cost).
* **Query latency** — mean wall-clock of one retrieval (query embedding + vector
  search), the user-visible RAG latency attributable to the configuration.

The application is never modified: we instantiate the real ``VectorStore`` /
``RAGService`` classes and inject the embedding provider under test, exactly as
the production container does.

Run::

    python experiments/run_experiments.py

Outputs land in ``experiments/results/`` (CSV, plotly HTML charts, summary).
"""

from __future__ import annotations

import gc
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Make the application importable when run from inside experiments/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402  (imports must follow the sys.path setup above)
import plotly.graph_objects as go  # noqa: E402
from chunkers import CHUNKERS  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CSV_DIR = RESULTS_DIR / "csv"
GRAPH_DIR = RESULTS_DIR / "graphs"
DATA_DIR = HERE / "data"
TOP_K = 4

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
# (model_id, short_label). First is the application baseline.
EMBEDDING_MODELS = [
    ("all-MiniLM-L6-v2",           "MiniLM-L6-v2\n(baseline)"),
    ("multi-qa-MiniLM-L6-cos-v1",  "multi-qa-MiniLM\n-cos-v1"),
    ("paraphrase-MiniLM-L6-v2",    "paraphrase-MiniLM\n-L6-v2"),
    ("all-mpnet-base-v2",          "all-mpnet\n-base-v2"),
    ("BAAI/bge-small-en-v1.5",     "bge-small\n-en-v1.5"),
]

# (query, expected source filename or None for the negative control).
QUERIES = [
    # Photosynthesis (notes_photosynthesis.txt)
    ("What is the chemical equation for photosynthesis?", "notes_photosynthesis.txt"),
    ("Which wavelengths of light does chlorophyll absorb most strongly?",
     "notes_photosynthesis.txt"),
    ("Where do the light-dependent reactions of photosynthesis take place?",
     "notes_photosynthesis.txt"),
    # Newton's laws (notes_newton.txt)
    ("State Newton's second law of motion.", "notes_newton.txt"),
    ("How are force, mass and acceleration related?", "notes_newton.txt"),
    ("What is the value of g on Earth for the weight formula?", "notes_newton.txt"),
    # Cells (notes_cells.txt)
    ("What is the function of mitochondria in a cell?", "notes_cells.txt"),
    ("How do plant cells differ from animal cells?", "notes_cells.txt"),
    ("What is the difference between mitosis and meiosis?", "notes_cells.txt"),
    # Electricity (notes_electricity.txt)
    ("What does Ohm's law state about voltage and current?", "notes_electricity.txt"),
    ("How do series and parallel circuits differ?", "notes_electricity.txt"),
    ("What is the role of a fuse in a circuit?", "notes_electricity.txt"),
    # World War One (notes_ww1.txt)
    ("When did World War One begin and how?", "notes_ww1.txt"),
    ("What were the main alliances in the First World War?", "notes_ww1.txt"),
    ("How did World War One come to an end?", "notes_ww1.txt"),
    # Negative control — no relevant document exists.
    ("Who wrote the novel Moby Dick?", None),
]


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _build_rag(model_name: str, persist_dir: Path):
    """Return a RAGService over a fresh Chroma store using ``model_name``."""
    from config.settings import VectorStoreSettings
    from services.rag.embeddings import SentenceTransformerProvider
    from services.rag.service import RAGService
    from services.rag.vector_store import VectorStore

    settings = VectorStoreSettings(
        persist_directory=str(persist_dir),
        embedding_model=model_name,
    )
    provider = SentenceTransformerProvider(model_name)
    store = VectorStore(settings, embedder=provider)
    return RAGService(settings, vector_store=store)


def _load_corpus() -> list[tuple[str, list[dict]]]:
    """Return [(doc_id, [{text, page, source}, ...]), ...] using app extraction."""
    from services.rag.ingestion import extract_pages

    docs = []
    for path in sorted(DATA_DIR.glob("notes_*.txt")):
        pages = extract_pages(path)
        docs.append((path.stem, path.name, pages))
    return docs


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_retrieval(rag, query: str, expected: str | None) -> dict:
    """Run one query against the store and score it (M3 methodology)."""
    context = rag.build_context(query, top_k=TOP_K)
    chunks = context.retrieved_chunks
    sources = [c.source for c in chunks]

    hit_at_1 = 1.0 if (expected and sources and sources[0] == expected) else 0.0
    rank = (sources.index(expected) + 1) if expected in sources else None
    mrr = 1.0 / rank if rank else 0.0
    relevant_retrieved = 1 if (expected and expected in sources) else 0
    precision = (relevant_retrieved / len(sources)) if sources else 0.0
    recall = relevant_retrieved if expected else 0.0
    return {
        "hit_at_1": hit_at_1,
        "precision_at_4": precision,
        "recall_at_4": recall,
        "mrr": mrr,
        "negative_expected": expected is None,
    }


def run_combo(model_id: str, chunker_id: str, chunk_fn, tmp_base: Path) -> dict:
    """Run one (model, chunker) combination; returns aggregated result row."""
    persist_dir = tmp_base / f"{model_id.replace('/', '_')}__{chunker_id}"
    rag = _build_rag(model_id, persist_dir)

    # Corpus: chunk (timing ingest/chunking) then embed+store (embedding time).
    corpus_docs = _load_corpus()
    chunk_times: list[float] = []
    embed_times: list[float] = []
    n_chunks_total = 0
    for doc_id, _, pages in corpus_docs:
        for page_number, text in pages:
            t0 = time.perf_counter()
            pieces = chunk_fn(text)
            chunk_times.append(time.perf_counter() - t0)
            n_chunks_total += len(pieces)
            chunk_dicts = [
                {"text": p, "page": page_number, "source": f"{doc_id}.txt"}
                for p in pieces
            ]
            t1 = time.perf_counter()
            rag.ingest_chunks(doc_id, chunk_dicts)
            embed_times.append(time.perf_counter() - t1)

    # Retrieval evaluation (timing each search).
    query_times: list[float] = []
    agg = {"hit_at_1": [], "precision_at_4": [], "recall_at_4": [], "mrr": []}
    negative_results: list[bool] = []
    for query, expected in QUERIES:
        t0 = time.perf_counter()
        result = evaluate_retrieval(rag, query, expected)
        query_times.append(time.perf_counter() - t0)
        if result["negative_expected"]:
            negative_results.append(result["hit_at_1"])
        else:
            for key, values in agg.items():
                values.append(result[key])

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    row = {
        "embedding_model": model_id,
        "chunking": chunker_id,
        "hit_at_1": round(mean(agg["hit_at_1"]), 4),
        "precision_at_4": round(mean(agg["precision_at_4"]), 4),
        "recall_at_4": round(mean(agg["recall_at_4"]), 4),
        "mrr": round(mean(agg["mrr"]), 4),
        "composite_quality": round(mean([
            mean(agg["hit_at_1"]), mean(agg["precision_at_4"]), mean(agg["mrr"])
        ]), 4),
        "negative_control_miss": round(1 - mean(negative_results), 4),
        "n_chunks_total": int(n_chunks_total),
        "embed_s": round(sum(embed_times), 4),
        "chunk_s": round(sum(chunk_times), 4),
        "query_s_mean": round(mean(query_times), 4),
    }
    # Release the store + Chroma handle before the next combo.
    del rag
    gc.collect()
    return row


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _heatmap(df: pd.DataFrame, value_col: str, title: str, zmin: float, zmax: float):
    pivot = df.pivot(index="embedding_model", columns="chunking", values=value_col)
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        zmin=zmin, zmax=zmax,
        colorscale="Blues",
        text=[[f"{v:.3f}" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        hovertemplate="model=%{y}<br>chunking=%{x}<br>value=%{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=title, yaxis_title="Embedding model", xaxis_title="Chunking strategy",
        template="plotly_white", height=480, margin={"l": 100, "r": 40, "t": 70, "b": 60},
    )
    return fig


def _grouped_bars(df: pd.DataFrame, title: str):
    fig = go.Figure()
    colors = {"hit_at_1": "#4C78A8", "precision_at_4": "#F58518",
              "mrr": "#54A24B"}
    for metric_key in ["hit_at_1", "precision_at_4", "mrr"]:
        fig.add_trace(go.Bar(
            name=metric_key,
            x=[f"{r.embedding_model} · {r.chunking}" for r in df.itertuples()],
            y=[getattr(r, metric_key) for r in df.itertuples()],
            marker_color=colors[metric_key],
        ))
    fig.update_layout(
        title=title, template="plotly_white", height=460,
        barmode="group", margin={"l": 60, "r": 40, "t": 60, "b": 160},
        xaxis_tickangle=-90, yaxis={"range": [0, 1], "title": "score (0-1)"},
    )
    return fig


def _latency_bars(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="embed corpus (s)",
        x=[f"{r.embedding_model} · {r.chunking}" for r in df.itertuples()],
        y=[r.embed_s for r in df.itertuples()], marker_color="#4C78A8",
    ))
    fig.add_trace(go.Bar(
        name="mean query (s)",
        x=[f"{r.embedding_model} · {r.chunking}" for r in df.itertuples()],
        y=[r.query_s_mean for r in df.itertuples()], marker_color="#E45756",
    ))
    fig.update_layout(
        title="Embedding & query latency by configuration", template="plotly_white",
        height=460, barmode="group", margin={"l": 60, "r": 40, "t": 60, "b": 160},
        xaxis_tickangle=-90, yaxis={"title": "seconds"},
    )
    return fig


def _tradeoff_scatter(df: pd.DataFrame):
    fig = go.Figure()
    for model_id in df["embedding_model"].unique():
        sub = df[df["embedding_model"] == model_id]
        fig.add_trace(go.Scatter(
            name=model_id,
            x=sub["query_s_mean"], y=sub["composite_quality"], mode="markers",
            text=[f"{r.chunking}" for r in sub.itertuples()],
            customdata=[f"{r.chunking}" for r in sub.itertuples()],
            hovertemplate="%{text}<br>query=%{x:.4f}s<br>quality=%{y:.3f}<extra></extra>",
            marker={"size": 12},
        ))
    fig.update_layout(
        title="Retrieval quality vs query latency (each point = one chunking strategy)",
        template="plotly_white", height=460,
        xaxis={"title": "mean query latency (s)"},
        yaxis={"title": "composite quality (0-1)", "range": [0, 1]},
        margin={"l": 60, "r": 40, "t": 70, "b": 60},
    )
    return fig


def _write_chart(fig, name: str) -> Path:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    path = GRAPH_DIR / f"{name}.html"
    fig.write_html(str(path))
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not list(DATA_DIR.glob("notes_*.txt")):
        import runpy
        runpy.run_path(str(HERE / "make_corpus.py"), run_name="__main__")
        print(f"Corpus ready in {DATA_DIR}")

    # Warm up all models first so downloads never pollute timing, and a
    # failed download marks the model as unavailable rather than crashing.
    print("Warming up embedding models ...")
    from services.rag.embeddings import SentenceTransformerProvider
    available: list[tuple[str, str]] = []
    for model_id, label in EMBEDDING_MODELS:
        try:
            provider = SentenceTransformerProvider(model_id)
            provider.embed_texts(["warmup"])
            available.append((model_id, label))
            print(f"  ok   {model_id}")
        except Exception as exc:
            print(f"  SKIP {model_id} ({exc})")
        del provider
        gc.collect()

    print(f"\nRunning {len(available)} models x {len(CHUNKERS)} chunkers ...")
    rows: list[dict] = []
    tmp_root = Path(tempfile.mkdtemp(prefix="eduexp_"))
    try:
        for model_id, _label in available:
            for chunker_id, _desc, chunk_fn in CHUNKERS:
                print(f"  {model_id} / {chunker_id}")
                try:
                    row = run_combo(model_id, chunker_id, chunk_fn, tmp_root)
                except Exception as exc:
                    print(f"    ERROR: {exc}")
                    continue
                rows.append(row)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    df = pd.DataFrame(rows)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_DIR / "experiment_results.csv", index=False)

    # ---- Charts ----
    paths = {
        "retrieval_heatmap": _write_chart(
            _heatmap(df, "composite_quality",
                     "Retrieval quality (hit@1·precision@4·MRR) by configuration",
                     0.0, 1.0), "retrieval_heatmap"),
        "retrieval_bars": _write_chart(
            _grouped_bars(df, "Retrieval metrics by configuration"), "retrieval_bars"),
        "latency_comparison": _write_chart(
            _latency_bars(df), "latency_comparison"),
        "quality_vs_latency": _write_chart(
            _tradeoff_scatter(df), "quality_vs_latency"),
    }

    # ---- Summary ----
    best_by_quality = df.sort_values("composite_quality", ascending=False).head(1)
    best_by_latency = df.sort_values("query_s_mean").head(1)
    summary = {
        "configs_tested": f"{len(available)} embedding models x {len(CHUNKERS)} chunking = {len(df)} combos",
        "best_overall_quality": best_by_quality.iloc[0].to_dict(),
        "fastest_query": best_by_latency.iloc[0].to_dict(),
        "charts": {k: str(v) for k, v in paths.items()},
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n=== Best retrieval quality ===")
    print(best_by_quality.iloc[0].to_string())
    print("\n=== Fastest query ===")
    print(best_by_latency.iloc[0].to_string())
    print(f"\nResults: {CSV_DIR / 'experiment_results.csv'}")
    for name, path in paths.items():
        print(f"  chart {name}: {path}")


if __name__ == "__main__":
    main()
