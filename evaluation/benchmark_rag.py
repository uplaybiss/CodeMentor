"""M3 — Retrieval Accuracy (RAG).

Definition
----------
How well the vector store retrieves the correct source document for a query.
This measures the *retrieval subsystem* (ChromaDB + all-MiniLM-L6-v2 embeddings
+ chunking) that the RAG agent uses to ground answers.

Why a hermetic run?
-------------------
The production vector store is shared across all students and already contains
documents from live usage, so recall/precision computed there would be
contaminated by unrelated chunks. This benchmark therefore builds its own
temporary ChromaDB, ingests a small ground-truth corpus (``data/notes_*.txt``)
and measures retrieval against it. The exact same ``RAGService`` / chunking code
used by the API is exercised — only the store is isolated.

Metrics (per query, top-k = 4)
------------------------------
* ``hit_at_1``   — 1 if the expected source is the top-1 hit
* ``precision_at_4`` — |retrieved ∩ relevant| / |retrieved|
* ``recall_at_4`` — |retrieved ∩ relevant| / |relevant|
* ``mrr``        — 1 / rank of the expected source (0 if absent)

One query is a *negative control* (no relevant document exists); it is reported
separately so a correct "no relevant hit" is not punished.

Calculation
-----------
    retrieval_accuracy = mean over relevant queries of (precision_at_4)
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

from common import (
    DATA_DIR,
    STUDENT_PREFIX,
    base_fig,
    hermetic_container,
    save_csv,
    write_chart,
)

METRIC_ID = "m3"
METRIC_NAME = "Retrieval Accuracy (RAG)"
UNIT = "fraction (0-1)"
TOP_K = 4

# (query, expected source filename or None for the negative control)
QUERIES = [
    ("What is the chemical equation for photosynthesis?", "notes_photosynthesis.txt"),
    ("Which wavelengths of light does chlorophyll absorb most strongly?",
     "notes_photosynthesis.txt"),
    ("Where do the light-dependent reactions of photosynthesis take place?",
     "notes_photosynthesis.txt"),
    ("State Newton's second law of motion.", "notes_newton.txt"),
    ("How are force, mass and acceleration related?", "notes_newton.txt"),
    ("What is the value of g on Earth for the weight formula?",
     "notes_newton.txt"),
    ("Who wrote the novel Moby Dick?", None),  # negative control
]


async def _run_async() -> dict:
    from services.rag.ingestion import ingest_document

    tmp = Path(tempfile.mkdtemp(prefix="edueval_rag_"))
    container = None
    try:
        container = hermetic_container(tmp)
        ctx = container.context()

        # Build the controlled corpus.
        corpus = sorted(DATA_DIR.glob("notes_*.txt"))
        assert corpus, f"Run evaluation/make_sample_docs.py first (no corpus in {DATA_DIR})"
        for doc_path in corpus:
            chunks = ingest_document(doc_path, source=doc_path.name)
            container.rag.ingest_chunks(doc_path.stem, chunks)

        rows: list[dict] = []
        for query, expected in QUERIES:
            context = ctx.rag.build_context(query, top_k=TOP_K)
            retrieved = context.retrieved_chunks
            sources = [c.source for c in retrieved]
            scores = [c.score for c in retrieved]

            hit_at_1 = 1.0 if (expected and sources and sources[0] == expected) else 0.0
            rank = (sources.index(expected) + 1) if expected in sources else None
            mrr = 1.0 / rank if rank else 0.0
            relevant_retrieved = 1 if (expected and expected in sources) else 0
            precision = (relevant_retrieved / len(sources)) if sources else 0.0
            recall = relevant_retrieved if expected else 0.0  # |relevant| == 1

            rows.append({
                "query": query,
                "expected_source": expected or "(negative control)",
                "retrieved_sources": " | ".join(sources) or "(none)",
                "top1_score": round(scores[0], 4) if scores else None,
                "n_retrieved": len(sources),
                "hit_at_1": hit_at_1,
                "precision_at_4": round(precision, 4),
                "recall_at_4": recall,
                "mrr": round(mrr, 4),
            })

        relevant = [r for r in rows if r["expected_source"] != "(negative control)"]
        negative = [r for r in rows if r["expected_source"] == "(negative control)"]

        summary = {
            "metric_id": METRIC_ID,
            "metric_name": METRIC_NAME,
            "hit_at_1": round(sum(r["hit_at_1"] for r in relevant) / len(relevant), 4),
            "precision_at_4": round(sum(r["precision_at_4"] for r in relevant) / len(relevant), 4),
            "recall_at_4": round(sum(r["recall_at_4"] for r in relevant) / len(relevant), 4),
            "mrr": round(sum(r["mrr"] for r in relevant) / len(relevant), 4),
            "negative_control_top1_miss": (1 - negative[0]["hit_at_1"]) if negative else None,
        }
        csv_path = save_csv(rows, "retrieval_accuracy")

        fig = base_fig(
            "Retrieval Accuracy (RAG) — top-4 search over 2-doc corpus",
            "score (0-1)",
        )
        fig.add_trace(
            dict(
                type="bar",
                name="mean over relevant queries",
                x=["hit@1", "precision@4", "recall@4", "MRR"],
                y=[summary["hit_at_1"], summary["precision_at_4"],
                   summary["recall_at_4"], summary["mrr"]],
                marker_color="#4C78A8",
                text=[f"{v:.2f}" for v in
                      [summary["hit_at_1"], summary["precision_at_4"],
                       summary["recall_at_4"], summary["mrr"]]],
                textposition="outside",
            )
        )
        fig.update_yaxes(range=[0, 1])
        graph_path = write_chart(fig, "retrieval_accuracy")

        return {
            "metric_id": METRIC_ID,
            "metric_name": METRIC_NAME,
            "value": summary["precision_at_4"],
            "unit": UNIT,
            "detail": (
                f"hit@1={summary['hit_at_1']:.2f} precision@4={summary['precision_at_4']:.2f} "
                f"recall@4={summary['recall_at_4']:.2f} MRR={summary['mrr']:.2f} "
                f"({len(relevant)} relevant + {len(negative)} negative-control queries)"
            ),
            "csv": str(csv_path),
            "graph": str(graph_path),
            "num_cases": len(relevant),
            "corpus": [p.name for p in corpus],
        }
    finally:
        if container is not None:
            container.dispose()
        shutil.rmtree(tmp, ignore_errors=True)


def run() -> dict:
    return asyncio.run(_run_async())


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
