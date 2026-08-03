"""M4 — Hallucination Rate.

Definition
----------
The fraction of *specific factual claims* in a tutor response that are NOT
supported by the retrieved notes (the RAG grounding context). Unsupported
specifics are hallucinations; generic tutoring prose is not.

Method
------
Each test *case* is an answer-question turn routed through the ``rag`` node, so
the teacher receives retrieved chunks as grounding. We run the compiled tutoring
graph hermetically (own DB + Chroma, corpus = ``data/notes_photosynthesis.txt``)
and capture the final lesson plus the retrieved chunk text.

For each response we extract *specific claims*:

* numeric quantities (``430 nm``, ``9.8``), and
* chemical/scientific formulas (``CO2``, ``H2O``, ``C6H12O6``), and
* proper nouns appearing mid-sentence.

Calculation
-----------
    hallucination_rate(turn)
        = (# specific claims NOT found in any retrieved chunk)
          / (# specific claims)          [0 if no specific claims were made]

    hallucination_rate = mean over turns of hallucination_rate(turn)

We additionally report **grounding coverage** — the fraction of response
sentences whose tokens overlap the retrieved chunks (token Jaccard >= 0.25).
Low coverage means the model wrote generic content instead of quoting the notes;
that is *not* counted as hallucination but is reported so the two can be told
apart. With the deterministic demo provider the system emits template prose and
no fabricated specifics, so the headline rate should be ~0; the same harness
measures real hallucination when a live LLM provider is configured.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import tempfile
from pathlib import Path

from common import (
    DATA_DIR,
    STUDENT_PREFIX,
    base_fig,
    hermetic_container,
    initial_state,
    save_csv,
    sentence_split,
    tokenize,
    write_chart,
)

METRIC_ID = "m4"
METRIC_NAME = "Hallucination Rate"
UNIT = "fraction (0-1)"
GROUNDING_THRESHOLD = 0.25

QUERIES = [
    "What is the chemical equation for photosynthesis?",
    "Which wavelengths of light does chlorophyll absorb most strongly?",
    "Where do the light-dependent reactions of photosynthesis take place?",
    "What does the Calvin cycle produce?",
]


def extract_specific_claims(text: str) -> list[str]:
    """Numeric quantities, scientific formulas and mid-sentence proper nouns."""
    claims: list[str] = []

    # Quantities with optional units.
    claims += re.findall(r"\b\d+(?:\.\d+)?(?:%|nm|kg|m|s|°C|K)?\b", text)

    # Chemical formulas such as CO2, H2O, C6H12O6, ATP.
    claims += re.findall(r"\b(?:[A-Z][a-z]?\d+){2,}\b", text)
    claims += re.findall(r"\b(?:CO2|O2|H2O|ATP|NADPH)\b", text)

    # Proper nouns occurring mid-sentence (not sentence-initial capitals).
    for sentence in sentence_split(text):
        words = re.findall(r"[\w']+", sentence)
        for idx, word in enumerate(words):
            if idx > 0 and word and word[0].isupper():
                claims.append(word)
    return [c for c in claims if c]


def sentence_grounded(sentence: str, chunk_texts: list[str]) -> bool:
    """True if the sentence's token set overlaps a chunk enough."""
    sent_tokens = tokenize(sentence)
    if not sent_tokens:
        return False
    best = 0.0
    for chunk in chunk_texts:
        chunk_tokens = tokenize(chunk)
        if not chunk_tokens:
            continue
        inter = len(sent_tokens & chunk_tokens)
        best = max(best, inter / len(sent_tokens | chunk_tokens))
    return best >= GROUNDING_THRESHOLD


async def _run_async() -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="edueval_hall_"))
    container = None
    try:
        from graphs.workflow import build_tutoring_graph
        from services.rag.ingestion import ingest_document

        container = hermetic_container(tmp)
        ctx = container.context()

        doc_path = DATA_DIR / "notes_photosynthesis.txt"
        assert doc_path.exists(), "Run evaluation/make_sample_docs.py first"
        chunks = ingest_document(doc_path, source=doc_path.name)
        container.rag.ingest_chunks("notes_photosynthesis", chunks)

        graph = build_tutoring_graph(ctx)
        student = f"{STUDENT_PREFIX}_hall"

        rows: list[dict] = []
        for query in QUERIES:
            state = await graph.ainvoke(initial_state(student, query))
            rag_context = state.get("rag_context") or {}
            retrieved = rag_context.get("retrieved_chunks") or []
            chunk_texts = [c.get("content", "") for c in retrieved]
            used = bool(retrieved)

            lesson = state.get("lesson") or {}
            explanation = lesson.get("explanation") or ""

            specific = extract_specific_claims(explanation)
            ground = " ".join(chunk_texts).lower()
            grounded = [c for c in specific if c.lower() in ground]
            fabrication = 0.0 if not specific else round(
                (len(specific) - len(grounded)) / len(specific), 4
            )

            sentences = sentence_split(explanation)
            coverage = round(
                sum(sentence_grounded(s, chunk_texts) for s in sentences) / len(sentences), 4
            ) if sentences else 1.0

            rows.append({
                "query": query,
                "rag_used": used,
                "n_chunks_retrieved": len(retrieved),
                "sources": " | ".join(dict.fromkeys(c.get("source") for c in retrieved)),
                "explanation_len": len(explanation),
                "n_specific_claims": len(specific),
                "specific_claims": "; ".join(specific) or "(none)",
                "n_grounded_claims": len(grounded),
                "hallucination_rate": fabrication,
                "grounding_coverage": coverage,
                "rag_sources_in_state": state.get("rag_sources") or [],
            })

        hallucination_rate = round(
            sum(r["hallucination_rate"] for r in rows) / len(rows), 4
        )
        coverage_mean = round(sum(r["grounding_coverage"] for r in rows) / len(rows), 4)
        csv_path = save_csv(rows, "hallucination")

        fig = base_fig(
            "Hallucination Rate per Turn (specific-claim grounding)",
            "rate (0-1)",
        )
        fig.add_trace(
            dict(type="bar", name="hallucination rate",
                 x=[f"Q{i+1}" for i in range(len(rows))],
                 y=[r["hallucination_rate"] for r in rows],
                 marker_color="#4C78A8",
                 text=[f"{r['hallucination_rate']:.2f}" for r in rows],
                 textposition="outside")
        )
        fig.add_trace(
            dict(type="bar", name="grounding coverage",
                 x=[f"Q{i+1}" for i in range(len(rows))],
                 y=[r["grounding_coverage"] for r in rows],
                 marker_color="#E45756", opacity=0.8)
        )
        fig.update_layout(barmode="group", yaxis=dict(title="rate (0-1)", range=[0, 1]))
        graph_path = write_chart(fig, "hallucination_rate")

        return {
            "metric_id": METRIC_ID,
            "metric_name": METRIC_NAME,
            "value": hallucination_rate,
            "unit": UNIT,
            "detail": (
                f"mean of {len(rows)} RAG-grounded turns; "
                f"grounding coverage {coverage_mean:.2f} (generic demo prose, see README)"
            ),
            "csv": str(csv_path),
            "graph": str(graph_path),
            "num_cases": len(rows),
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
