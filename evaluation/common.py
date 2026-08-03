"""Shared utilities for the EduAgent evaluation framework.

This module is *pure tooling* — it only reads public API responses and, for the
hermetic benchmarks (retrieval, hallucination, agent timing), drives the already-
compiled LangGraph with a throw-away container. It never modifies application
code; the application is only ever imported, never edited.

Two kinds of benchmarks exist here:

* **HTTP benchmarks** talk to a running EduAgent server (start with
  ``python main.py``) and measure the full stack including serialization.
* **Hermetic benchmarks** build their own :class:`Container` against a temporary
  SQLite database and a temporary ChromaDB directory, so retrieval and timing
  metrics are measured against a controlled corpus with zero interference from
  the live server or any other student's documents.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import httpx
import pandas as pd
import plotly.graph_objects as go

# Make the application packages importable when a benchmark runs from inside
# the evaluation/ directory (the hermetic benchmarks import services.*,
# config.* and graphs.* to drive the compiled LangGraph directly).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
BASE_URL = os.environ.get("EDUAGENT_BASE_URL", "http://localhost:8000")
HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CSV_DIR = RESULTS_DIR / "csv"
GRAPH_DIR = RESULTS_DIR / "graphs"
DATA_DIR = HERE / "data"

DEFAULT_RUNS = int(os.environ.get("EVAL_RUNS", "5"))
STUDENT_PREFIX = f"eval_{os.getpid()}"

# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
_STOPWORDS = set(
    """
    a an the and or but if then else for to of in on at by with from into over
    under about teach me give practice quiz on what is explain why how are were
    was be been being do does did it its this that these those my your our their
    i want would like step by please help understand about more of for the
    """.split()
)


def keywords(text: str) -> set[str]:
    """Lower-cased alpha-numeric keyword set with stopwords removed."""
    return {
        w
        for w in re.findall(r"[a-z0-9']{2,}", str(text).lower())
        if w not in _STOPWORDS
    }


def sentence_split(text: str) -> list[str]:
    """Very light sentence splitter ('.', '!', '?' boundaries)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", str(text).lower())}


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile (numpy-style) over a sorted copy."""
    values = sorted(float(v) for v in values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    frac = k - lo
    return values[lo] + (values[hi] - values[lo]) * frac


def latency_stats(times: list[float]) -> dict[str, float]:
    """Summary statistics for a list of wall-clock times (seconds)."""
    if not times:
        return {"mean": 0.0, "median": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0,
                "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "p50": percentile(times, 50),
        "p90": percentile(times, 90),
        "p95": percentile(times, 95),
        "min": min(times),
        "max": max(times),
        "n": len(times),
    }


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def ensure_server() -> dict:
    """Raise if the EduAgent server is not reachable at ``BASE_URL``."""
    with httpx.Client(base_url=BASE_URL, timeout=5.0) as client:
        r = client.get("/health")
        r.raise_for_status()
        return r.json()


def timed_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Call ``fn`` and return ``(result, elapsed_seconds)``."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def save_csv(rows: list[dict], name: str) -> Path:
    """Write a list of dict rows to ``results/csv/<name>.csv``."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    path = CSV_DIR / f"{name}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def save_json(obj: Any, name: str) -> Path:
    """Write JSON to ``results/<name>.json``."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    Path(path).write_text(json.dumps(obj, indent=2, encoding="utf-8"), encoding="utf-8")
    return path


def write_chart(fig: go.Figure, name: str) -> Path:
    """Write a plotly figure to ``results/graphs/<name>.html``."""
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    path = GRAPH_DIR / f"{name}.html"
    fig.write_html(str(path))
    return path


def base_fig(title: str, ylabel: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        yaxis_title=ylabel,
        template="plotly_white",
        height=460,
        margin=dict(l=80, r=40, t=60, b=60),
    )
    return fig


def bar_chart(
    title: str,
    x: list[str],
    y: list[float],
    ylabel: str,
    text: list[str] | None = None,
    color: str = "#4C78A8",
) -> go.Figure:
    """Simple labelled bar chart used across the benchmarks."""
    fig = base_fig(title, ylabel)
    fig.add_trace(go.Bar(x=x, y=y, text=text, textposition="outside",
                         marker_color=color, name=title))
    fig.update_yaxes(rangemode="tozero")
    return fig


# --------------------------------------------------------------------------- #
# Hermetic container (used by retrieval / hallucination / agent-timing metrics)
# --------------------------------------------------------------------------- #
def hermetic_container(tmp_base: Path):
    """Build an isolated :class:`Container` on a throw-away DB + Chroma dir.

    Lazy imports keep the HTTP-only benchmarks from pulling heavy dependencies
    (sentence-transformers / chromadb) when they don't need them.
    """
    from config.settings import DatabaseSettings, LLMSettings, Settings, VectorStoreSettings
    from services.container import Container

    settings = Settings(
        llm=LLMSettings(
            provider="demo",
            openai_api_key="",
            gemini_api_key="",
            openai_model="gpt-4o-mini",
            gemini_model="gemini-1.5-flash",
        ),
        database=DatabaseSettings(
            url=f"sqlite:///{(tmp_base / 'eval.db').as_posix()}",
            create_on_startup=True,
            echo=False,
        ),
        vector_store=VectorStoreSettings(
            persist_directory=str(tmp_base / "chroma"),
            embedding_model="all-MiniLM-L6-v2",
        ),
    )
    return Container(settings)


def initial_state(
    student_id: str, message: str, with_quiz: bool = False, topic: str | None = None
) -> dict[str, Any]:
    """Reproduce the exact LangGraph input the API builds in ``api/routers/chat.py``."""
    return {
        "student_id": student_id,
        "user_input": message,
        "topic": topic or None,
        "with_quiz": with_quiz,
        "intent": "general",
        "messages": [],
        "errors": [],
        "memory_updates": [],
        "lesson": None,
        "quiz": None,
        "evaluation": None,
        "recommendations": None,
        "rag_context": None,
        "rag_sources": [],
    }
