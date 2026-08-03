"""M7 — Agent Execution Time.

Definition
----------
Wall-clock execution time of each LangGraph node (planner, rag, teacher, quiz,
evaluation, memory, recommendation) and of a full tutoring turn. This measures
the multi-agent orchestration cost — the part of latency that is the LangGraph
workflow itself.

Method
------
A hermetic :class:`Container` (own DB + Chroma, corpus = ``data/notes_*.txt``)
runs the *same compiled graphs* the API uses. For each turn we iterate
``graph.astream(..., stream_mode="updates")``; LangGraph emits one ``{node:
update}`` per node as it completes, and the graph executes sequentially, so the
wall-clock between successive emissions approximates each node's execution time.

Turn types exercised
--------------------
* ``teach``   — planner → teacher → memory → recommendation
* ``quiz``    — planner → teacher → quiz → memory → recommendation
* ``answer``  — planner → rag → teacher → memory → recommendation
* ``submit``  — evaluation → memory → recommendation  (quiz-submission graph)

Calculation
-----------
    agent_execution_time = mean full-tutoring-graph time (teach turns)
    per-node stats       = {mean, p50, p90, std} of each node's execution time
"""

from __future__ import annotations

import asyncio
import shutil
import statistics
import tempfile
import time
from pathlib import Path

from common import (
    DATA_DIR,
    DEFAULT_RUNS,
    STUDENT_PREFIX,
    base_fig,
    hermetic_container,
    initial_state,
    latency_stats,
    save_csv,
    write_chart,
)

METRIC_ID = "m7"
METRIC_NAME = "Agent Execution Time"
UNIT = "seconds"

TURN_TYPES = ["teach", "quiz", "answer", "submit"]


async def time_turn(graph, state: dict) -> tuple[dict[str, float], float]:
    """Return (per-node durations, total wall-clock) for one graph invocation."""
    node_times: dict[str, float] = {}
    prev = time.perf_counter()
    async for update in graph.astream(state, stream_mode="updates"):
        now = time.perf_counter()
        for name in update:
            if name.startswith("__"):
                continue
            node_times[name] = now - prev
        prev = now
    return node_times, sum(node_times.values())


async def _run_async() -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="edueval_agents_"))
    container = None
    try:
        from graphs.workflow import build_evaluation_graph, build_tutoring_graph
        from services.rag.ingestion import ingest_document

        container = hermetic_container(tmp)
        ctx = container.context()

        for doc_path in sorted(DATA_DIR.glob("notes_*.txt")):
            chunks = ingest_document(doc_path, source=doc_path.name)
            container.rag.ingest_chunks(doc_path.stem, chunks)

        tutoring = build_tutoring_graph(ctx)
        evaluation = build_evaluation_graph(ctx)
        student = f"{STUDENT_PREFIX}_agent"
        runs = DEFAULT_RUNS

        # Seed one quiz + answers for the submit turn type.
        seeded = await tutoring.ainvoke(
            initial_state(student, "Give me a practice quiz on fractions", with_quiz=True)
        )
        quiz_id = seeded["quiz"]["id"]
        answers = [{"question_id": q["id"], "answer": q["correct_answer"]}
                   for q in seeded["quiz"]["questions"]]

        # Per-turn-type per-node durations.
        per_node: dict[str, dict[str, list[float]]] = {t: {} for t in TURN_TYPES}
        per_turn_total: dict[str, list[float]] = {t: [] for t in TURN_TYPES}

        for _ in range(runs):
            specs = {
                "teach": (tutoring, initial_state(student, "Teach me about photosynthesis")),
                "quiz": (tutoring, initial_state(
                    student, "Give me a practice quiz on the periodic table", with_quiz=True)),
                "answer": (tutoring, initial_state(
                    student, "What is the chemical equation for photosynthesis?")),
                "submit": (evaluation, {
                    "student_id": student, "quiz_id": quiz_id, "answers": answers,
                    "messages": [], "errors": [], "memory_updates": [],
                }),
            }
            for turn_type, (graph, state) in specs.items():
                node_times, total = await time_turn(graph, state)
                per_turn_total[turn_type].append(total)
                for name, dur in node_times.items():
                    per_node[turn_type].setdefault(name, []).append(dur)

        # Aggregate across all turn types for a combined per-node view.
        all_node_durations: dict[str, list[float]] = {}
        for t in per_node:
            for name, durations in per_node[t].items():
                all_node_durations.setdefault(name, []).extend(durations)

        rows: list[dict] = []
        for node in sorted(all_node_durations):
            times = all_node_durations[node]
            s = latency_stats(times)
            rows.append({
                "node": node,
                "n": s["n"],
                "mean_s": round(s["mean"], 4),
                "median_s": round(s["median"], 4),
                "p50_s": round(s["p50"], 4),
                "p90_s": round(s["p90"], 4),
                "std_s": round(statistics.stdev(times), 4) if len(times) > 1 else 0.0,
                "min_s": round(s["min"], 4),
                "max_s": round(s["max"], 4),
            })

        for turn_type in TURN_TYPES:
            totals = per_turn_total[turn_type]
            s = latency_stats(totals)
            rows.append({
                "node": f"TOTAL_{turn_type}",
                "n": s["n"],
                "mean_s": round(s["mean"], 4),
                "median_s": round(s["median"], 4),
                "p50_s": round(s["p50"], 4),
                "p90_s": round(s["p90"], 4),
                "std_s": round(statistics.stdev(totals), 4) if len(totals) > 1 else 0.0,
                "min_s": round(s["min"], 4),
                "max_s": round(s["max"], 4),
            })

        csv_path = save_csv(rows, "agent_timing")

        # --- Chart: two subplots ---
        # Left:  node bars sorted by mean (aggregated across all turn types)
        # Right: total per turn type (stacked contribution of each node)
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Mean Node Execution Time", "Total Time by Turn Type"],
            column_widths=[0.6, 0.4],
        )

        node_rows = [r for r in rows if not r["node"].startswith("TOTAL_")]
        node_rows.sort(key=lambda r: r["mean_s"], reverse=True)
        fig.add_trace(
            dict(type="bar", name="node mean", x=[r["node"] for r in node_rows],
                 y=[r["mean_s"] for r in node_rows],
                 text=[f"{r['mean_s']:.3f}s" for r in node_rows], textposition="outside",
                 marker_color="#4C78A8", showlegend=False),
            row=1, col=1,
        )

        turn_colors = {"teach": "#4C78A8", "quiz": "#F58518", "answer": "#54A24B",
                       "submit": "#E45756"}
        turn_totals = [(t, statistics.mean(per_turn_total[t])) for t in TURN_TYPES]
        fig.add_trace(
            dict(type="bar", name="total time",
                 x=[t for t, _ in turn_totals], y=[v for _, v in turn_totals],
                 text=[f"{v:.3f}s" for _, v in turn_totals], textposition="outside",
                 marker_color=[turn_colors[t] for t, _ in turn_totals], showlegend=False),
            row=1, col=2,
        )

        fig.update_layout(height=420, template="plotly_white",
                          margin=dict(l=60, r=40, t=60, b=60))
        fig.update_xaxes(tickangle=-30, row=1, col=1)
        fig.update_yaxes(title_text="seconds", rangemode="tozero", row=1, col=1)
        fig.update_yaxes(title_text="seconds", rangemode="tozero", row=1, col=2)
        graph_path = write_chart(fig, "agent_timing")

        teach_totals = per_turn_total["teach"]
        headline = statistics.mean(teach_totals)
        node_detail = ", ".join(f"{r['node']}={r['mean_s']:.3f}s" for r in node_rows)
        return {
            "metric_id": METRIC_ID,
            "metric_name": METRIC_NAME,
            "value": round(headline, 4),
            "unit": UNIT,
            "detail": f"mean teach-turn graph = {headline:.3f}s; " + node_detail,
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
