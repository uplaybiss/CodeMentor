"""Evaluation Suite Runner — runs all 8 benchmarks and writes summary.

Usage::

    python evaluation/run_all.py          # run everything
    python evaluation/run_all.py --only m1 m3 m5   # run specific metrics

HTTP benchmarks (M1, M2, M5, M6, M8) require the server to be running::

    python main.py          # in one terminal
    python evaluation/run_all.py   # in another

Hermetic benchmarks (M3, M4, M7) need NO server — they build a temporary
container with their own SQLite DB and ChromaDB directory.

Output
------
``evaluation/results/summary_metrics.csv`` — one row per metric
``evaluation/results/summary.json``        — full structured report
``evaluation/results/graphs/index.html``   — dashboard linking every chart
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CSV_DIR = RESULTS_DIR / "csv"
GRAPH_DIR = RESULTS_DIR / "graphs"

ALL_METRICS = [
    ("m1", "benchmark_accuracy",       "Response Accuracy"),
    ("m2", "benchmark_quiz",           "Quiz Generation Quality"),
    ("m3", "benchmark_rag",            "Retrieval Accuracy (RAG)"),
    ("m4", "benchmark_hallucination",  "Hallucination Rate"),
    ("m5", "benchmark_latency",        "Response Latency"),
    ("m6", "benchmark_memory",         "Memory Recall Accuracy"),
    ("m7", "benchmark_agents",         "Agent Execution Time"),
    ("m8", "benchmark_e2e",            "End-to-End Workflow Time"),
]

METRIC_IDS = {m[0] for m in ALL_METRICS}


def _get_metric_ids(args: list[str] | None = None) -> list[tuple[str, str, str]]:
    if args:
        ids = set()
        for a in args:
            a = a.strip().lower()
            if a.startswith("m"):
                ids.add(a)
            else:
                ids.add(f"m{a}")
        return [m for m in ALL_METRICS if m[0] in ids]
    return ALL_METRICS


def _write_dashboard_html(summaries: list[dict]) -> Path:
    """Write a simple HTML dashboard linking to every chart."""
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    path = GRAPH_DIR / "index.html"
    rows = ""
    for s in summaries:
        graph = s.get("graph", "")
        graph_file = Path(graph).name if graph else ""
        if graph_file:
            link = f'<a href="{graph_file}" target="_blank">{graph_file}</a>'
        else:
            link = '<em>(no chart)</em>'
        val = s.get("value", "?")
        rows += (
            f"<tr>"
            f"<td>{s.get('metric_id', '')}</td>"
            f"<td>{s.get('metric_name', '')}</td>"
            f"<td>{val}</td>"
            f"<td>{s.get('unit', '')}</td>"
            f"<td>{link}</td>"
            f"</tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>EduAgent Evaluation Dashboard</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
  h1 {{ color: #333; }}
  table {{ border-collapse: collapse; width: 100%; max-width: 960px; }}
  th, td {{ text-align: left; padding: 10px 14px; border: 1px solid #ddd; }}
  th {{ background: #f5f5f5; }}
  tr:hover {{ background: #f9f9f9; }}
</style>
</head>
<body>
<h1>EduAgent Evaluation Dashboard</h1>
<table>
<tr><th>ID</th><th>Metric</th><th>Value</th><th>Unit</th><th>Chart</th></tr>
{rows}
</table>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")
    return path


def main() -> None:
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    if "--only" in args:
        idx = args.index("--only")
        only_ids = args[idx + 1:]
        args = args[:idx]
    else:
        only_ids = None

    metrics = _get_metric_ids(only_ids)
    summaries: list[dict] = []

    for metric_id, module_name, metric_name in metrics:
        print(f"\n{'='*60}")
        print(f"Running {metric_id}: {metric_name}")
        print(f"{'='*60}")
        try:
            mod = importlib.import_module(module_name)
            result = mod.run()
            summaries.append(result)
            print(f"  -> {metric_name} = {result.get('value', '?')} {result.get('unit', '')}")
            print(f"     {result.get('detail', '')}")
        except Exception as exc:
            print(f"  ** ERROR: {exc}")
            summaries.append({
                "metric_id": metric_id,
                "metric_name": metric_name,
                "value": None,
                "unit": "error",
                "detail": str(exc),
            })

    # ---- Write summary outputs ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    # summary_metrics.csv
    csv_rows = [
        {k: s.get(k, "") for k in ["metric_id", "metric_name", "value", "unit", "detail"]}
        for s in summaries
    ]
    import pandas as pd
    pd.DataFrame(csv_rows).to_csv(CSV_DIR / "summary_metrics.csv", index=False)

    # summary.json
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summaries, indent=2, default=str), encoding="utf-8"
    )

    # Dashboard HTML
    dashboard_path = _write_dashboard_html(summaries)

    print(f"\n{'='*60}")
    print("Evaluation complete.")
    print(f"  CSV summary : {CSV_DIR / 'summary_metrics.csv'}")
    print(f"  JSON report : {RESULTS_DIR / 'summary.json'}")
    print(f"  Dashboard   : {dashboard_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
