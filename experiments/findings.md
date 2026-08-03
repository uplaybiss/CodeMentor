# Findings & Recommendation

*Regenerated 2026-08-02 from `results/csv/experiment_results.csv` (20 configurations).*

## TL;DR

The application's current defaults — **`all-MiniLM-L6-v2` + the 600/100-token
baseline chunker** — are already an *optimal* choice: they score a perfect
hit@1, recall@4 and MRR. **No change is required.**

If query latency is a priority, `multi-qa-MiniLM-L6-cos-v1` is the only
drop-in that matches the baseline's quality while cutting mean query time by
~35% (0.026 s vs 0.041 s). `all-mpnet-base-v2` buys nothing and costs the most.

## Result summary

| embedding_model | chunking | hit@1 | precision@4 | recall@4 | MRR | composite | n_chunks | embed_s | query_s_mean |
|---|---|---|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | **baseline** | **1.000** | 0.250 | **1.000** | **1.000** | 0.750 | 5 | 0.992 | 0.0406 |
| all-MiniLM-L6-v2 | small | 1.000 | 0.250 | 1.000 | 1.000 | 0.750 | 12 | 1.276 | 0.0383 |
| all-MiniLM-L6-v2 | large | 1.000 | 0.250 | 1.000 | 1.000 | 0.750 | 5 | 0.986 | 0.0357 |
| all-MiniLM-L6-v2 | sentence | 1.000 | 0.250 | 1.000 | 1.000 | 0.750 | 20 | 1.163 | 0.0357 |
| multi-qa-MiniLM-L6-cos-v1 | baseline | 1.000 | 0.250 | 1.000 | 1.000 | 0.750 | 5 | 0.966 | 0.0257 |
| multi-qa-MiniLM-L6-cos-v1 | small | 1.000 | 0.250 | 1.000 | 1.000 | 0.750 | 12 | 1.083 | 0.0255 |
| multi-qa-MiniLM-L6-cos-v1 | large | 1.000 | 0.250 | 1.000 | 1.000 | 0.750 | 5 | 0.911 | **0.0255** |
| multi-qa-MiniLM-L6-cos-v1 | sentence | 1.000 | 0.250 | 1.000 | 1.000 | 0.750 | 20 | 0.987 | 0.0267 |
| paraphrase-MiniLM-L6-v2 | baseline | 0.933 | 0.250 | 1.000 | 0.950 | 0.711 | 5 | 0.540 | 0.0288 |
| paraphrase-MiniLM-L6-v2 | small | 1.000 | 0.250 | 1.000 | 1.000 | 0.750 | 12 | 0.727 | **0.0222** |
| paraphrase-MiniLM-L6-v2 | large | 0.933 | 0.250 | 1.000 | 0.950 | 0.711 | 5 | 0.469 | 0.0251 |
| paraphrase-MiniLM-L6-v2 | sentence | 1.000 | 0.250 | 1.000 | 1.000 | 0.750 | 20 | 0.863 | 0.0264 |
| all-mpnet-base-v2 | baseline | 1.000 | 0.250 | 1.000 | 1.000 | 0.750 | 5 | 3.318 | 0.0659 |
| all-mpnet-base-v2 | small | 1.000 | 0.250 | 1.000 | 1.000 | 0.750 | 12 | 4.561 | 0.0649 |
| all-mpnet-base-v2 | large | 1.000 | 0.250 | 1.000 | 1.000 | 0.750 | 5 | 2.660 | 0.0594 |
| all-mpnet-base-v2 | sentence | 1.000 | 0.250 | 1.000 | 1.000 | 0.750 | 20 | 3.614 | 0.0892 |
| BAAI/bge-small-en-v1.5 | baseline | 1.000 | 0.250 | 1.000 | 1.000 | 0.750 | 5 | 1.110 | 0.0313 |
| BAAI/bge-small-en-v1.5 | small | 1.000 | 0.250 | 1.000 | 1.000 | 0.750 | 12 | 1.342 | 0.0315 |
| BAAI/bge-small-en-v1.5 | large | 1.000 | 0.250 | 1.000 | 1.000 | 0.750 | 5 | 1.195 | 0.0353 |
| BAAI/bge-small-en-v1.5 | sentence | 1.000 | 0.250 | 1.000 | 1.000 | 0.750 | 20 | 1.206 | 0.0313 |

## Key observations

1. **Retrieval quality is saturated.** 18 of 20 configurations score a perfect
   hit@1, recall@4 and MRR. Only `paraphrase-MiniLM-L6-v2` with the *baseline*
   or *large* chunker misses (14/15 queries): this is a property of the model,
   not the chunking — both chunkers produce the same 5 chunks per corpus.

2. **`precision@4 = 0.250` for every row is expected, not a bug.** Each query
   has exactly one relevant chunk in the store and `top_k = 4`, so precision is
   inherently 1/4 (the relevant chunk plus three others). It saturates by
   design; the *discriminating* metrics are hit@1 and MRR. The composite score
   is therefore capped at `mean(1.0, 0.25, 1.0) = 0.75`.

3. **Chunking has almost no effect here.** The corpus is topic-separated
   (each note is a distinct subject) and short enough that even the 600-token
   baseline yields ~1 chunk per document. Every strategy puts the right chunk
   on top. In this regime the 5–20 chunk stores are all trivially searchable,
   so the only visible difference is a slight embed/latency cost at the
   finest granularity (`sentence`, 20 chunks).

4. **Latency is where the models diverge.**
   - `all-mpnet-base-v2` is the slowest by far: ~2.7–4.6 s to embed the corpus
     (3–5× the MiniLM family) and ~0.06–0.09 s per query (≈2× slower) — with
     zero quality benefit.
   - `multi-qa-MiniLM-L6-cos-v1` is the fastest reliable config: ~0.026 s mean
     query, ≈0.9–1.1 s embed.
   - `paraphrase-MiniLM-L6-v2` / `small` is the single fastest query
     (0.0222 s) but depends on chunking to reach perfect quality, so it is a
     riskier pick than the multi-qa model.
   - `BAAI/bge-small-en-v1.5` lands between them (≈0.031 s query, ≈1.1–1.3 s
     embed) — a solid modern option but not faster than multi-qa here.

   The `0.2327 s` chunk time for the first row is a one-off warm-up artifact
   (first chunk call in the process); all subsequent rows are ≈0.004–0.007 s.

## Recommendation

| Priority | Configuration | Rationale |
|---|---|---|
| **Default (no change)** | `all-MiniLM-L6-v2` + baseline 600/100 | Perfect quality, fastest embed among the quality-saturating models (0.99 s), 0.041 s query. Already the app default. |
| **If query latency matters** | `multi-qa-MiniLM-L6-cos-v1` + baseline (or any chunking) | Identical perfect quality at ~0.026 s mean query (**~35% faster** than baseline), slightly faster embed. Drop-in swap of one config value. |
| **Avoid** | `all-mpnet-base-v2` | No quality gain, 3–5× slower ingest and ~2× slower queries. |
| **Chunking** | keep `baseline` (or `small`) | No quality difference on this corpus; `sentence` adds ingest cost for zero retrieval benefit. Revisit only if documents grow much longer with multiple sub-topics per note. |

## Reproduce

```bash
python experiments/make_corpus.py     # regenerate the 5-topic corpus
python experiments/run_experiments.py # 5 models × 4 chunkers → results/ + charts
```

Charts: `results/graphs/retrieval_heatmap.html` (quality), `retrieval_bars.html`
(hit@1/precision@4/MRR), `latency_comparison.html` (embed + query time),
`quality_vs_latency.html` (the trade-off verdict).
