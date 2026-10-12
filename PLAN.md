# PLAN: a show-not-tell notebook series

The blog post tells the story of stress-testing Laya on pharma surveillance.
This repo shows it. Seven notebooks, one act each. Every section opens with an
artifact (a live model call, a chart, a table of real outputs) and the takeaway
is one sentence after it. No walls of prose.

## The series

| # | notebook | the moment (what you see) | runs |
|---|---|---|---|
| 01 | `01-one-forward-pass.py` | The same support email answered four ways in one 40ms forward pass: `choice`, `score`, `noul` primitives side by side, with a visible timer. Framed against what the same job costs an LLM. | live |
| 02 | `02-the-confidence-trap.py` | Khmer text into the English checkpoint: 0.952 confidence at random accuracy. Then the Router catching the script *before* the forward pass. Then the same billing complaint in 11 languages, consistent and fast. | live |
| 03 | `03-one-clause-flips-the-model.py` | Same 50 synthetic posts, two question schemas. The exclusion-clause schema: 0/38 adverse events detected, praise post at P=0.66. The positively-phrased schema: 30/38, zero false alarms. Per-post probability scatter, before and after. | live, caches to `data/` |
| 04 | `04-real-documents.py` | The corpus: 50 → 265 real scraped docs (FDA, PubMed, DOJ, EDGAR, Federal Register, FiercePharma, docket comments). Zero-shot 100% on the 50-doc operational categories, then the failure gallery: the FiercePharma article *about* a warning letter, spam dressed as science. | data |
| 05 | `05-twenty-minutes-of-fine-tuning.py` | Before/after RLCD on held-out docs: five questions, accuracy and Brier, table and paired bars. Predictions that flipped, shown as documents. The training loop code is shown; re-running is a 20-minute laptop job, results are inlined. | data |
| 06 | `06-what-calibrated-means.py` | Reliability diagrams before and after. Base model so overconfident the fitted temperature pegs the ceiling (T=100); fine-tuned lands at 0.11–0.73. ECE 0.46 → 0.26 → ~0.07 vs Jev's shipped 0.144. Plus the demo that temperature scaling never changes the ranking, only the trust. | data |
| 07 | `07-the-cheap-front-half.py` | Deployment economics from inlined Modal measurements: CPU vs T4 latency distributions, cold starts, cost per 1M requests, break-even concurrency. Closes on the cases the cheap model missed (the BEC scam at P(phishing)=0.000): why a System 2 stays on the funnel. | data |

## Design rules

1. **Artifact first, sentence second.** A section is a model call or a chart,
   then one line of what to notice. If a paragraph is needed, the notebook is
   telling, not showing; cut it or move it to the blog post.
2. **One moment per notebook.** Each act has a single output that carries the
   point (Khmer 0.952; 0/38 → 30/38; the reliability curves crossing). Build
   toward it; end on it.
3. **Inlined data by default, live inference only when the point is the live
   behavior.** Numbers live in `data/*.json` (committed from
   `nll-ai/laya-demo`) so every figure re-renders anywhere, same contract as
   the blog companion notebook. Live cells (01, 02, 03) are marked, stay
   under ~1s per call, and notebook 03 writes its per-post results to
   `data/pv_schema_comparison.json` on first run so the flip chart stays
   reproducible without a model.
4. **Each notebook is standalone** and runs top-to-bottom:
   `uvx marimo edit --sandbox notebooks/NN-name.py`. Dependencies declared in
   inline script metadata (`# /// script` block), per repo convention.
5. **The laya package is the only heavyweight dependency.** Notebooks that run
   live need the model checkpoint (~400MB download on first run); notebooks
   that run on data need nothing but marimo + plotly.

## Data provenance

- `data/synthetic_ae_dataset.json` — 50 synthetic posts, 13 languages,
  ground-truth labels authored by hand (notebook 03).
- `data/doc_triage_real_results.json` — first 50 real scraped docs + zero-shot
  predictions (notebook 04).
- `data/real_docs_corpus_large.json` — the scaled 215-doc corpus (with the
  first 50, 265 total; notebook 04).
- `data/finetune_experiment_results.json` — before/after metrics, fitted
  temperatures, per-bucket calibration (notebooks 05, 06).
- Notebook 03 computes its schema comparison live (both checkpoints, 200
  single-question forward passes) and caches per-post probabilities to
  `data/pv_schema_comparison.json`; delete that file to re-run. The
  0/38 → 30/38 numbers come from `pv_compare.py` / `diag_noul2.py` in
  `nll-ai/laya-demo`.
- Notebook 07's Modal measurements are inlined from the benchmark harness in
  `nll-ai/laya-demo/modal/benchmark.py`; the cost-per-1M math is computed in
  the notebook from those constants.

## Build order

1, 2, 3 are the hook (fast, live, dramatic). Then 3's data prep, then 5, 6
(share `finetune_experiment_results.json`), then 4, 7. One notebook per PR into
this repo so each act gets reviewed as a unit.
