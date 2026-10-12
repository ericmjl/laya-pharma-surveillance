# laya-pharma-surveillance

Marimo notebooks that **show**, not tell, what an open decision model can and
cannot do on pharma surveillance workloads.

The subject is [Laya](https://huggingface.co/convaiinnovations/laya), an
Apache-2.0 421M-parameter decision model: typed questions in, probability
distributions out, one forward pass, no text generation. The story ran as a
[blog post](https://ericmjl.github.io/website/blog/2026/9/24/stress-testing-laya-pharma-surveillance/);
this repo is the same story split into seven acts you can run.

## The acts

1. **One forward pass** — four answers, 40 milliseconds, zero text generated.
2. **The confidence trap** — 0.952 confidence at random accuracy, and the Router that catches it.
3. **One clause flips the model** — 0/38 → 30/38 adverse events detected by rephrasing one question.
4. **Real documents** — 265 scraped pharma docs, where zero-shot shines and where it fumbles.
5. **Twenty minutes of fine-tuning** — held-out accuracy 0.26–0.58 → 0.89–0.96.
6. **What calibrated means** — reliability diagrams, T=100 → T~0.5, ECE below the closed competitor's shipped number.
7. **The cheap front half** — CPU vs GPU economics, and why a System 2 stays on the funnel.

Details and build order: [PLAN.md](PLAN.md).

## Run

```bash
uvx marimo edit --sandbox notebooks/01-one-forward-pass.py
```

Each notebook is standalone. Notebooks 01–03 make live model calls (first run
downloads the ~400MB checkpoint); 04–07 render from `data/*.json` with no
model download.

## Provenance

Experiment artifacts come from [`nll-ai/laya-demo`](https://github.com/nll-ai/laya-demo)
(synthetic dataset, scraped corpora, fine-tuning and calibration results).
Fine-tuning scripts: `laya_training_src/` there. Companion blog notebook:
`laya_pharma_walkthrough.py` there and in the
[website repo](https://github.com/ericmjl/website/tree/main/content/blog/stress-testing-laya-pharma-surveillance).
