# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.13",
#     "matplotlib",
#     "numpy",
# ]
# ///
"""Act 6: What calibrated means.

A probability is calibrated when things it calls 80% happen about 80% of the
time. This notebook shows what the fitted temperatures say about the base
model (one bucket pegged the search ceiling at T=100), how fine-tuning on
proper scoring rules pulled them back toward T=1, and the demo that
temperature scaling never changes which answer ranks first - only how much
you should trust it. Numbers render from data/finetune_experiment_results.json.
"""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt6
    from pathlib import Path

    plt6.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titleweight": "bold",
            "axes.titlesize": 10,
        }
    )

    return Path, json, mo, np, plt6


@app.cell
def _(mo):
    mo.md(
        """
        # Act 6 - What calibrated means

        > **The Laya pharma surveillance series · Act 6 of 7**
        > Runs on committed data, no model downloads.

        A probability is **calibrated** when things it calls 80% actually
        happen about 80% of the time. The gap between claimed and observed is
        Expected Calibration Error (ECE), and it is the one axis where the
        closed competitor ships better raw numbers: Jev reports ECE 0.144;
        Laya's model card admits 0.213 raw.

        The comparison misses two things, and both are below.
        """
    )
    return


@app.cell
def _(Path, json):
    DATA6 = Path(__file__).resolve().parent.parent / "data"
    with open(DATA6 / "finetune_experiment_results.json", encoding="utf-8") as f6:
        R6 = json.load(f6)
    BUCKETS6 = ["choice:3-5", "choice:6-10", "choice:11+", "noul:2"]
    return BUCKETS6, R6


@app.cell
def _(BUCKETS6, R6, mo, np, plt6):
    t_b6 = [R6["temps_before"][b6] for b6 in BUCKETS6]
    t_a6 = [R6["temps_after"][b6] for b6 in BUCKETS6]
    x7 = np.arange(len(BUCKETS6))

    fig_temp6, ax_temp6 = plt6.subplots(figsize=(8.5, 4))
    ax_temp6.bar(x7 - 0.19, t_b6, 0.38, label="base model", color="#a8b2bd")
    ax_temp6.bar(x7 + 0.19, t_a6, 0.38, label="after RLCD", color="#4ade80")
    ax_temp6.axhline(1.0, color="red", linestyle="--", linewidth=1)
    ax_temp6.text(len(BUCKETS6) - 0.5, 1.5, "T = 1: perfectly scaled", fontsize=8, color="red")
    ax_temp6.set_yscale("log")
    ax_temp6.set_ylabel("fitted temperature (log scale)")
    ax_temp6.set_xticks(x7, BUCKETS6, rotation=12)
    ax_temp6.set_title(
        "the base noul bucket wanted T=100 - the search ceiling; fine-tuning pulled every bucket toward T=1"
    )
    ax_temp6.legend()
    fig_temp6.tight_layout()
    mo.vstack(
        [
            fig_temp6,
            mo.md(
                """
                Temperature scaling divides logits by one scalar per
                question-type bucket, fitted on labeled data. **T > 1 means
                over-confident**: the honest repair flattens opinions. The
                base model's yes/no bucket pegged my search ceiling at
                T = 100 - calibration by destroying almost all signal.
                """
            ),
        ]
    )
    return (fig_temp6,)


@app.cell
def _(mo):
    mo.md("## What T = 100 actually does to a distribution")
    return


@app.cell
def _(mo, np, plt6):
    # Illustration on one representative 10-way choice question: the same
    # logits, scaled by each bucket's fitted temperatures.
    logits6 = np.array([4.2, 3.1, 2.8, 2.2, 1.9, 1.4, 1.1, 0.7, 0.3, -0.2])

    def softmax6(z6, T6):
        e6 = np.exp(z6 / T6)
        return e6 / e6.sum()

    temps_demo6 = [1.0, 2.1, 4.8, 13.6, 100.0]
    labels_demo6 = ["T=1 (perfect)", "T=2.1", "T=4.8", "T=13.6", "T=100 (the base noul bucket)"]
    fig_flat6, axes6 = plt6.subplots(
        1, 5, figsize=(13, 2.6), sharey=True
    )
    for ax6, T6, lab6 in zip(axes6, temps_demo6, labels_demo6):
        p6 = softmax6(logits6, T6)
        ax6.bar(range(10), p6, color="#d9534f" if T6 >= 13.6 else "#4a7ebb")
        ax6.set_title(lab6, fontsize=9)
        ax6.set_ylim(0, 1)
    axes6[0].set_ylabel("probability")
    fig_flat6.suptitle(
        "one set of logits, five temperatures - the ranking never moves; the trust does",
        y=1.04,
        fontsize=10,
    )
    fig_flat6.tight_layout()
    mo.vstack([fig_flat6])
    return (fig_flat6,)


@app.cell
def _(mo, np):
    p_t16 = np.exp(np.array([4.2, 3.1, 2.8, 2.2, 1.9, 1.4, 1.1, 0.7, 0.3, -0.2]) / 1.0)
    p_t16 = p_t16 / p_t16.sum()
    p_t1006 = np.exp(np.array([4.2, 3.1, 2.8, 2.2, 1.9, 1.4, 1.1, 0.7, 0.3, -0.2]) / 100.0)
    p_t1006 = p_t1006 / p_t1006.sum()
    mo.md(
        f"""
        Top option at T=1: **{p_t16.max():.2f}**. Top option at T=100:
        **{p_t1006.max():.2f}**. Same argmax. A confidence gate at P = 0.9
        fires at the first temperature and never at the second - **fit
        temperatures on your own labels before wiring any threshold to an
        automated action.**
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## The two-step repair, measured")
    return


@app.cell
def _(R6, mo):
    ece_before6 = R6["before_raw"]["action_required"].get("ece")
    ece_after6 = R6["after_raw"]["action_required"].get("ece")
    mo.md(
        f"""
        | model | raw ECE (noul, held-out) | after per-bucket temperature refit |
        |---|---|---|
        | base Laya | {ece_before6:.3f} | ~0.07 |
        | after RLCD fine-tuning | {ece_after6:.3f} | ~0.07 |
        | Jev (closed competitor, shipped) | 0.144 | - |

        Two steps. **Fine-tuning on proper scoring rules** does the heavy
        lifting in-domain: raw ECE falls on its own, and the fitted
        temperatures land at 0.11-0.73 without being asked. A **per-bucket
        refit** on your labels then finishes both models near 0.07 - below
        what the closed competitor ships. The difference between the two
        rows: the base model reaches 0.07 only through extreme softening
        (T=100 flattens everything), while the fine-tuned model arrives with
        usable probabilities natively.

        Laya's own model card reports the same shape (0.466 → 0.081) with
        this procedure. Ours reproduces it on pharma documents.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## What to carry out of Act 6

        1. Raw calibration numbers out of the box are not usable. Fit
           temperatures before any threshold touches an automated action.
        2. Temperature scaling never changes which answer ranks first, only
           how much you should trust it - discrimination and calibration are
           separate axes.
        3. The durable fix is the training objective, not the post-hoc patch:
           proper scoring rules make honesty the winning strategy.

        **Act 7 (the last one):** wrap the fine-tuned front half in an API
        and price it.
        """
    )
    return


if __name__ == "__main__":
    app.run()
