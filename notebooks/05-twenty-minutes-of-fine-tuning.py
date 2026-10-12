# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo==0.25.1",
#     "matplotlib",
#     "numpy",
# ]
# ///
"""Act 5: Twenty minutes of fine-tuning.

The RLCD loop, reimplemented at laptop scale on top of Laya's own training
utilities: group-relative policy gradients with Gaussian logit noise, a
strictly proper scoring reward, full encoder fine-tune. Three epochs over 212
labeled documents, ~20 minutes on an M4 Max. This notebook shows the
before/after on 53 held-out documents. Numbers render from
data/finetune_experiment_results.json; no model downloads.
"""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt5
    from pathlib import Path

    plt5.rcParams.update(
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

    return Path, json, mo, np, plt5


@app.cell
def _(mo):
    mo.md(
        """
        # Act 5 - Twenty minutes of fine-tuning

        > **The Laya pharma surveillance series · Act 5 of 7**
        > Runs on committed data, no model downloads.

        Act 4 ended on the edges: spam dressed as science, press about warning
        letters reading as warnings, and owning-team/action/urgency barely
        above coin-flip zero-shot.

        The fix is not a bigger model. It is 212 labeled documents from your
        own distribution, and 20 minutes of the training loop below.
        """
    )
    return


@app.cell
def _(Path, json):
    DATA5 = Path(__file__).resolve().parent.parent / "data"
    with open(DATA5 / "finetune_experiment_results.json", encoding="utf-8") as f5:
        R5 = json.load(f5)
    QORDER5 = ["doc_type", "secondary_type", "route_to", "action_required", "urgency"]
    QLABELS5 = {
        "doc_type": "doc type (10-way)",
        "secondary_type": "secondary type (11-way)",
        "route_to": "owning team (8-way)",
        "action_required": "action required (yes/no)",
        "urgency": "urgency (3-way)",
    }
    return QLABELS5, QORDER5, R5


@app.cell
def _(mo):
    mo.md(
        """
        ## Held-out results, before and after

        53 test documents the model never trained on. Same schema, same
        checkpoint, one fine-tuning run in between.
        """
    )
    return


@app.cell
def _(QLABELS5, QORDER5, R5, mo):
    table5 = [
        {
            "question": QLABELS5[q5],
            "accuracy before": round(R5["before_raw"][q5]["acc"], 3),
            "accuracy after": round(R5["after_raw"][q5]["acc"], 3),
            "Brier before": round(R5["before_raw"][q5]["brier"], 3),
            "Brier after": round(R5["after_raw"][q5]["brier"], 3),
        }
        for q5 in QORDER5
    ]
    mo.ui.table(table5, selection=None)
    return


@app.cell
def _(QLABELS5, QORDER5, R5, mo, np, plt5):
    x5 = np.arange(len(QORDER5))
    w5 = 0.38
    acc_b5 = [R5["before_raw"][q5]["acc"] for q5 in QORDER5]
    acc_a5 = [R5["after_raw"][q5]["acc"] for q5 in QORDER5]

    fig_acc5, ax_acc5 = plt5.subplots(figsize=(9, 3.8))
    ax_acc5.bar(x5 - w5 / 2, acc_b5, w5, label="before", color="#a8b2bd")
    ax_acc5.bar(x5 + w5 / 2, acc_a5, w5, label="after RLCD", color="#4ade80")
    ax_acc5.set_xticks(x5, [QLABELS5[q5] for q5 in QORDER5], rotation=12)
    ax_acc5.set_ylim(0, 1.05)
    ax_acc5.set_ylabel("held-out accuracy (53 docs)")
    ax_acc5.legend()
    fig_acc5.tight_layout()
    mo.vstack([fig_acc5])
    return (fig_acc5,)


@app.cell
def _(QLABELS5, QORDER5, R5, mo, np, plt5):
    brier_b5 = [R5["before_raw"][q5]["brier"] for q5 in QORDER5]
    brier_a5 = [R5["after_raw"][q5]["brier"] for q5 in QORDER5]
    ls_b5 = [R5["before_raw"][q5]["log_score"] for q5 in QORDER5]
    ls_a5 = [R5["after_raw"][q5]["log_score"] for q5 in QORDER5]
    x6 = np.arange(len(QORDER5))
    w6 = 0.38

    fig_brier5, (axL5, axR5) = plt5.subplots(1, 2, figsize=(11, 3.8))
    axL5.bar(x6 - w6 / 2, ls_b5, w6, label="before", color="#a8b2bd")
    axL5.bar(x6 + w6 / 2, ls_a5, w6, label="after RLCD", color="#4ade80")
    axL5.set_title("mean log score (higher is better)")
    axL5.set_xticks(x6, [QLABELS5[q5] for q5 in QORDER5], rotation=12)
    axL5.legend()
    axR5.bar(x6 - w6 / 2, brier_b5, w6, label="before", color="#a8b2bd")
    axR5.bar(x6 + w6 / 2, brier_a5, w6, label="after RLCD", color="#4ade80")
    axR5.set_title("Brier score (lower is better)")
    axR5.set_xticks(x6, [QLABELS5[q5] for q5 in QORDER5], rotation=12)
    axR5.legend()
    fig_brier5.tight_layout()
    mo.vstack(
        [
            fig_brier5,
            mo.md(
                """
                Accuracy is the headline; the probability metrics are the
                story. Log score and Brier measure the quality of the whole
                distribution, and both moved 3-7x. A model can luck into
                argmax accuracy. It cannot luck into a distribution this much
                sharper across 53 unseen documents.
                """
            ),
        ]
    )
    return (fig_brier5,)


@app.cell
def _(mo):
    mo.md("## The loop, all 20 minutes of it")
    return


@app.cell
def _(mo):
    mo.md(
        f"""
        ```python
        EPOCHS = 3        # over 212 labeled documents
        LR = 1e-5         # AdamW, full encoder fine-tune
        G = 4             # noise candidates per question (the "group")
        SIGMA = 0.3       # Gaussian logit-noise scale (exploration)

        # inside train_epoch, per batch:
        #   1. one no-grad forward pass -> option logits z
        z = z + SIGMA * torch.randn_like(z)     # 2. G noisy candidates
        #   3. score each candidate with a strictly proper reward:
        #      proper_reward = log score + spherical score + RPS
        r = proper_reward(p, tgt, qtg, msk)     # [G, N]
        #   4. group-normalize r into advantages (GRPO-style),
        #      weight the differentiable reward, plus a small
        #      soft-CE anchor (0.3) for stability on tiny data
        ```
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        The reward is the whole trick. Proper scoring rules have one
        property that matters: the only way to maximize expected reward is
        to report honest probabilities. Fine-tune against that reward and
        calibration falls out as a side effect of the objective, which is
        exactly what Act 6 measures.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## The price of the lever

        212 hand-labeled documents. 20 minutes on an M4 Max laptop, or a
        free Kaggle GPU. No infrastructure, no API bills, the checkpoint
        stays yours.

        **Act 6:** accuracy was the easy part. The question that decides
        whether you can wire a P = 0.9 threshold to an automated action is
        what the probabilities actually mean.
        """
    )
    return


if __name__ == "__main__":
    app.run()
