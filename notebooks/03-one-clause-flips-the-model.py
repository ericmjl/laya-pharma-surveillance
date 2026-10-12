# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.13",
#     "laya==0.3.5",
#     "matplotlib",
#     "numpy",
# ]
# ///
"""Act 3: One clause flips the model.

50 synthetic social-media posts about a fictional drug, 13 languages, 38 real
adverse-event reports and 12 pieces of chatter, hand-labeled. The first
version of the detection question contained a helpful-looking exclusion
clause. It detected 0 of 38 adverse events. This notebook runs BOTH schemas
live and shows the polarity flip. First run downloads both checkpoints.
"""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt3
    import json
    from pathlib import Path

    plt3.rcParams.update(
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

    return Path, json, mo, np, plt3


@app.cell
def _(mo):
    mo.md(
        """
        # Act 3 - One clause flips the model

        > **The Laya pharma surveillance series · Act 3 of 7**
        > Run me: `uvx marimo edit --sandbox notebooks/03-one-clause-flips-the-model.py`

        Scenario: a pharmacovigilance team monitors social media for adverse
        events of a fictional drug, "zorvafenib". 50 posts, 13 languages,
        hand-authored ground truth: 38 adverse-event reports, 12 pieces of
        chatter (price complaints, praise, availability, spam).

        The first run of the pipeline detected **0 of 38** adverse events.
        The pipeline was fine; the question was broken.
        """
    )
    return


@app.cell
def _(Path, json, mo):
    DATA3 = Path(__file__).resolve().parent.parent / "data"
    with open(DATA3 / "synthetic_ae_dataset.json", encoding="utf-8") as f3:
        POSTS3 = json.load(f3)

    n_ae3 = sum(1 for p3 in POSTS3 if p3["gt"][0])
    n_chatter3 = len(POSTS3) - n_ae3
    n_langs3 = len({p3["lang"] for p3 in POSTS3})

    preview_rows3 = [
        {
            "id": p3["id"],
            "lang": p3["lang"],
            "is_ae": p3["gt"][0],
            "text": p3["text"][:90] + ("..." if len(p3["text"]) > 90 else ""),
        }
        for p3 in (POSTS3[:4] + POSTS3[-4:])
    ]
    mo.vstack(
        [
            mo.md(
                f"**The dataset**: {len(POSTS3)} posts · {n_ae3} adverse events · "
                f"{n_chatter3} chatter · {n_langs3} languages · labels are authorial ground truth"
            ),
            mo.ui.table(preview_rows3, selection=None),
        ]
    )
    return POSTS3, n_ae3, n_chatter3, n_langs3


@app.cell
def _(mo):
    mo.md(
        """
        ## The same question, drafted twice

        The detection question went through two drafts. The first one reads
        like thorough prompt engineering to a human. To the yes/no head it
        reads like a polarity flip.
        """
    )
    return


@app.cell
def _(mo):
    LONG3 = (
        "Does the author of this post describe a symptom or health problem "
        "that they themselves experienced while taking the medication? Adverse event reports "
        "include physical or mental symptoms, however mild. Exclude posts about price, "
        "availability, refills, general questions, praise, news, or community chat."
    )
    SHORT3 = (
        "Does the author describe a symptom they experienced while taking the medication?"
    )
    mo.hstack(
        [
            mo.md(
                f"""**Schema A - the "thorough" draft** (broken)\n\n```text\n{LONG3}\n```"""
            ),
            mo.md(
                f"""**Schema B - the fix** (short, positively phrased)\n\n```text\n{SHORT3}\n```"""
            ),
        ],
        widths=[1.4, 1],
    )
    return LONG3, SHORT3


@app.cell
def _(mo):
    import contextlib
    import os

    os.environ.setdefault("USE_TF", "0")

    import laya

    load_ctx3 = (
        mo.spinner("loading both checkpoints ...")
        if hasattr(mo, "spinner")
        else contextlib.nullcontext()
    )
    with load_ctx3:
        english3 = laya.load("convaiinnovations/laya")
        multi3 = laya.load("convaiinnovations/laya-multilingual")
    return contextlib, english3, multi3


@app.cell
def _(LONG3, POSTS3, SHORT3, contextlib, english3, mo, multi3, json, Path):
    DATA3b = Path(__file__).resolve().parent.parent / "data"
    CACHE3 = DATA3b / "pv_schema_comparison.json"

    def is_ae_pass3(agent3, instructions3, post3):
        q3 = {"is_ae": {"type": "noul", "instructions": instructions3}}
        return agent3.predict({"post": post3}, q3)["answers"]["is_ae"]["noul"]

    if CACHE3.exists():
        with open(CACHE3, encoding="utf-8") as f3b:
            PREDS3 = json.load(f3b)
        ran_note3 = f"(loaded from {CACHE3.name}; delete it to re-run live)"
    else:
        PREDS3 = {}
        run_ctx3 = (
            mo.spinner(
                "running 200 single-question forward passes (~1 min on M-series CPU) ..."
            )
            if hasattr(mo, "spinner")
            else contextlib.nullcontext()
        )
        with run_ctx3:
            for ck3, ag3 in [("english", english3), ("multilingual", multi3)]:
                for schema3, ins3 in [("long", LONG3), ("short", SHORT3)]:
                    for p3 in POSTS3:
                        PREDS3.setdefault(ck3, {}).setdefault(schema3, {})[p3["id"]] = (
                            is_ae_pass3(ag3, ins3, p3["text"])
                        )
        with open(CACHE3, "w", encoding="utf-8") as f3b:
            json.dump(PREDS3, f3b)
        ran_note3 = "(computed live; cached to data/pv_schema_comparison.json)"

    LATIN3 = {"en", "es", "fr", "de", "pt", "it", "sw"}

    def det3(p3s, preds3):
        tp3 = sum(1 for x3 in p3s if x3["gt"][0] and preds3[x3["id"]] >= 0.5)
        fn3 = sum(1 for x3 in p3s if x3["gt"][0] and preds3[x3["id"]] < 0.5)
        fp3 = sum(1 for x3 in p3s if not x3["gt"][0] and preds3[x3["id"]] >= 0.5)
        return tp3, fn3, fp3

    summary3 = []
    for label3, get3 in [
        ("english + schema A (long)", lambda x3: PREDS3["english"]["long"][x3["id"]]),
        ("english + schema B (short)", lambda x3: PREDS3["english"]["short"][x3["id"]]),
        (
            "multilingual + schema A (long)",
            lambda x3: PREDS3["multilingual"]["long"][x3["id"]],
        ),
        (
            "multilingual + schema B (short)",
            lambda x3: PREDS3["multilingual"]["short"][x3["id"]],
        ),
        (
            "routed pipeline + schema B (latin->en, else multi)",
            lambda x3: PREDS3["english" if x3["lang"] in LATIN3 else "multilingual"][
                "short"
            ][x3["id"]],
        ),
        (
            "routed pipeline + schema A (latin->en, else multi)",
            lambda x3: PREDS3["english" if x3["lang"] in LATIN3 else "multilingual"][
                "long"
            ][x3["id"]],
        ),
    ]:
        tp3, fn3, fp3 = det3(POSTS3, [get3(x3) for x3 in POSTS3] and get3 and None or None) if False else det3(
            POSTS3, {x3["id"]: get3(x3) for x3 in POSTS3}
        )
        summary3.append(
            {
                "configuration": label3,
                "AE detected": f"{tp3}/38",
                "missed": fn3,
                "false alarms": fp3,
            }
        )
    return PREDS3, summary3


@app.cell
def _(mo, summary3):
    mo.vstack(
        [
            mo.ui.table(summary3, selection=None),
            mo.md(
                """
                Schema A: the exclusion clause, written to *help*, suppressed
                every adverse event. Schema B: same posts, same threshold,
                same checkpoints. The clause did not add context. It inverted
                the head.
                """
            ),
        ]
    )
    return


@app.cell
def _(POSTS3, json, mo, plt3, Path):
    DATA3c = Path(__file__).resolve().parent.parent / "data"
    with open(DATA3c / "pv_schema_comparison.json", encoding="utf-8") as f3c:
        P3 = json.load(f3c)

    xs3 = [P3["multilingual"]["long"][x3["id"]] for x3 in POSTS3]
    ys3 = [P3["multilingual"]["short"][x3["id"]] for x3 in POSTS3]
    cols3 = ["#d9534f" if x3["gt"][0] else "#9aa5b1" for x3 in POSTS3]

    fig_flip3, ax_flip3 = plt3.subplots(figsize=(6.5, 6))
    ax_flip3.scatter(xs3, ys3, c=cols3, s=42, alpha=0.85, edgecolors="none")
    ax_flip3.plot([0, 1], [0, 1], color="grey", linewidth=0.8, linestyle="--")
    ax_flip3.axvline(0.5, color="#d9534f", linewidth=0.8, linestyle=":")
    ax_flip3.axhline(0.5, color="#4ade80", linewidth=0.8, linestyle=":")
    ax_flip3.set_xlabel("P(adverse event) under schema A (long, broken)")
    ax_flip3.set_ylabel("P(adverse event) under schema B (short, fixed)")
    ax_flip3.set_title("same posts, same checkpoint - red = true adverse event")
    ax_flip3.set_xlim(-0.03, 1.03)
    ax_flip3.set_ylim(-0.03, 1.03)
    fig_flip3.tight_layout()
    mo.vstack(
        [
            fig_flip3,
            mo.md(
                "Every red point below the diagonal is an adverse event the "
                "long schema pushed *down*. The clause read as exclusion "
                "evidence: mention price, praise, or news, and the post gets "
                "scored as not-a-symptom - even when it describes one."
            ),
        ]
    )
    return


@app.cell
def _(POSTS3, json, mo, Path):
    DATA3d = Path(__file__).resolve().parent.parent / "data"
    with open(DATA3d / "pv_schema_comparison.json", encoding="utf-8") as f3d:
        P4 = json.load(f3d)

    praise3 = next(x3 for x3 in POSTS3 if x3["id"] == "D-01")
    er3 = next(x3 for x3 in POSTS3 if x3["id"] == "GI-05")
    mo.md(
        f"""
        ## Two posts to remember

        **The praise post that became an "adverse event"** (`D-01`, en)

        > {praise3["text"]}

        P(is_ae): **{P4["multilingual"]["long"]["D-01"]:.2f}** under schema A →
        **{P4["multilingual"]["short"]["D-01"]:.2f}** under schema B.
        The word "Exclude ... praise" made the head look *for* praise.

        **The emergency-room post that vanished** (`GI-05`, zh)

        > {er3["text"]}

        P(is_ae): **{P4["multilingual"]["long"]["GI-05"]:.2f}** under schema A →
        **{P4["multilingual"]["short"]["GI-05"]:.2f}** under schema B.
        A hospital visit scored at near zero, in a pipeline that archives
        everything below threshold.

        ## What to carry out of Act 3

        Question schemas are code. They deserve review discipline in your pull
        requests: keep decision questions short and positively phrased, and
        re-run a labeled probe set whenever a schema changes. A clause that
        reads like helpful context to a human acted like a polarity flip on
        this model.
        """
    )
    return


if __name__ == "__main__":
    app.run()
