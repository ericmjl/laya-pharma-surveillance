# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo==0.25.1",
#     "plotly==7.1.0",
#     "marimo-lens==0.2.3",
#     "marimo-studio==0.3.0",
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

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import json
    from pathlib import Path

    import marimo as mo
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Pharmacopeia theme (see .agents/skills/laya-pharma-theme/SKILL.md)
    INK = "#0F2233"  # clinical navy: banners, dark cards
    INK2 = "#16303F"  # gradient end, chip fills
    TEAL = "#1F7A8C"  # assay teal: primary data series
    GREEN = "#2F9E68"  # recovery green: safe / selected
    AMBER = "#C9821F"  # caution amber: urgency / warning
    CRIMSON = "#B3282D"  # adverse crimson: danger ONLY
    SLATE = "#8A97A5"  # secondary series
    MIST = "#A7B3BD"  # muted text on ink
    KICK = "#8FB0B8"  # kicker text on ink
    LINE = "#D9D4CA"  # warm hairline
    PAPER = "#F7F5F1"  # light card background
    return (
        CRIMSON,
        GREEN,
        INK,
        INK2,
        KICK,
        MIST,
        Path,
        SLATE,
        TEAL,
        go,
        json,
        make_subplots,
        mo,
    )


@app.cell(hide_code=True)
def banner(
    ACC_AFTER5,
    ACC_BEFORE5,
    CRIMSON,
    GREEN,
    INK,
    INK2,
    KICK,
    MIST,
    N_TEST5,
    SLATE,
    TEAL,
    mo,
):
    banner_w_before5 = [round(a * 90) for a in ACC_BEFORE5]
    banner_w_after5 = [round(a * 90) for a in ACC_AFTER5]
    banner_ys5 = [62, 81, 100, 119, 138]
    banner_pairs5 = "".join(
        f'<rect x="806" y="{y}" width="{banner_w_before5[i]}" height="6" rx="3" fill="{SLATE}"/>'
        f'<rect x="806" y="{y + 8}" width="{banner_w_after5[i]}" height="6" rx="3" fill="{GREEN}"/>'
        for i, y in enumerate(banner_ys5)
    )
    mo.Html(
        f"""
        <div style="line-height:0;border-radius:16px;overflow:hidden;">
        <svg viewBox="0 0 960 210" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Act 5 - Twenty minutes of fine-tuning">
          <defs>
            <linearGradient id="bg5" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stop-color="{INK}"/><stop offset="1" stop-color="{INK2}"/>
            </linearGradient>
            <linearGradient id="acc5" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0" stop-color="{TEAL}"/><stop offset="0.55" stop-color="{GREEN}"/><stop offset="1" stop-color="{CRIMSON}"/>
            </linearGradient>
          </defs>
          <rect x="1" y="1" width="958" height="208" rx="16" fill="url(#bg5)" stroke="#24404F"/>
          <text x="700" y="185" font-family="ui-monospace,Menlo,monospace" font-size="170" font-weight="700" fill="none" stroke="#26404F" stroke-width="2" opacity="0.5">05</text>
          <rect x="798" y="34" width="122" height="26" rx="13" fill="{INK2}" stroke="#2E5566"/>
          <text x="813" y="51" font-family="ui-monospace,Menlo,monospace" font-size="11" letter-spacing="2" fill="{KICK}">ACT 5 OF 7</text>
          <text x="40" y="54" font-family="ui-monospace,Menlo,monospace" font-size="12" letter-spacing="3.5" fill="{KICK}">THE LAYA PHARMA SURVEILLANCE SERIES</text>
          <text x="40" y="112" font-family="-apple-system,Segoe UI,sans-serif" font-size="40" font-weight="800" fill="#EEF4FB">Twenty minutes of fine-tuning</text>
          <text x="40" y="148" font-family="ui-monospace,Menlo,monospace" font-size="14.5" fill="{MIST}">212 labeled documents &#183; 3 epochs &#183; 0.26-0.58 to 0.89-0.96 held-out</text>
          <g font-family="ui-monospace,Menlo,monospace" font-size="11">
            <rect x="596" y="52" width="74" height="21" rx="6" fill="{INK2}" stroke="#2E5566"/><text x="606" y="66" fill="{KICK}">212 docs</text>
            <rect x="596" y="82" width="74" height="21" rx="6" fill="{INK2}" stroke="#2E5566"/><text x="606" y="96" fill="{KICK}">3 epochs</text>
            <rect x="596" y="112" width="74" height="21" rx="6" fill="{INK2}" stroke="#2E5566"/><text x="606" y="126" fill="{KICK}">G = 4</text>
            <rect x="596" y="142" width="74" height="21" rx="6" fill="{INK2}" stroke="#2E5566"/><text x="606" y="156" fill="{KICK}">&#963; = 0.3</text>
          </g>
          <path d="M670 62 C 710 62 726 100 755 101" stroke="#2E5566" fill="none"/>
          <path d="M670 92 C 706 92 724 101 755 101" stroke="#2E5566" fill="none"/>
          <path d="M670 122 C 706 122 724 103 755 103" stroke="#2E5566" fill="none"/>
          <path d="M670 152 C 710 152 726 104 755 103" stroke="#2E5566" fill="none"/>
          <circle cx="771" cy="102" r="17" fill="{INK}" stroke="{TEAL}" stroke-width="2"/>
          <text x="771" y="106" text-anchor="middle" font-family="ui-monospace,Menlo,monospace" font-size="10" fill="#EEF4FB">RLCD</text>
          {banner_pairs5}
          <text x="806" y="166" font-family="ui-monospace,Menlo,monospace" font-size="10" fill="#6E93A0">held-out acc, {N_TEST5} docs</text>
          <rect x="24" y="186" width="620" height="4" rx="2" fill="url(#acc5)"/>
        </svg>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def goal(N_TEST5, N_TRAIN5, mo):
    mo.md(f"""
    In this notebook, we will show what twenty minutes of fine-tuning does to
    the same checkpoint from the first four acts. No new model and no new
    infrastructure: {N_TRAIN5} labeled documents and one small training loop
    sit between the before and the after.

    By the end of the notebook, you will:

    - see the before-and-after results on {N_TEST5} held-out documents, in
      accuracy and in the probability scores underneath it, and
    - see the anatomy of the training loop that produced the jump, reward,
      group-normalized advantages, and stability anchor included.
    """)
    return


@app.cell(hide_code=True)
def context_intro(N_TEST5, N_TRAIN5, mo):
    mo.md(f"""
    ## Where the first four acts left off

    Act 4 ended on the edges: spam dressed as science, press about warning
    letters reading as warnings, and owning-team/action/urgency barely above
    coin-flip zero-shot.

    The fix is not a bigger model. It is {N_TRAIN5} labeled documents from
    your own distribution, and 20 minutes of the training loop below.

    Everything in this notebook renders from one committed file,
    `data/finetune_experiment_results.json`: before and after metrics for
    five questions, evaluated on the same {N_TEST5} held-out documents.
    Nothing here downloads a model.
    """)
    return


@app.cell(hide_code=True)
def load_data(Path, json):
    DATA5 = Path(__file__).resolve().parent.parent / "data"
    with open(
        DATA5 / "finetune_experiment_results.json", encoding="utf-8"
    ) as f5:
        R5 = json.load(f5)

    QORDER5 = [
        "doc_type",
        "secondary_type",
        "route_to",
        "action_required",
        "urgency",
    ]
    QLABELS5 = {
        "doc_type": "doc type (10-way)",
        "secondary_type": "secondary type (11-way)",
        "route_to": "owning team (8-way)",
        "action_required": "action required (yes/no)",
        "urgency": "urgency (3-way)",
    }
    SHORT5 = {
        "doc_type": "doc type",
        "secondary_type": "secondary type",
        "route_to": "route to",
        "action_required": "action req.",
        "urgency": "urgency",
    }

    N_TRAIN5 = len(R5["train_docs"])
    N_TEST5 = len(R5["test_docs"])

    ACC_BEFORE5 = [R5["before_raw"][q]["acc"] for q in QORDER5]
    ACC_AFTER5 = [R5["after_raw"][q]["acc"] for q in QORDER5]
    BRIER_BEFORE5 = [R5["before_raw"][q]["brier"] for q in QORDER5]
    BRIER_AFTER5 = [R5["after_raw"][q]["brier"] for q in QORDER5]
    LOG_BEFORE5 = [R5["before_raw"][q]["log_score"] for q in QORDER5]
    LOG_AFTER5 = [R5["after_raw"][q]["log_score"] for q in QORDER5]

    BRIER_GAIN5 = [b / a for b, a in zip(BRIER_BEFORE5, BRIER_AFTER5)]
    LOG_GAIN5 = [b / a for b, a in zip(LOG_BEFORE5, LOG_AFTER5)]
    BRIER_GAIN_LO5 = min(BRIER_GAIN5)
    BRIER_GAIN_HI5 = max(BRIER_GAIN5)
    LOG_GAIN_LO5 = min(LOG_GAIN5)
    LOG_GAIN_HI5 = max(LOG_GAIN5)

    SEC_BEFORE5 = R5["before_raw"]["secondary_type"]["acc"]
    SEC_AFTER5 = R5["after_raw"]["secondary_type"]["acc"]
    ROUTE_BEFORE5 = R5["before_raw"]["route_to"]["acc"]
    ROUTE_AFTER5 = R5["after_raw"]["route_to"]["acc"]
    QLABEL_JOINED5 = ", ".join(QLABELS5[q] for q in QORDER5)
    return (
        ACC_AFTER5,
        ACC_BEFORE5,
        BRIER_AFTER5,
        BRIER_BEFORE5,
        BRIER_GAIN_HI5,
        BRIER_GAIN_LO5,
        LOG_AFTER5,
        LOG_BEFORE5,
        LOG_GAIN_HI5,
        LOG_GAIN_LO5,
        N_TEST5,
        N_TRAIN5,
        QLABELS5,
        QLABEL_JOINED5,
        QORDER5,
        R5,
        ROUTE_AFTER5,
        ROUTE_BEFORE5,
        SEC_AFTER5,
        SEC_BEFORE5,
        SHORT5,
    )


@app.cell(hide_code=True)
def data_reading(N_TEST5, N_TRAIN5, QLABEL_JOINED5, mo):
    mo.md(f"""
        The file holds the split ({N_TRAIN5} training documents, {N_TEST5} test
        documents) and before/after metrics for five questions:
        {QLABEL_JOINED5}. The "before" rows are the base checkpoint from Acts
        1-4; the "after" rows are the same checkpoint after RLCD fine-tuning.
        Every number in the prose below is computed from this file, so the text
        cannot drift away from the charts.
        """)
    return


@app.cell(hide_code=True)
def heldout_intro(N_TEST5, mo):
    mo.md(f"""
    ## Held-out results, before and after

    {N_TEST5} test documents the model never trained on. Same schema, same
    checkpoint, one fine-tuning run in between. The table shows accuracy,
    Brier, and mean log score side by side; the charts after it plot the
    same numbers as bars.
    """)
    return


@app.cell(hide_code=True)
def before_after_table(QLABELS5, QORDER5, R5, mo):
    table5 = [
        {
            "question": QLABELS5[q],
            "accuracy before": round(R5["before_raw"][q]["acc"], 3),
            "accuracy after": round(R5["after_raw"][q]["acc"], 3),
            "Brier before": round(R5["before_raw"][q]["brier"], 3),
            "Brier after": round(R5["after_raw"][q]["brier"], 3),
            "log score before": round(R5["before_raw"][q]["log_score"], 3),
            "log score after": round(R5["after_raw"][q]["log_score"], 3),
        }
        for q in QORDER5
    ]
    mo.ui.table(table5, selection=None)
    return


@app.cell(hide_code=True)
def table_reading(ACC_AFTER5, ACC_BEFORE5, SEC_BEFORE5, mo):
    mo.md(f"""
        Read the table column by column. Before fine-tuning, accuracy runs from
        {min(ACC_BEFORE5):.2f} to {max(ACC_BEFORE5):.2f} across the five
        questions, and secondary type sits at {SEC_BEFORE5:.2f}, close to random
        for an 11-way pick. After fine-tuning, every question lands between
        {min(ACC_AFTER5):.2f} and {max(ACC_AFTER5):.2f}. The probability columns
        move in the same direction on every question, which is what the next two
        charts show.
        """)
    return


@app.cell(hide_code=True)
def acc_plot_intro(mo):
    mo.md(f"""
    ## Accuracy: the headline

    Here is the accuracy comparison as a chart. Slate bars are the base
    checkpoint, green bars are after RLCD fine-tuning, and every bar carries
    its own value label.
    """)
    return


@app.cell(hide_code=True)
def acc_plot(
    ACC_AFTER5,
    ACC_BEFORE5,
    GREEN,
    N_TEST5,
    QLABELS5,
    QORDER5,
    SLATE,
    go,
):
    fig_acc5 = go.Figure()
    fig_acc5.add_trace(
        go.Bar(
            x=[QLABELS5[q] for q in QORDER5],
            y=ACC_BEFORE5,
            name="before",
            marker_color=SLATE,
            text=[f"{v:.2f}" for v in ACC_BEFORE5],
            textposition="outside",
        )
    )
    fig_acc5.add_trace(
        go.Bar(
            x=[QLABELS5[q] for q in QORDER5],
            y=ACC_AFTER5,
            name="after RLCD",
            marker_color=GREEN,
            text=[f"{v:.2f}" for v in ACC_AFTER5],
            textposition="outside",
        )
    )
    fig_acc5.update_layout(
        barmode="group",
        template="plotly_white",
        title=dict(
            text=f"Held-out accuracy on {N_TEST5} unseen documents",
            font=dict(size=13),
        ),
        height=380,
        margin=dict(l=10, r=30, t=60, b=40),
        yaxis=dict(range=[0, 1.09], title="accuracy"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig_acc5
    return


@app.cell(hide_code=True)
def acc_plot_reading(
    ACC_AFTER5,
    N_TEST5,
    ROUTE_AFTER5,
    ROUTE_BEFORE5,
    SEC_AFTER5,
    SEC_BEFORE5,
    mo,
):
    mo.md(f"""
        Every question moved into the {min(ACC_AFTER5):.2f}-{max(ACC_AFTER5):.2f} band. The two weakest zero-shot
        questions improved the most: secondary type went from {SEC_BEFORE5:.2f}
        to {SEC_AFTER5:.2f}, and route-to went from {ROUTE_BEFORE5:.2f} to
        {ROUTE_AFTER5:.2f}.

        One caveat keeps this honest: accuracy on {N_TEST5} documents is grainy,
        and each document moves it by about {1 / N_TEST5:.3f}. That graininess is
        exactly why the probability scores below carry the real story.
        """)
    return


@app.cell(hide_code=True)
def prob_scores_intro(mo):
    mo.md("""
    ## The probability scores underneath

    Accuracy only checks the top option. Brier and mean log score grade the
    whole distribution: they reward putting most of the probability mass on
    the right answer, and they punish confident wrong answers. Here are both,
    before and after.
    """)
    return


@app.cell(hide_code=True)
def prob_scores_plot(
    BRIER_AFTER5,
    BRIER_BEFORE5,
    GREEN,
    LOG_AFTER5,
    LOG_BEFORE5,
    QORDER5,
    SHORT5,
    SLATE,
    go,
    make_subplots,
):
    fig_scores5 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "mean log score (higher is better)",
            "Brier score (lower is better)",
        ),
    )
    fig_scores5.add_trace(
        go.Bar(
            x=[SHORT5[q] for q in QORDER5],
            y=LOG_BEFORE5,
            name="before",
            marker_color=SLATE,
            text=[f"{v:.2f}" for v in LOG_BEFORE5],
            textposition="outside",
        ),
        row=1,
        col=1,
    )
    fig_scores5.add_trace(
        go.Bar(
            x=[SHORT5[q] for q in QORDER5],
            y=LOG_AFTER5,
            name="after RLCD",
            marker_color=GREEN,
            text=[f"{v:.2f}" for v in LOG_AFTER5],
            textposition="outside",
        ),
        row=1,
        col=1,
    )
    fig_scores5.add_trace(
        go.Bar(
            x=[SHORT5[q] for q in QORDER5],
            y=BRIER_BEFORE5,
            name="before",
            marker_color=SLATE,
            text=[f"{v:.2f}" for v in BRIER_BEFORE5],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig_scores5.add_trace(
        go.Bar(
            x=[SHORT5[q] for q in QORDER5],
            y=BRIER_AFTER5,
            name="after RLCD",
            marker_color=GREEN,
            text=[f"{v:.2f}" for v in BRIER_AFTER5],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig_scores5.update_layout(
        barmode="group",
        template="plotly_white",
        height=400,
        margin=dict(l=10, r=30, t=70, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0),
    )
    fig_scores5.update_yaxes(range=[0, None], row=1, col=1)
    fig_scores5.update_yaxes(range=[0, None], row=1, col=2)
    fig_scores5.update_xaxes(tickangle=-18)
    fig_scores5
    return


@app.cell(hide_code=True)
def prob_scores_reading(
    BRIER_GAIN_HI5,
    BRIER_GAIN_LO5,
    LOG_GAIN_HI5,
    LOG_GAIN_LO5,
    N_TEST5,
    mo,
):
    mo.md(f"""
    Accuracy is the headline; the probability metrics are the story. Log
    score and Brier measure the quality of the whole distribution, and both
    improved by large factors: Brier dropped {BRIER_GAIN_LO5:.1f}x to
    {BRIER_GAIN_HI5:.1f}x across the five questions, and mean log score
    {LOG_GAIN_LO5:.1f}x to {LOG_GAIN_HI5:.1f}x. A model can luck into argmax
    accuracy. It cannot luck into a distribution this much sharper across
    {N_TEST5} unseen documents.
    """)
    return


@app.cell(hide_code=True)
def loop_intro(N_TRAIN5, mo):
    mo.md(f"""
    ## The loop, all 20 minutes of it

    The gains above came from one small training loop, reimplemented at
    laptop scale on top of Laya's own training utilities: group-relative
    policy gradients with Gaussian logit noise, a strictly proper scoring
    reward, and a full encoder fine-tune. Three epochs over {N_TRAIN5}
    documents. Here is the anatomy.
    """)
    return


@app.cell(hide_code=True)
def loop_anatomy(N_TRAIN5, mo):
    mo.md(f"""
    ```python
    EPOCHS = 3        # over {N_TRAIN5} labeled documents
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
    """)
    return


@app.cell(hide_code=True)
def reward_reading(mo):
    mo.md("""
    The reward is the whole trick. Proper scoring rules have one property
    that matters: the only way to maximize expected reward is to report
    honest probabilities. Fine-tune against that reward and calibration
    falls out as a side effect of the objective, which is exactly what Act 6
    measures.
    """)
    return


@app.cell(hide_code=True)
def price_of_lever(N_TRAIN5, mo):
    mo.md(f"""
    ## The price of the lever

    {N_TRAIN5} hand-labeled documents. 20 minutes on an M4 Max laptop, or a
    free Kaggle GPU. No infrastructure, no API bills, and the checkpoint
    stays yours. The lever is {N_TRAIN5} labeled documents and 20 minutes.
    """)
    return


@app.cell(hide_code=True)
def next_act(mo):
    mo.md(f"""
    **Act 6:** accuracy was the easy part. The question that decides whether
    you can wire a P = 0.9 threshold to an automated action is what the
    probabilities actually mean. That is a calibration question, and it is
    the whole subject of notebook 6.
    """)
    return


if __name__ == "__main__":
    app.run()
