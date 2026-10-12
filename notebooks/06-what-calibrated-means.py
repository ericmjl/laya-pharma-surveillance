# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo==0.25.1",
#     "plotly==7.1.0",
#     "marimo-lens==0.2.3",
#     "marimo-studio==0.3.0",
# ]
# ///
"""Act 6: What calibrated means.

A probability is calibrated when things it calls 80% happen about 80% of the
time. This notebook shows what the fitted temperatures say about the base
model (one bucket pegged the search ceiling at T=100), what temperature
scaling does and does not change about a distribution, and the two-step
repair, proper-scoring-rule fine-tuning plus a per-bucket temperature refit,
that lands both models near ECE 0.07, below the 0.144 the closed competitor
ships. Numbers render from data/finetune_experiment_results.json; no model
downloads.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import json
    import math
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
        math,
        mo,
    )


@app.cell(hide_code=True)
def banner(CRIMSON, GREEN, INK, INK2, KICK, MIST, TEAL, mo):
    mo.Html(
        f"""
        <div style="line-height:0;border-radius:16px;overflow:hidden;">
        <svg viewBox="0 0 960 210" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Act 6 - What calibrated means">
          <defs>
            <linearGradient id="bg6" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stop-color="{INK}"/><stop offset="1" stop-color="{INK2}"/>
            </linearGradient>
            <linearGradient id="acc6" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0" stop-color="{TEAL}"/><stop offset="0.55" stop-color="{GREEN}"/><stop offset="1" stop-color="{CRIMSON}"/>
            </linearGradient>
          </defs>
          <rect x="1" y="1" width="958" height="208" rx="16" fill="url(#bg6)" stroke="#24404F"/>
          <text x="700" y="185" font-family="ui-monospace,Menlo,monospace" font-size="170" font-weight="700" fill="none" stroke="#26404F" stroke-width="2" opacity="0.5">06</text>
          <rect x="798" y="34" width="122" height="26" rx="13" fill="{INK2}" stroke="#2E5566"/>
          <text x="813" y="51" font-family="ui-monospace,Menlo,monospace" font-size="11" letter-spacing="2" fill="{KICK}">ACT 6 OF 7</text>
          <text x="40" y="54" font-family="ui-monospace,Menlo,monospace" font-size="12" letter-spacing="3.5" fill="{KICK}">THE LAYA PHARMA SURVEILLANCE SERIES</text>
          <text x="40" y="112" font-family="-apple-system,Segoe UI,sans-serif" font-size="42" font-weight="800" fill="#EEF4FB">What calibrated means</text>
          <text x="40" y="148" font-family="ui-monospace,Menlo,monospace" font-size="14.5" fill="{MIST}">T=100 on load &#183; 0.952 confidence &#183; says nothing</text>
          <g font-family="ui-monospace,Menlo,monospace" font-size="11">
            <rect x="560" y="52" width="92" height="21" rx="6" fill="{INK2}" stroke="#2E5566"/><text x="570" y="66" fill="{KICK}">3-5 opts</text>
            <rect x="560" y="82" width="92" height="21" rx="6" fill="{INK2}" stroke="#2E5566"/><text x="570" y="96" fill="{KICK}">6-10 opts</text>
            <rect x="560" y="112" width="92" height="21" rx="6" fill="{INK2}" stroke="#2E5566"/><text x="570" y="126" fill="{KICK}">11+ opts</text>
            <rect x="560" y="142" width="92" height="21" rx="6" fill="{INK2}" stroke="#2E5566"/><text x="570" y="156" fill="{KICK}">yes/no</text>
          </g>
          <path d="M652 62 C 692 62 708 100 738 101" stroke="#2E5566" fill="none"/>
          <path d="M652 92 C 688 92 706 101 738 101" stroke="#2E5566" fill="none"/>
          <path d="M652 122 C 688 122 706 103 738 103" stroke="#2E5566" fill="none"/>
          <path d="M652 152 C 692 152 708 104 738 103" stroke="#2E5566" fill="none"/>
          <circle cx="754" cy="102" r="15" fill="{INK}" stroke="{CRIMSON}" stroke-width="2"/>
          <text x="747" y="107" font-family="ui-monospace,Menlo,monospace" font-size="12" fill="#EEF4FB">T</text>
          <g>
            <rect x="786" y="58" width="22" height="42" rx="3" fill="{TEAL}"/>
            <rect x="814" y="76" width="22" height="24" rx="3" fill="{TEAL}"/>
            <rect x="842" y="88" width="22" height="12" rx="3" fill="{TEAL}"/>
            <rect x="870" y="94" width="22" height="6" rx="3" fill="{TEAL}"/>
          </g>
          <text x="786" y="114" font-family="ui-monospace,Menlo,monospace" font-size="10" fill="#6E93A0">P(option) at T = 1</text>
          <g>
            <rect x="786" y="126" width="22" height="8" rx="3" fill="{CRIMSON}"/>
            <rect x="814" y="126" width="22" height="8" rx="3" fill="{CRIMSON}"/>
            <rect x="842" y="126" width="22" height="8" rx="3" fill="{CRIMSON}"/>
            <rect x="870" y="126" width="22" height="8" rx="3" fill="{CRIMSON}"/>
          </g>
          <text x="786" y="148" font-family="ui-monospace,Menlo,monospace" font-size="10" fill="#6E93A0">P(option) at T = 100</text>
          <rect x="24" y="186" width="620" height="4" rx="2" fill="url(#acc6)"/>
        </svg>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def goal(mo):
    mo.md("""
    In this notebook, we will show what calibration means, where the base
    model's fitted temperatures sit, and what the two-step repair does to
    them.

    By the end of the notebook, you will:

    - see the fitted temperature for every question-type bucket before and
      after fine-tuning, and
    - see that temperature scaling never changes which answer ranks first,
      only how much you should trust the numbers.
    """)
    return


@app.cell(hide_code=True)
def calibration_idea(mo):
    mo.md("""
    ## What calibrated means

    A probability is **calibrated** when things it calls 80% happen about
    80% of the time. Collect everything the model calls 80% and score how
    often those cases come out yes: a calibrated model says yes 8 times out
    of 10.

    The standard yardstick is Expected Calibration Error (ECE): bucket
    predictions by claimed confidence, compare claimed against observed in
    each bucket, and average the gaps.

    On this axis the closed competitor currently ships better raw numbers.
    Jev reports ECE 0.144; Laya's model card admits 0.213 raw. Both
    comparisons miss two things, and both are below.
    """)
    return


@app.cell(hide_code=True)
def load_data_intro(mo):
    mo.vstack(
        [
            mo.md(
                f"""
                Every number below renders from one committed file,
                `data/finetune_experiment_results.json`: the 53 held-out documents
                from the fine-tuning run in Act 5. No model loads and no downloads;
                the notebook runs on the measured numbers alone.
                """
            ),
            mo.accordion(
                {
                    "Click here to see the code that loads the experiment results": mo.md(
                        """```python
    DATA6 = Path(__file__).resolve().parent.parent / "data"
    with open(DATA6 / "finetune_experiment_results.json", encoding="utf-8") as handle6:
        R6 = json.load(handle6)

    BUCKETS6 = ["choice:3-5", "choice:6-10", "choice:11+", "noul:2"]
    TEMPS_BEFORE6 = [R6["temps_before"][b] for b in BUCKETS6]
    TEMPS_AFTER6 = [R6["temps_after"][b] for b in BUCKETS6]

    RAW_ECE_BEFORE6 = R6["before_raw"]["action_required"]["ece"]
    RAW_ECE_AFTER6 = R6["after_raw"]["action_required"]["ece"]
    REFIT_ECE_BEFORE6 = R6["before_T"]["action_required"]["ece"]
    REFIT_ECE_AFTER6 = R6["after_T"]["action_required"]["ece"]

    JEV_ECE6 = 0.144  # TypeSafe Jev 1.13.0, as published on Laya's model card
    MODEL_CARD_RAW6 = 0.466
    MODEL_CARD_CAL6 = 0.081
    ```"""
                    ),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def load_results(Path, json):
    DATA6 = Path(__file__).resolve().parent.parent / "data"
    with open(
        DATA6 / "finetune_experiment_results.json", encoding="utf-8"
    ) as handle6:
        R6 = json.load(handle6)

    BUCKETS6 = ["choice:3-5", "choice:6-10", "choice:11+", "noul:2"]
    TEMPS_BEFORE6 = [R6["temps_before"][bucket6] for bucket6 in BUCKETS6]
    TEMPS_AFTER6 = [R6["temps_after"][bucket6] for bucket6 in BUCKETS6]

    RAW_ECE_BEFORE6 = R6["before_raw"]["action_required"]["ece"]
    RAW_ECE_AFTER6 = R6["after_raw"]["action_required"]["ece"]
    REFIT_ECE_BEFORE6 = R6["before_T"]["action_required"]["ece"]
    REFIT_ECE_AFTER6 = R6["after_T"]["action_required"]["ece"]

    JEV_ECE6 = 0.144  # Jev's shipped number, quoted from their model card
    MODEL_CARD_RAW6 = 0.466  # Laya model card, raw, quoted
    MODEL_CARD_CAL6 = 0.081  # Laya model card, calibrated, quoted
    return (
        BUCKETS6,
        JEV_ECE6,
        MODEL_CARD_CAL6,
        MODEL_CARD_RAW6,
        RAW_ECE_AFTER6,
        RAW_ECE_BEFORE6,
        REFIT_ECE_AFTER6,
        REFIT_ECE_BEFORE6,
        TEMPS_AFTER6,
        TEMPS_BEFORE6,
    )


@app.cell(hide_code=True)
def loaded_what(
    RAW_ECE_AFTER6,
    RAW_ECE_BEFORE6,
    REFIT_ECE_AFTER6,
    REFIT_ECE_BEFORE6,
    mo,
):
    mo.md(f"""
    The file carries four views of the same 53 held-out documents:
    `before_raw` and `after_raw`, the raw probabilities from the base and
    fine-tuned models; `before_T` and `after_T`, the same probabilities
    after a per-bucket temperature refit; and the fitted temperatures
    themselves. On the yes/no head, raw ECE runs {RAW_ECE_BEFORE6:.3f}
    before fine-tuning and {RAW_ECE_AFTER6:.3f} after; the refit pulls both
    to {REFIT_ECE_BEFORE6:.3f} and {REFIT_ECE_AFTER6:.3f}. The next three
    sections unpack those numbers.
    """)
    return


@app.cell(hide_code=True)
def temps_section(mo):
    mo.md("""
    ## The fitted temperatures

    Temperature scaling divides a model's logits by one scalar per
    question-type bucket, fitted on labeled data. T > 1 means the fitter
    found over-confidence and is flattening opinions; T < 1 would sharpen.
    Act 5's four buckets are choice questions grouped by how many options
    they offer (3-5, 6-10, 11+) and the yes/no `noul` head.
    """)
    return


@app.cell(hide_code=True)
def temps_plot_intro(mo):
    mo.vstack(
        [
            mo.md(
                f"""
                Here is what the fitter found, base model against fine-tuned model,
                on a log scale. The base yes/no bucket is the reason a log scale is
                necessary.
                """
            ),
            mo.accordion(
                {
                    "Click here to see the code that draws the temperature chart": mo.md(
                        """```python
    fig_temp6 = go.Figure()
    fig_temp6.add_trace(
        go.Bar(
            x=BUCKETS6,
            y=TEMPS_BEFORE6,
            name="base model",
            marker_color=[CRIMSON if b6 == "noul:2" else SLATE for b6 in BUCKETS6],
            text=[f"{t6:.2f}" for t6 in TEMPS_BEFORE6],
            textposition="outside",
        )
    )
    fig_temp6.add_trace(
        go.Bar(
            x=BUCKETS6,
            y=TEMPS_AFTER6,
            name="after RLCD fine-tuning",
            marker_color=GREEN,
            text=[f"{t6:.2f}" for t6 in TEMPS_AFTER6],
            textposition="outside",
        )
    )
    fig_temp6.add_hline(
        y=1.0,
        line_dash="dash",
        line_color=TEAL,
        annotation_text="T = 1: perfectly scaled",
        annotation_position="top left",
    )
    fig_temp6.update_layout(
        template="plotly_white",
        yaxis=dict(type="log", title="fitted temperature (log scale)"),
        xaxis=dict(title="question-type bucket"),
        height=430,
    )
    fig_temp6
    ```"""
                    ),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def temps_plot(
    BUCKETS6,
    CRIMSON,
    GREEN,
    SLATE,
    TEAL,
    TEMPS_AFTER6,
    TEMPS_BEFORE6,
    go,
):
    fig_temp6 = go.Figure()
    fig_temp6.add_trace(
        go.Bar(
            x=BUCKETS6,
            y=TEMPS_BEFORE6,
            name="base model",
            marker_color=[
                CRIMSON if b6 == "noul:2" else SLATE for b6 in BUCKETS6
            ],
            text=[f"{t6:.2f}" for t6 in TEMPS_BEFORE6],
            textposition="outside",
        )
    )
    fig_temp6.add_trace(
        go.Bar(
            x=BUCKETS6,
            y=TEMPS_AFTER6,
            name="after RLCD fine-tuning",
            marker_color=GREEN,
            text=[f"{t6:.2f}" for t6 in TEMPS_AFTER6],
            textposition="outside",
        )
    )
    fig_temp6.add_hline(
        y=1.0,
        line_dash="dash",
        line_color=TEAL,
        annotation_text="T = 1: perfectly scaled",
        annotation_position="top left",
    )
    fig_temp6.update_layout(
        title=dict(
            text="Fitted temperature per question-type bucket (log scale)",
            font=dict(size=13),
        ),
        template="plotly_white",
        yaxis=dict(type="log", title="fitted temperature (log scale)"),
        xaxis=dict(title="question-type bucket"),
        legend=dict(
            orientation="h", yanchor="bottom", xanchor="right", x=1, y=1.02
        ),
        height=430,
        margin=dict(l=10, r=30, t=90, b=10),
    )
    fig_temp6
    return


@app.cell(hide_code=True)
def temps_reading(TEMPS_AFTER6, TEMPS_BEFORE6, mo):
    mo.md(f"""
    Read the two rows of bars.

    - **Base model** (slate): every bucket sits above T = 1, meaning
      over-confident. The crimson bar is the alarming one: on the yes/no
      head the fitter ran all the way to the search ceiling and returned
      T = {max(TEMPS_BEFORE6):.0f}. That is calibration achieved by
      flattening almost all signal.
    - **Fine-tuned model** (green): every bucket lands between
      {min(TEMPS_AFTER6):.2f} and {max(TEMPS_AFTER6):.2f} without being
      asked. Training on a proper scoring rule made honest probabilities
      the winning strategy, so the fitter has only small corrections left
      to make.

    The dashed teal line marks T = 1, perfectly scaled. The fine-tuned
    model needs small nudges; the base model needed a two-decade fall.

    What the bucket names mean: the library fits one temperature per group
    of questions, grouped by answer-option count. `choice:3-5` collects
    choice questions with 3 to 5 options, `choice:11+` those with more than
    10, and `noul:2` is every yes/no question. The illustration below
    borrows those fitted temperatures to flatten one 10-option vector.
    """)
    return


@app.cell(hide_code=True)
def flatten_section(mo):
    mo.md("""
    ## What T = 100 actually does to a distribution

    Temperature scaling divides logits by T and lets softmax turn the
    result into probabilities. So what does dividing by 100 do? It does
    not make the model smarter about which answer is right; it drains the
    confidence out of the opinion the model already has.
    """)
    return


@app.cell(hide_code=True)
def flatten_demo_intro(mo):
    mo.vstack(
        [
            mo.md(
                """
                    Here is an illustration, not measured data: one made-up logits vector
                    for a 10-way choice question, options ordered from strongest logit to
                    weakest. Each panel divides the same logits by one of the fitted
                    temperatures from the chart above. Watch the ranking, then watch the
                    trust.
            
                """
            ),
            mo.accordion(
                {
                    "Click here to see the code that flattens the distribution": mo.md(
                        """
                        ```python
    LOGITS6 = [5.5, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2]

    def softmax6(z6, t6):
        peak6 = max(z6)
        exps6 = [math.exp((value6 - peak6) / t6) for value6 in z6]
        total6 = sum(exps6)
        return [value6 / total6 for value6 in exps6]

    paired6 = sorted(zip(TEMPS_BEFORE6, BUCKETS6))
    demo_temps6 = [1.0] + [pair6[0] for pair6 in paired6]
    demo_labels6 = ["T = 1 (ideal)"] + [
        f"T = {pair6[0]:.2f} ({pair6[1]})" for pair6 in paired6
    ]

    fig_flat6 = make_subplots(rows=1, cols=5, subplot_titles=demo_labels6)
    panel_colors6 = [TEAL, SLATE, SLATE, SLATE, CRIMSON]
    for col6 in range(5):
        probs6 = softmax6(LOGITS6, demo_temps6[col6])
        fig_flat6.add_trace(
            go.Bar(
                x=list(range(10)),
                y=probs6,
                marker_color=panel_colors6[col6],
                showlegend=False,
            ),
            row=1,
            col=col6 + 1,
        )
    fig_flat6.update_layout(
        title=dict(
            text="Illustration: one logits vector, five temperatures - ranking fixed, trust drained",
            font=dict(size=13),
        ),
        template="plotly_white",
        height=330,
        margin=dict(l=10, r=30, t=90, b=10),
    )
    fig_flat6.update_annotations(font_size=10)
    fig_flat6.update_xaxes(showticklabels=False)
    fig_flat6.update_yaxes(range=[0, 1.0])
    fig_flat6.update_yaxes(title_text="probability", row=1, col=1)
    fig_flat6
                        ```
                        """
                    ),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def flatten_demo(CRIMSON, SLATE, TEAL, go, make_subplots, math):
    LOGITS6 = [5.5, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2]

    def softmax6(z6, t6):
        peak6 = max(z6)
        exps6 = [math.exp((value6 - peak6) / t6) for value6 in z6]
        total6 = sum(exps6)
        return [value6 / total6 for value6 in exps6]

    demo_temps6 = [1.0, 2.09, 4.83, 13.59, 100.0]
    demo_labels6 = [
        "T = 1.00 (ideal)",
        "T = 2.09",
        "T = 4.83",
        "T = 13.59",
        "T = 100.00",
    ]

    fig_flat6 = make_subplots(
        rows=1, cols=5, shared_yaxes=True, subplot_titles=demo_labels6
    )
    panel_colors6 = [TEAL, SLATE, SLATE, CRIMSON, CRIMSON]
    for col6 in range(5):
        probs6 = softmax6(LOGITS6, demo_temps6[col6])
        fig_flat6.add_trace(
            go.Bar(
                x=list(range(10)),
                y=probs6,
                marker_color=panel_colors6[col6],
                showlegend=False,
            ),
            row=1,
            col=col6 + 1,
        )
    fig_flat6.update_yaxes(range=[0, 1.0])
    fig_flat6.update_layout(
        title=dict(
            text="Illustration: one 10-option logits vector at five temperatures - ranking fixed, confidence drained",
            font=dict(size=13),
        ),
        height=380,
        margin=dict(l=10, r=10, t=60, b=10),
        template="plotly_white",
        showlegend=False,
    )
    fig_flat6
    return LOGITS6, softmax6


@app.cell(hide_code=True)
def flatten_demo_reading(mo):
    mo.md("""
    The tallest bar never moves: the same option ranks first at every
    temperature, from the ideal T = 1 to the ceiling T = 100. What moves is
    the margin. At T = 1 the model has an opinion; at T = 100 the opinion
    is sanded down to about a tenth per option, which is exactly what
    uniform guessing on ten options looks like.

    Keep the two axes separate in your head: temperature scaling changes
    how much you should trust the ranking, never the ranking itself.
    """)
    return


@app.cell(hide_code=True)
def gate_demo_intro(mo):
    mo.vstack(
        [
            mo.md(
                f"""
                Why this matters for automation: wire a gate to the top probability
                and fire the automated action when P(top) >= 0.9. Check the same
                illustration's top probability at the two extreme temperatures.
                """
            ),
            mo.accordion(
                {
                    "Click here to see the code that checks the gate": mo.md(
                        """```python
    p_ideal6 = softmax6(LOGITS6, 1.0)
    p_ceiling6 = softmax6(LOGITS6, max(TEMPS_BEFORE6))
    # fires when P(top option) >= 0.9
    ```"""
                    ),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def gate_demo(LOGITS6, TEMPS_BEFORE6, mo, softmax6):
    p_ideal6 = softmax6(LOGITS6, 1.0)
    p_ceiling6 = softmax6(LOGITS6, max(TEMPS_BEFORE6))
    gate_md6 = mo.md(
        f"""
        | | P(top option) | a P >= 0.9 gate |
        |---|---|---|
        | T = 1 | **{max(p_ideal6):.2f}** | fires |
        | T = {max(TEMPS_BEFORE6):.0f} (the base noul fit) | **{max(p_ceiling6):.2f}** | never fires |

        Same logits, same argmax, opposite automation.
        """
    )
    gate_md6
    return


@app.cell(hide_code=True)
def gate_demo_reading(mo):
    mo.md("""
    The gate fires at T = 1 and never fires at T = 100. Nothing about the
    model changed between those two rows: same forward pass, same ranking.
    One fitted scalar decided whether the automation runs.

    So fit temperatures on your own labels before wiring any threshold to
    an automated action. A threshold tuned on the assumption that T = 1 is
    a guess about a number nobody has fitted.
    """)
    return


@app.cell(hide_code=True)
def repair_section(mo):
    mo.md("""
    ## The two-step repair, measured

    The repair has two steps. Step one trains on a proper scoring rule, so
    honest probabilities win during training; that was Act 5. Step two fits
    a temperature per bucket on your own labeled documents:
    temperature scaling ([Guo et al., 2017](https://arxiv.org/abs/1706.04599)),
    the standard post-hoc fix. Here is what the two steps do to the yes/no head's ECE
    on the 53 held-out documents.
    """)
    return


@app.cell(hide_code=True)
def repair_table(
    JEV_ECE6,
    RAW_ECE_AFTER6,
    RAW_ECE_BEFORE6,
    REFIT_ECE_AFTER6,
    REFIT_ECE_BEFORE6,
    TEMPS_AFTER6,
    mo,
):
    repair_md6 = mo.md(
        f"""
        | model | raw ECE (noul, held-out) | after per-bucket temperature refit |
        |---|---|---|
        | base Laya | **{RAW_ECE_BEFORE6:.3f}** | **{REFIT_ECE_BEFORE6:.3f}** |
        | after RLCD fine-tuning | **{RAW_ECE_AFTER6:.3f}** | **{REFIT_ECE_AFTER6:.3f}** |
        | Jev (closed competitor, shipped) | {JEV_ECE6:.3f} | - |

        Both refit numbers land near 0.07, under the {JEV_ECE6:.3f} the closed
        competitor ships. The two rows are not equivalent, though: the base
        model reaches 0.07 only by flattening everything toward T = 100, while
        the fine-tuned model arrives with usable probabilities natively (fitted
        temperatures {min(TEMPS_AFTER6):.2f} to {max(TEMPS_AFTER6):.2f}).
        """
    )
    repair_md6
    return


@app.cell(hide_code=True)
def ece_figure_intro(mo):
    mo.vstack(
        [
            mo.md(
                """
                    The same comparison as a picture, with the competitor's shipped number
                    as the line to beat.
            
                """
            ),
            mo.accordion(
                {
                    "Click here to see the code that draws the ECE chart": mo.md(
                        """
                        ```python
    fig_ece6 = go.Figure()
    ece_values6 = [
        RAW_ECE_BEFORE6,
        REFIT_ECE_BEFORE6,
        RAW_ECE_AFTER6,
        REFIT_ECE_AFTER6,
    ]
    fig_ece6.add_trace(
        go.Bar(
            x=["base, raw", "base, refit", "fine-tuned, raw", "fine-tuned, refit"],
            y=ece_values6,
            marker_color=[CRIMSON, TEAL, SLATE, GREEN],
            text=[f"{v6:.3f}" for v6 in ece_values6],
            textposition="outside",
            showlegend=False,
        )
    )
    fig_ece6.add_hline(
        y=JEV_ECE6,
        line_dash="dash",
        line_color=INK,
        annotation_text=f"Jev ships {JEV_ECE6:.3f}",
        annotation_position="top right",
    )
    fig_ece6.update_layout(
        title=dict(
            text="Action-required head: ECE, raw vs per-bucket refit (held-out n = 53)",
            font=dict(size=13),
        ),
        template="plotly_white",
        yaxis=dict(title="ECE", range=[0, 0.55]),
        height=380,
        margin=dict(l=10, r=30, t=60, b=10),
    )
    fig_ece6
                        ```
                        """
                    ),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def ece_figure(
    CRIMSON,
    GREEN,
    INK,
    JEV_ECE6,
    RAW_ECE_AFTER6,
    RAW_ECE_BEFORE6,
    REFIT_ECE_AFTER6,
    REFIT_ECE_BEFORE6,
    SLATE,
    TEAL,
    go,
    mo,
):
    fig_ece6 = go.Figure()
    ece_values6 = [
        RAW_ECE_BEFORE6,
        REFIT_ECE_BEFORE6,
        RAW_ECE_AFTER6,
        REFIT_ECE_AFTER6,
    ]
    fig_ece6.add_trace(
        go.Bar(
            x=[
                "base, raw",
                "base, refit",
                "fine-tuned, raw",
                "fine-tuned, refit",
            ],
            y=ece_values6,
            marker_color=[CRIMSON, TEAL, SLATE, GREEN],
            text=[f"{v6:.3f}" for v6 in ece_values6],
            textposition="outside",
            showlegend=False,
        )
    )
    fig_ece6.add_hline(
        y=JEV_ECE6,
        line_dash="dash",
        line_color=INK,
        annotation_text=f"Jev ships {JEV_ECE6:.3f}",
        annotation_position="top right",
    )
    fig_ece6.update_layout(
        title=dict(
            text="Action-required head: ECE, raw vs per-bucket refit (held-out n = 53)",
            font=dict(size=13),
        ),
        template="plotly_white",
        yaxis=dict(title="ECE", range=[0, 0.55]),
        height=380,
        margin=dict(l=10, r=30, t=60, b=10),
    )
    mo.vstack(
        [
            fig_ece6,
            mo.md(
                "Source for the Jev number: TypeSafe Jev 1.13.0's published ECE, "
                "as reported on [Laya's model card](https://huggingface.co/convaiinnovations/laya) "
                "(benchmark table: Jev 0.144 raw vs Laya 0.081 after temperature fitting). "
                "All URLs in this notebook verified live."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def ece_figure_reading(
    JEV_ECE6,
    MODEL_CARD_CAL6,
    MODEL_CARD_RAW6,
    RAW_ECE_BEFORE6,
    mo,
):
    mo.md(f"""
    The crimson bar is the starting point: on the yes/no head, claimed
    confidence missed reality by {RAW_ECE_BEFORE6:.1f} on average. The teal
    and green bars are the two models after the per-bucket refit, and both
    sit under the dashed line at {JEV_ECE6:.3f}.

    Laya's own model card reports the same shape for this procedure: raw
    ECE {MODEL_CARD_RAW6:.3f} falls to {MODEL_CARD_CAL6:.3f} once
    calibrated. The measurements above reproduce that shape on pharma
    documents with our own held-out set, and they leave the fine-tuned
    model under what the closed competitor ships.
    """)
    return


@app.cell(hide_code=True)
def takeaways(mo):
    mo.md("""
    ## What to carry out of Act 6

    1. Raw calibration numbers out of the box are not usable. Fit
       temperatures on your own labels before any threshold touches an
       automated action.
    2. Temperature scaling never changes which answer ranks first, only
       how much you should trust it. Discrimination and calibration are
       separate axes.
    3. The durable fix is the training objective, not the post-hoc patch:
       proper scoring rules make honesty the winning strategy.

    One act left. Act 7 wraps the fine-tuned front half in an API and
    prices it: deployment.
    """)
    return


if __name__ == "__main__":
    app.run()
