# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo==0.25.1",
#     "laya==0.3.5",
#     "plotly==7.1.0",
#     "marimo-lens==0.2.3",
#     "marimo-studio==0.3.0",
#     "anywidget==0.11.0",
#     "traitlets==5.16.1",
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

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def banner(CRIMSON, GREEN, INK, INK2, KICK, MIST, TEAL, mo):
    mo.Html(
        f"""
        <div style="line-height:0;border-radius:16px;overflow:hidden;">
        <svg viewBox="0 0 960 210" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Act 3 - One clause flips the model">
          <defs>
            <linearGradient id="bg3" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stop-color="{INK}"/><stop offset="1" stop-color="{INK2}"/>
            </linearGradient>
            <linearGradient id="acc3" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0" stop-color="{TEAL}"/><stop offset="0.55" stop-color="{GREEN}"/><stop offset="1" stop-color="{CRIMSON}"/>
            </linearGradient>
          </defs>
          <rect x="1" y="1" width="958" height="208" rx="16" fill="url(#bg3)" stroke="#24404F"/>
          <text x="690" y="188" font-family="ui-monospace,Menlo,monospace" font-size="160" font-weight="700" fill="none" stroke="#26404F" stroke-width="2" opacity="0.5">03</text>
          <rect x="798" y="34" width="122" height="26" rx="13" fill="{INK2}" stroke="#2E5566"/>
          <text x="813" y="51" font-family="ui-monospace,Menlo,monospace" font-size="11" letter-spacing="2" fill="{KICK}">ACT 3 OF 7</text>
          <text x="40" y="54" font-family="ui-monospace,Menlo,monospace" font-size="12" letter-spacing="3.5" fill="{KICK}">THE LAYA PHARMA SURVEILLANCE SERIES</text>
          <text x="40" y="112" font-family="-apple-system,Segoe UI,sans-serif" font-size="42" font-weight="800" fill="#EEF4FB">One clause flips the model</text>
          <text x="40" y="148" font-family="ui-monospace,Menlo,monospace" font-size="14.5" fill="{MIST}">0 of 38 adverse events detected. Then the question was rephrased.</text>
          <g font-family="ui-monospace,Menlo,monospace" font-size="11">
            <rect x="620" y="62" width="74" height="21" rx="6" fill="{INK2}" stroke="#2E5566"/><text x="630" y="76" fill="{KICK}">draft A</text>
            <rect x="620" y="132" width="74" height="21" rx="6" fill="{INK2}" stroke="#2E5566"/><text x="630" y="146" fill="{KICK}">draft B</text>
          </g>
          <rect x="726" y="70" width="16" height="9" rx="3" fill="{CRIMSON}"/>
          <rect x="726" y="140" width="98" height="9" rx="3" fill="{GREEN}"/>
          <text x="726" y="92" font-family="ui-monospace,Menlo,monospace" font-size="10" fill="#6E93A0">0 / 38 detected</text>
          <text x="726" y="162" font-family="ui-monospace,Menlo,monospace" font-size="10" fill="#6E93A0">30 / 38 detected</text>
          <rect x="24" y="186" width="560" height="4" rx="2" fill="url(#acc3)"/>
        </svg>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def goal(mo):
    mo.md("""
    In this notebook, we will show how one badly-phrased question can silently
    break a decision-model pipeline.

    By the end of the notebook, you will:

    - watch a surveillance pipeline detect 0 of 38 adverse events,
    - watch the same pipeline detect most of them after rephrasing that one
      question, and
    - learn the schema-review rule that keeps this from shipping.
    """)
    return


@app.cell(hide_code=True)
def pre_data(mo):
    mo.md("""
    ## The surveillance inbox

    Scenario: a pharmacovigilance team watches social media for adverse
    events of a fictional drug, "zorvafenib". The dataset is 50 posts in 13
    languages with hand-authored ground truth: 38 genuine adverse-event
    reports and 12 pieces of chatter (price complaints, praise, availability,
    spam). The posts are synthetic and the labels are authorial ground truth.
    That is a caveat, not a flaw: it makes every grade below unambiguous.
    """)
    return


@app.cell(hide_code=True)
def imports():
    import contextlib
    import json
    import os
    import sys
    from pathlib import Path

    os.environ.setdefault("USE_TF", "0")

    import laya
    import marimo as mo
    import plotly.graph_objects as go
    import time
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
        AMBER,
        CRIMSON,
        GREEN,
        INK,
        INK2,
        KICK,
        LINE,
        MIST,
        PAPER,
        Path,
        SLATE,
        TEAL,
        contextlib,
        go,
        json,
        laya,
        make_subplots,
        mo,
    )


@app.cell(hide_code=True)
def draft_panels_class():
    import anywidget
    import traitlets

    class DraftPanels(anywidget.AnyWidget):
        """Two side-by-side schema-draft panels, 50% each."""

        _esm = """
    function render({ model, el }) {
      const esc = (s) => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      const drafts = model.get("drafts") || [];
      el.style.fontFamily = "-apple-system,BlinkMacSystemFont,'SF Pro Text',Helvetica,Arial,sans-serif";
      el.innerHTML =
        '<div style="display:flex;gap:16px;align-items:stretch;">' +
        drafts.map((d) =>
          '<div style="flex:1 1 50%;min-width:0;border:1px solid #D9D4CA;border-radius:12px;overflow:hidden;background:#fff;">' +
          '<div style="padding:12px 16px;border-bottom:1px solid #D9D4CA;">' +
          '<span style="font-weight:700;font-size:14px;color:#1D2530;">' + esc(d.label) + '</span>' +
          '<span style="margin-left:8px;font-size:12px;color:#5C6B7A;">(' + esc(d.badge) + ')</span>' +
          '</div>' +
          '<div style="margin:14px 16px;background:#F7F5F1;border-radius:8px;padding:12px 14px;' +
          'font-family:ui-monospace,Menlo,monospace;font-size:12.5px;line-height:1.6;' +
          'white-space:pre-wrap;word-break:break-word;color:#1D2530;">' + esc(d.code) + '</div>' +
          '</div>'
        ).join("") +
        '</div>';
    }
    export default { render };
    """

        drafts = traitlets.List().tag(sync=True)

    return (DraftPanels,)


@app.cell(hide_code=True)
def load_posts(Path, json):
    DATA3 = Path(__file__).resolve().parent.parent / "data"
    with open(DATA3 / "synthetic_ae_dataset.json", encoding="utf-8") as f3:
        POSTS3 = json.load(f3)

    N_AE3 = sum(1 for p3 in POSTS3 if p3["gt"][0])
    N_CHATTER3 = len(POSTS3) - N_AE3
    N_LANGS3 = len({p3["lang"] for p3 in POSTS3})
    return DATA3, N_AE3, N_CHATTER3, N_LANGS3, POSTS3


@app.cell(hide_code=True)
def load_reading(N_AE3, N_CHATTER3, N_LANGS3, POSTS3, mo):
    mo.md(f"""
    **The dataset**: {len(POSTS3)} posts, {N_AE3} adverse events,
    {N_CHATTER3} chatter, {N_LANGS3} languages. A sample, including the two
    posts we will study closely at the end (one English, one Mandarin, so
    both can be verified):
    """)
    return


@app.cell(hide_code=True)
def posts_preview(POSTS3, mo):
    preview_rows3 = [
        {
            "id": p3["id"],
            "lang": p3["lang"],
            "adverse event": p3["gt"][0],
            "post": p3["text"][:84] + ("..." if len(p3["text"]) > 84 else ""),
        }
        for p3 in POSTS3[:4] + POSTS3[-4:]
    ]
    mo.ui.table(preview_rows3, selection=None)
    return


@app.cell(hide_code=True)
def draft_texts():
    LONG3 = (
        "Does the author of this post describe a symptom or health problem "
        "that they themselves experienced while taking the medication? Adverse event reports "
        "include physical or mental symptoms, however mild. Exclude posts about price, "
        "availability, refills, general questions, praise, news, or community chat."
    )
    SHORT3 = "Does the author describe a symptom they experienced while taking the medication?"
    return LONG3, SHORT3


@app.cell(hide_code=True)
def schema_intro(mo):
    mo.md("""
    ## The question that broke the pipeline

    The whole pipeline hangs on one yes/no question: *is this post an
    adverse-event report at all?* It went through two drafts, shown verbatim
    below. Draft A reads like thorough prompt engineering to a human. To a
    yes/no head it reads like a polarity flip.
    """)
    return


@app.cell(hide_code=True)
def schema_drafts(DraftPanels, LONG3, SHORT3):
    DraftPanels(
        drafts=[
            {
                "label": "Draft A - the exclusion clause",
                "badge": "broke it",
                "code": LONG3,
            },
            {
                "label": "Draft B - short, positively phrased",
                "badge": "fixed it",
                "code": SHORT3,
            },
        ]
    )
    return


@app.cell(hide_code=True)
def schema_reading(mo):
    mo.md("""
    The clause reads as helpful context to a human: *of course* we do not
    want price talk in the AE queue. To the yes/no head it does something
    else entirely: it teaches the head that mentions of price, praise, or
    news are evidence *about the question*, and the question flips.
    """)
    return


@app.cell(hide_code=True)
def load_models(contextlib, laya, mo):
    load_ctx3 = (
        mo.spinner("loading both checkpoints ...")
        if hasattr(mo, "spinner")
        else contextlib.nullcontext()
    )
    with load_ctx3:
        english3 = laya.load("convaiinnovations/laya")
        multi3 = laya.load("convaiinnovations/laya-multilingual")
    return english3, multi3


@app.cell(hide_code=True)
def run_intro(mo):
    mo.vstack(
        [
            mo.md(
                """
                ## Run both drafts over all 50 posts

                Each run is a single-question pass per post: two checkpoints, two
                drafts, fifty posts. Under a minute on an M-series laptop. The
                detection threshold is 0.5, and the Latin-script posts route to
                the English checkpoint while the rest go multilingual, the
                routing rule Act 2 justified.
                """
            ),
            mo.accordion(
                {
                    "Click here to see the code that grades the posts": mo.md(
                        """
                        ```python
                        q = {"is_ae": {"type": "noul", "instructions": draft}}
                        p_adverse = agent.predict({"post": post}, q)["answers"]["is_ae"]["noul"]
                        # detected when p_adverse >= 0.5
                        ```
                        """
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def schema_run(
    DATA3,
    LONG3,
    N_AE3,
    POSTS3,
    SHORT3,
    contextlib,
    english3,
    json,
    mo,
    multi3,
):
    CACHE3 = DATA3 / "pv_schema_comparison.json"

    if CACHE3.exists():
        with open(CACHE3, encoding="utf-8") as f3b:
            PREDS3 = json.load(f3b)
        RUN_NOTE3 = "loaded from data/pv_schema_comparison.json (delete it to re-run live)"
    else:
        PREDS3 = {}
        run_ctx3 = (
            mo.spinner("running 200 single-question forward passes ...")
            if hasattr(mo, "spinner")
            else contextlib.nullcontext()
        )
        with run_ctx3:
            for ck3, ag3 in [("english", english3), ("multilingual", multi3)]:
                for draft3, ins3 in [("long", LONG3), ("short", SHORT3)]:
                    for p3 in POSTS3:
                        q3 = {"is_ae": {"type": "noul", "instructions": ins3}}
                        PREDS3.setdefault(ck3, {}).setdefault(draft3, {})[
                            p3["id"]
                        ] = float(
                            ag3.predict({"post": p3["text"]}, q3)["answers"][
                                "is_ae"
                            ]["noul"]
                        )
        with open(CACHE3, "w", encoding="utf-8") as f3c:
            json.dump(PREDS3, f3c)
        RUN_NOTE3 = "computed live; cached to data/pv_schema_comparison.json"

    LATIN3 = {"en", "es", "fr", "de", "pt", "it", "sw"}

    def routed3(p3, draft3):
        ck3 = "english" if p3["lang"] in LATIN3 else "multilingual"
        return PREDS3[ck3][draft3][p3["id"]]

    def summarize3(predict3):
        tp3 = sum(
            1 for x3 in POSTS3 if x3["gt"][0] and predict3[x3["id"]] >= 0.5
        )
        fn3 = sum(
            1 for x3 in POSTS3 if x3["gt"][0] and predict3[x3["id"]] < 0.5
        )
        fp3 = sum(
            1 for x3 in POSTS3 if not x3["gt"][0] and predict3[x3["id"]] >= 0.5
        )
        return tp3, fn3, fp3

    CONFIGS3 = [
        (
            "draft A, multilingual",
            lambda x3: PREDS3["multilingual"]["long"][x3["id"]],
        ),
        (
            "draft B, multilingual",
            lambda x3: PREDS3["multilingual"]["short"][x3["id"]],
        ),
        ("draft A, english", lambda x3: PREDS3["english"]["long"][x3["id"]]),
        ("draft B, english", lambda x3: PREDS3["english"]["short"][x3["id"]]),
        ("routed pipeline, draft A", lambda x3: routed3(x3, "long")),
        ("routed pipeline, draft B", lambda x3: routed3(x3, "short")),
    ]

    SUMMARY3 = []
    for label3, get3 in CONFIGS3:
        tp3, fn3, fp3 = summarize3({x3["id"]: get3(x3) for x3 in POSTS3})
        SUMMARY3.append(
            {
                "configuration": label3,
                "AE detected": f"{tp3}/{N_AE3}",
                "missed": fn3,
                "false alarms": fp3,
            }
        )
    RUN_NOTE3
    return PREDS3, RUN_NOTE3, SUMMARY3


@app.cell(hide_code=True, expand_output=True)
def detection_table(
    AMBER,
    CRIMSON,
    RUN_NOTE3,
    SLATE,
    SUMMARY3,
    TEAL,
    make_subplots,
    mo,
):
    labels3 = [s3["configuration"] for s3 in SUMMARY3][::-1]
    det3 = [int(s3["AE detected"].split("/")[0]) for s3 in SUMMARY3][::-1]
    miss3 = [s3["missed"] for s3 in SUMMARY3][::-1]
    fa3 = [s3["false alarms"] for s3 in SUMMARY3][::-1]
    archived3 = [12 - f3 for f3 in fa3]

    fig_det3 = make_subplots(
        rows=2,
        cols=1,
        shared_yaxes=True,
        vertical_spacing=0.14,
        subplot_titles=(
            "of 38 adverse events",
            "of 12 chatter posts",
        ),
    )
    fig_det3.add_bar(
        y=labels3,
        x=det3,
        orientation="h",
        name="detected",
        marker_color=TEAL,
        text=det3,
        textposition="inside",
        insidetextanchor="middle",
    )
    fig_det3.add_bar(
        y=labels3,
        x=miss3,
        orientation="h",
        name="missed",
        marker_color=CRIMSON,
        text=miss3,
        textposition="inside",
        insidetextanchor="middle",
    )
    fig_det3.add_bar(
        y=labels3,
        x=fa3,
        orientation="h",
        name="false alarms",
        showlegend=False,
        marker_color=AMBER,
        text=[f3 if f3 else "" for f3 in fa3],
        textposition="inside",
        insidetextanchor="middle",
    )
    fig_det3.add_bar(
        y=labels3,
        x=archived3,
        orientation="h",
        name="correctly archived",
        showlegend=False,
        marker_color=SLATE,
        text=[a3 if a3 else "" for a3 in archived3],
        textposition="inside",
        insidetextanchor="middle",
    )
    fig_det3.update_layout(
        barmode="stack",
        title=dict(
            text="detection outcomes per configuration - detection threshold 0.5",
            font=dict(size=13),
        ),
        height=560,
        margin=dict(l=10, r=20, t=60, b=10),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig_det3.update_xaxes(range=[0, 38])

    mo.vstack(
        [
            mo.ui.table(SUMMARY3, selection=None),
            fig_det3,
            mo.md(f"*{RUN_NOTE3}*"),
        ]
    )
    return


@app.cell(hide_code=True)
def detection_reading(SUMMARY3, mo):
    mo.md(f"""
    Read the two routed rows, the configuration that would ship.

    On the multilingual checkpoint, draft A detected
    **{SUMMARY3[0]["AE detected"]}** adverse events: the exclusion clause,
    written to *help*, suppressed every one of them. The shipped configuration,
    the routed pipeline, tells the same story: draft A
    **{SUMMARY3[4]["AE detected"]}**, draft B **{SUMMARY3[5]["AE detected"]}**
    with {SUMMARY3[5]["false alarms"]} false alarms. The clause did not add
    context; it inverted the head.
    """)
    return


@app.cell(hide_code=True)
def flip_scatter(CRIMSON, POSTS3, PREDS3, SLATE, go):
    xs3 = [PREDS3["multilingual"]["long"][x3["id"]] for x3 in POSTS3]
    ys3 = [PREDS3["multilingual"]["short"][x3["id"]] for x3 in POSTS3]
    cols3 = [CRIMSON if x3["gt"][0] else SLATE for x3 in POSTS3]
    hover3 = [f"{x3['id']} ({x3['lang']})" for x3 in POSTS3]

    fig_flip3 = go.Figure()
    fig_flip3.add_trace(
        go.Scatter(
            x=xs3,
            y=ys3,
            mode="markers",
            marker=dict(color=cols3, size=11, opacity=0.85),
            text=hover3,
            hoverinfo="text+x+y",
        )
    )
    fig_flip3.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color=SLATE, width=1, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig_flip3.update_layout(
        title=dict(
            text="P(adverse event) per post: draft A (x) vs draft B (y) - red = true adverse event, grey = chatter",
            font=dict(size=13),
        ),
        xaxis=dict(title="draft A (long, broken)", range=[-0.03, 1.03]),
        yaxis=dict(title="draft B (short, fixed)", range=[-0.03, 1.03]),
        height=520,
        margin=dict(l=10, r=20, t=60, b=10),
        template="plotly_white",
        showlegend=False,
    )
    fig_flip3
    return


@app.cell(hide_code=True)
def flip_reading(mo):
    mo.md("""
    Every red point below the diagonal is an adverse event the long schema
    pushed *down*. The clause made the head treat mentions of price, praise,
    or news as evidence against a symptom, even when a symptom was there.
    Nothing about the model changed; the question did.
    """)
    return


@app.cell(hide_code=True)
def spotlight_intro(mo):
    mo.md("""
    ## Two posts to remember

    Both are verifiable. The first is plain English. The second is Mandarin,
    which I read, and it describes an emergency-room visit.
    """)
    return


@app.cell(hide_code=True)
def spotlights(CRIMSON, GREEN, LINE, PAPER, POSTS3, PREDS3, mo):
    praise3 = next(x3 for x3 in POSTS3 if x3["id"] == "D-01")
    er3 = next(x3 for x3 in POSTS3 if x3["id"] == "GI-05")

    def post_card3(title3, body3, p_long3, p_short3, note3):
        return mo.Html(
            f"""
            <div style="border:1px solid {LINE};border-radius:12px;overflow:hidden;background:{PAPER};font-family:-apple-system,BlinkMacSystemFont,sans-serif;">
              <div style="padding:12px 18px;background:#EFEDE7;border-bottom:1px solid {LINE};font-size:12px;font-weight:600;color:#5C6B7A;">{title3}</div>
              <div style="padding:14px 18px;">
                <div style="font-size:13.5px;line-height:1.55;color:#1D2530;white-space:pre-wrap;">{body3}</div>
                <div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:12px;font-family:ui-monospace,Menlo,monospace;font-size:12px;">
                  <span style="padding:3px 10px;border-radius:999px;background:{CRIMSON}16;border:1px solid {CRIMSON}55;">draft A: P = <b>{p_long3:.2f}</b></span>
                  <span style="padding:3px 10px;border-radius:999px;background:{GREEN}16;border:1px solid {GREEN}55;">draft B: P = <b>{p_short3:.2f}</b></span>
                </div>
                <div style="font-size:12px;color:#5C6B7A;margin-top:10px;">{note3}</div>
              </div>
            </div>
            """
        )

    mo.vstack(
        [
            post_card3(
                "D-01 - praise, in English",
                praise3["text"],
                PREDS3["multilingual"]["long"]["D-01"],
                PREDS3["multilingual"]["short"]["D-01"],
                "The clause told the head to treat praise as evidence; the head started finding it everywhere.",
            ),
            post_card3(
                "GI-05 - emergency-room visit, in Mandarin (I can verify this one)",
                er3["text"],
                PREDS3["multilingual"]["long"]["GI-05"],
                PREDS3["multilingual"]["short"]["GI-05"],
                "English translation (we authored the post): Stomach pain three weeks into zorvafenib; family took me to the ER last night. A hospital visit scored near zero, in a pipeline that archives everything below threshold.",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def takeaways(mo):
    mo.md("""
    ## What to carry out of Act 3

    1. Question schemas are code. They get review discipline in pull
       requests: keep decision questions short and positively phrased.
    2. Keep a labeled probe set and re-run it whenever a schema changes; this
       notebook is that probe set, re-runnable live.
    3. A clause that reads as helpful context to a human acted as a polarity
       flip on the model.
    4. When failures come from inputs far outside the training data, the fix
       is labeled examples from your own traffic. Notebook 5 measures how far
       212 labels and 20 minutes of fine-tuning get you.
    """)
    return


if __name__ == "__main__":
    app.run()
