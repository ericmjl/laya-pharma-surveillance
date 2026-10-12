# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo==0.25.1",
#     "plotly==7.1.0",
#     "marimo-lens==0.2.3",
#     "marimo-studio==0.3.0",
#     "anywidget==0.11.0",
#     "traitlets==5.16.1",
# ]
# ///
"""Act 4: Real documents.

265 real, scraped pharma documents: FDA warning letters, Drug Safety
Communications, Federal Register notices, PubMed abstracts, DOJ False Claims
Act settlements, FiercePharma articles, vendor press releases, 10-K risk
factors, docket-comment letters, and synthetic spam. Zero-shot, Laya got all
20 documents in the four operational categories right on the first 50-doc
corpus, and fumbled the edges: press articles about warning letters, spam
dressed as announcements. This notebook shows the corpus, the zero-shot
sweep, and the failure gallery. No model downloads: everything renders from
data/*.json.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def banner(AMBER, CRIMSON, GREEN, INK, INK2, KICK, MIST, TEAL, mo):
    mo.Html(
        f"""
        <div style="line-height:0;border-radius:16px;overflow:hidden;">
        <svg viewBox="0 0 960 210" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Act 4 - Real documents">
          <defs>
            <linearGradient id="bg4" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stop-color="{INK}"/><stop offset="1" stop-color="{INK2}"/>
            </linearGradient>
            <linearGradient id="acc4" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0" stop-color="{TEAL}"/><stop offset="0.55" stop-color="{GREEN}"/><stop offset="1" stop-color="{CRIMSON}"/>
            </linearGradient>
          </defs>
          <rect x="1" y="1" width="958" height="208" rx="16" fill="url(#bg4)" stroke="#24404F"/>
          <text x="690" y="188" font-family="ui-monospace,Menlo,monospace" font-size="160" font-weight="700" fill="none" stroke="#26404F" stroke-width="2" opacity="0.5">04</text>
          <rect x="798" y="34" width="122" height="26" rx="13" fill="{INK2}" stroke="#2E5566"/>
          <text x="813" y="51" font-family="ui-monospace,Menlo,monospace" font-size="11" letter-spacing="2" fill="{KICK}">ACT 4 OF 7</text>
          <text x="40" y="54" font-family="ui-monospace,Menlo,monospace" font-size="12" letter-spacing="3.5" fill="{KICK}">THE LAYA PHARMA SURVEILLANCE SERIES</text>
          <text x="40" y="112" font-family="-apple-system,Segoe UI,sans-serif" font-size="42" font-weight="800" fill="#EEF4FB">Real documents</text>
          <text x="40" y="148" font-family="ui-monospace,Menlo,monospace" font-size="14.5" fill="{MIST}">265 scraped pharma documents &#183; zero-shot &#183; and the failure gallery</text>
          <g font-family="ui-monospace,Menlo,monospace" font-size="11">
            <rect x="620" y="62" width="74" height="21" rx="6" fill="{INK2}" stroke="#2E5566"/><text x="630" y="76" fill="{KICK}">FDA</text>
            <rect x="620" y="92" width="74" height="21" rx="6" fill="{INK2}" stroke="#2E5566"/><text x="630" y="106" fill="{KICK}">PubMed</text>
            <rect x="620" y="122" width="74" height="21" rx="6" fill="{INK2}" stroke="#2E5566"/><text x="630" y="136" fill="{KICK}">EDGAR</text>
            <rect x="620" y="152" width="74" height="21" rx="6" fill="{INK2}" stroke="#2E5566"/><text x="630" y="166" fill="{KICK}">press</text>
          </g>
          <rect x="726" y="70" width="98" height="9" rx="3" fill="{TEAL}"/>
          <rect x="726" y="94" width="72" height="9" rx="3" fill="{TEAL}" opacity="0.7"/>
          <rect x="726" y="118" width="50" height="9" rx="3" fill="{TEAL}" opacity="0.45"/>
          <rect x="726" y="142" width="30" height="9" rx="3" fill="{AMBER}" opacity="0.85"/>
          <text x="726" y="166" font-family="ui-monospace,Menlo,monospace" font-size="10" fill="#6E93A0">documents per category</text>
          <rect x="24" y="186" width="560" height="4" rx="2" fill="url(#acc4)"/>
        </svg>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def goal(mo):
    mo.md("""
    In this notebook, we will show what the decision model does zero-shot on
    real, scraped pharma documents.

    By the end of the notebook, you will:

    - browse the corpus: 265 real documents from FDA, PubMed, SEC EDGAR,
      Federal Register, FiercePharma, DOJ, and docket comments,
    - see zero-shot classify all 20 documents in the four operational
      categories perfectly,
    - watch it fail on exactly the edges: press about warning letters and
      spam dressed as science, and
    - see why Act 5's 212 labeled documents matter.
    """)
    return


@app.cell(hide_code=True)
def pre_data(mo):
    mo.md("""
    ## The corpus

    Synthetic posts prove mechanics; this act is 265 real documents. The
    first 50 were hand-curated, five per category, from FDA warning letters,
    Drug Safety Communications, Federal Register notices, PubMed abstracts,
    DOJ False Claims Act settlements, FiercePharma articles, vendor press
    releases, 10-K risk factors, docket-comment letters, and synthetic spam.
    I then scaled the corpus with script-side fetchers against the Federal
    Register API, PubMed E-utilities, FDA listing pages, SEC EDGAR full-text
    search, and FiercePharma's RSS, so the ingestion is reproducible. Labels
    are authorial ground truth; zero-shot predictions exist for the first 50
    documents only.
    """)
    return


@app.cell(hide_code=True)
def imports():
    import json
    import sys
    from collections import Counter
    from pathlib import Path

    import anywidget
    import marimo as mo
    import plotly.graph_objects as go
    import traitlets

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mail_viewer import DocViewer, SampleDocs

    # Pharmacopeia theme (see .agents/skills/laya-pharma-theme/SKILL.md)
    INK = "#0F2233"  # clinical navy: banners, dark cards
    INK2 = "#16303F"  # gradient end, chip fills
    TEAL = "#1F7A8C"  # assay teal: primary data series
    GREEN = "#2F9E68"  # recovery green: safe / selected
    AMBER = "#C9821F"  # caution amber: spam / warning
    CRIMSON = "#B3282D"  # adverse crimson: danger ONLY
    SLATE = "#8A97A5"  # secondary series
    MIST = "#A7B3BD"  # muted text on ink
    KICK = "#8FB0B8"  # kicker text on ink
    LINE = "#D9D4CA"  # warm hairline
    PAPER = "#F7F5F1"  # light card background

    CAT_DISPLAY = {
        "legal": "Legal",
        "public_comment": "Public Comment",
        "commercial": "Commercial",
        "inspection_finding": "Inspection Finding",
        "safety_communication": "Safety Communication",
        "press_coverage": "Press Coverage",
        "spam_or_irrelevant": "Spam or Irrelevant *",
        "regulatory_guidance": "Regulatory Guidance",
        "corporate_report": "Corporate Report",
        "scientific_publication": "Scientific Publication",
    }
    return (
        AMBER,
        CAT_DISPLAY,
        CRIMSON,
        Counter,
        DocViewer,
        GREEN,
        INK,
        INK2,
        KICK,
        MIST,
        Path,
        SampleDocs,
        TEAL,
        go,
        json,
        mo,
    )


@app.cell(hide_code=True)
def load_corpus(Path, json):
    DATA4 = Path(__file__).resolve().parent.parent / "data"
    with open(DATA4 / "doc_triage_real_results.json", encoding="utf-8") as f4:
        FIRST50 = json.load(f4)
    with open(DATA4 / "real_docs_corpus_large.json", encoding="utf-8") as f4b:
        LARGE = json.load(f4b)
    ALL_DOCS4 = FIRST50 + LARGE
    return ALL_DOCS4, FIRST50, LARGE


@app.cell(hide_code=True)
def load_reading(ALL_DOCS4, FIRST50, LARGE, mo):
    mo.md(f"""
    **The corpus**: {len(FIRST50)} hand-curated + {len(LARGE)} script-fetched
    = **{len(ALL_DOCS4)} documents** across ten categories. The first 50
    carry zero-shot predictions from the model; the larger corpus carries
    labels for the fine-tuning act.
    """)
    return


@app.cell(hide_code=True)
def corpus_sampler(ALL_DOCS4, CAT_DISPLAY, SampleDocs):
    sample_docs4 = [
        {
            "category": CAT_DISPLAY[d4["cat"]],
            "title": d4["title"],
            "source": d4["source"],
            "snippet": d4["text"][:180]
            + ("..." if len(d4["text"]) > 180 else ""),
            "text": d4["text"],
        }
        for d4 in ALL_DOCS4
    ]

    SampleDocs(docs=sample_docs4)
    return


@app.cell(hide_code=True)
def corpus_chart(ALL_DOCS4, CAT_DISPLAY, Counter, TEAL, go):
    counts4 = Counter(d4["cat"] for d4 in ALL_DOCS4)
    cats4 = sorted(counts4, key=counts4.get)
    labels4 = [CAT_DISPLAY[c4] for c4 in cats4]

    fig_cat4 = go.Figure(
        go.Bar(
            x=[counts4[c4] for c4 in cats4],
            y=labels4,
            orientation="h",
            marker_color=TEAL,
            text=[counts4[c4] for c4 in cats4],
            textposition="outside",
        )
    )
    fig_cat4.update_layout(
        title=dict(
            text="documents per category (* = the synthetic spam sample; everything else is real)",
            font=dict(size=13),
        ),
        height=420,
        margin=dict(l=10, r=40, t=46, b=10),
        template="plotly_white",
        xaxis=dict(title="documents", range=[0, max(counts4.values()) * 1.15]),
    )
    fig_cat4
    return


@app.cell(hide_code=True)
def corpus_chart_reading(mo):
    mo.md("""
    Amber is the only synthetic slice: representative spam samples marked as
    such. Everything else was scraped from the source. Browse it yourself
    below.
    """)
    return


@app.cell(hide_code=True)
def browser_controls(ALL_DOCS4, CAT_DISPLAY, mo):
    category_pick4 = mo.ui.dropdown(
        options=sorted({CAT_DISPLAY[d4["cat"]] for d4 in ALL_DOCS4}),
        value="Inspection Finding",
        label="category: ",
    )
    category_pick4
    return (category_pick4,)


@app.cell(hide_code=True)
def corpus_browser(ALL_DOCS4, CAT_DISPLAY, category_pick4, mo):
    slug_by_name4 = {CAT_DISPLAY[d4["cat"]]: d4["cat"] for d4 in ALL_DOCS4}
    picked_slug4 = slug_by_name4[category_pick4.value]
    picked_docs4 = [d4 for d4 in ALL_DOCS4 if d4["cat"] == picked_slug4]
    browser_rows4 = [
        {
            "id": d4["id"],
            "source": d4["source"],
            "title": d4["title"][:80],
            "text": d4["text"][:140]
            + ("..." if len(d4["text"]) > 140 else ""),
            "url": d4["url"],
        }
        for d4 in picked_docs4
    ]
    mo.vstack(
        [
            mo.md(f"{len(picked_docs4)} documents in {category_pick4.value}"),
            mo.ui.table(browser_rows4, selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def sweep_intro(mo):
    mo.vstack(
        [
            mo.md(
                """
                ## Zero-shot on the first 50

                Every document gets one forward pass with five questions: primary
                category, secondary type, owning team, action required, and
                urgency. Raw zero-shot: no training on this distribution, and
                predictions cover only the 50 hand-curated documents. Act 5
                fine-tunes this exact schema; here it runs raw. The chart below
                shows how often each answer matched the label.
                """
            ),
            mo.accordion(
                {
                    "Click here to see the code that runs the predictions": mo.md(
                        """
                        ```python
                        answers = agent.predict(document, TRIAGE_Q4)["answers"]
                        # TRIAGE_Q4: five typed questions per document -
                        #   primary category (choice, 10-way)
                        #   secondary type (choice)
                        #   owning team (choice, 8-way)
                        #   action required (noul, yes/no)
                        #   urgency (score, 0-2)
                        ```
                        """
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def sweep_data(FIRST50):
    OPERATIONAL4 = {
        "regulatory_guidance",
        "inspection_finding",
        "safety_communication",
        "scientific_publication",
    }

    ops4 = [d4 for d4 in FIRST50 if d4["cat"] in OPERATIONAL4]
    rest4 = [d4 for d4 in FIRST50 if d4["cat"] not in OPERATIONAL4]
    ops_acc4 = sum(
        1 for d4 in ops4 if d4["pred"]["doc_type"] == d4["cat"]
    ) / len(ops4)
    rest_acc4 = sum(
        1 for d4 in rest4 if d4["pred"]["doc_type"] == d4["cat"]
    ) / len(rest4)
    team_acc4 = sum(
        1 for d4 in FIRST50 if d4["pred"]["route_to"] == d4["team"]
    ) / len(FIRST50)
    action_acc4 = sum(
        1
        for d4 in FIRST50
        if (d4["pred"]["action_required"] >= 0.5) == d4["action"]
    ) / len(FIRST50)
    urg_acc4 = sum(
        1 for d4 in FIRST50 if d4["pred"]["urgency"] == d4["urg"]
    ) / len(FIRST50)

    WRONG4 = [d4 for d4 in FIRST50 if d4["pred"]["doc_type"] != d4["cat"]]
    return WRONG4, action_acc4, ops4, ops_acc4, rest_acc4, team_acc4, urg_acc4


@app.cell(hide_code=True)
def sweep_chart(
    GREEN,
    TEAL,
    action_acc4,
    go,
    ops_acc4,
    rest_acc4,
    team_acc4,
    urg_acc4,
):
    sweep_labels4 = [
        "primary category (operational 4)",
        "primary category (other 6)",
        "urgency (exact 0-2 match)",
        "action required (threshold 0.5)",
        "owning team (8-way)",
    ]
    sweep_vals4 = [ops_acc4, rest_acc4, urg_acc4, action_acc4, team_acc4]
    sweep_colors4 = [GREEN, TEAL, TEAL, TEAL, TEAL]

    fig_sweep4 = go.Figure(
        go.Bar(
            x=sweep_vals4,
            y=sweep_labels4,
            orientation="h",
            marker_color=sweep_colors4,
            text=[f"{v4:.0%}" for v4 in sweep_vals4],
            textposition="outside",
        )
    )
    fig_sweep4.update_layout(
        title=dict(
            text="zero-shot accuracy on 50 real documents (green = the operational core)",
            font=dict(size=13),
        ),
        height=300,
        margin=dict(l=10, r=60, t=46, b=10),
        template="plotly_white",
        xaxis=dict(title="accuracy", range=[0, 1.15]),
    )
    fig_sweep4
    return


@app.cell(hide_code=True)
def sweep_legend(mo):
    mo.md("""
    What each row measures:

    - **Primary category**: does the model's 10-way label match? The four
      operational categories - regulatory guidance, inspection findings,
      safety communications, scientific publications - are the routes a
      surveillance pipeline acts on; the other six are everything else.
    - **Owning team**: did the 8-way team assignment match the label?
    - **Action required**: does P(action) >= 0.5 match the yes/no label?
    - **Urgency**: did the 0-2 score match exactly?
    """)
    return


@app.cell(hide_code=True)
def sweep_reading(action_acc4, mo, ops4, rest_acc4, team_acc4, urg_acc4):
    mo.md(f"""
    The four operational categories - regulatory guidance, inspection
    findings, safety communications, scientific publications - are perfect
    zero-shot: {sum(1 for d4 in ops4 if d4["pred"]["doc_type"] == d4["cat"])}/{len(ops4)}.
    Nothing here is cherry-picked; those are the categories a surveillance
    pipeline actually routes on.

    The rest is exactly where the edges live: primary category drops to
    {rest_acc4:.0%} outside the operational core, and owning team
    ({team_acc4:.0%}), actionability ({action_acc4:.0%}), and urgency
    ({urg_acc4:.0%}) sit barely above coin flips. Hold that thought for
    notebook 5.
    """)
    return


@app.cell(hide_code=True)
def failure_intro(mo):
    mo.md("""
    ## The failure gallery

    Every primary-category miss in the 50-doc sweep, in full. Two patterns
    cover all of them: genre-versus-content collisions (press coverage
    *about* a warning letter reads as a warning letter) and spam dressed as
    a legitimate category.
    """)
    return


@app.cell(hide_code=True)
def failure_gallery(WRONG4, mo):
    gallery_rows4 = [
        {
            "id": d4["id"],
            "source": d4["source"],
            "label": d4["cat"],
            "predicted": d4["pred"]["doc_type"],
            "P(pred)": round(
                d4["pred"]["type_probs"].get(d4["pred"]["doc_type"], 0), 3
            ),
            "title": d4["title"][:70],
        }
        for d4 in WRONG4
    ]
    mo.ui.table(gallery_rows4, selection=None)
    return


@app.cell(hide_code=True)
def failure_viewer(DocViewer, WRONG4):
    viewer_docs4 = [
        {
            "id": d4["id"],
            "source": d4["source"],
            "title": d4["title"],
            "body": d4["text"][:500]
            + ("..." if len(d4["text"]) > 500 else ""),
            "label": d4["cat"],
            "predicted": d4["pred"]["doc_type"],
            "prob": f"{d4['pred']['type_probs'].get(d4['pred']['doc_type'], 0):.2f}",
        }
        for d4 in WRONG4
    ]

    DocViewer(docs=viewer_docs4)
    return


@app.cell(hide_code=True)
def failure_reading(mo):
    mo.md("""
    Both edges have the same shape: nothing is random. A FiercePharma
    article about an inspection carries the inspection's vocabulary; a
    predatory-journal solicitation carries a paper's formatting. Zero-shot
    triage lets them through, and a surveillance pipeline cannot afford
    that.

    The fix is not a bigger model; it is labeled examples from this exact
    distribution. Notebook 5: 212 of them, 20 minutes, and the edges close.
    """)
    return


@app.cell(hide_code=True)
def takeaways(mo):
    mo.md("""
    ## What to carry out of Act 4

    1. Zero-shot covers the boring core: well-separated operational
       categories classify perfectly on real documents.
    2. The edges are predictable - genre-versus-content collisions and spam
       - which means they can be tested for, not discovered in production.
    3. The corpus is script-side and reproducible: the fetchers, not a chat
       assistant, built the 265-document dataset.
    4. The fix for the edges is labeled examples from your own traffic, and
       notebook 5 shows how little it takes.
    """)
    return


if __name__ == "__main__":
    app.run()
