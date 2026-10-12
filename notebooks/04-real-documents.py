# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo==0.25.1",
#     "matplotlib",
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

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import marimo as mo
    import matplotlib.pyplot as plt4
    from collections import Counter
    from pathlib import Path

    plt4.rcParams.update(
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

    return Counter, Path, json, mo, plt4


@app.cell
def _(mo):
    mo.md(
        """
        # Act 4 - Real documents

        > **The Laya pharma surveillance series · Act 4 of 7**
        > Runs on committed data, no model downloads.

        Synthetic posts prove mechanics. This act is 265 real documents,
        fetched by script: five per category hand-curated, then scaled with
        fetchers against the Federal Register API, PubMed E-utilities, FDA
        listing pages, SEC EDGAR full-text search, FiercePharma's RSS, and
        docket-comment letters.
        """
    )
    return


@app.cell
def _(Counter, Path, json, mo):
    DATA4 = Path(__file__).resolve().parent.parent / "data"
    with open(DATA4 / "doc_triage_real_results.json", encoding="utf-8") as f4:
        FIRST50 = json.load(f4)
    with open(DATA4 / "real_docs_corpus_large.json", encoding="utf-8") as f4b:
        LARGE = json.load(f4b)
    n4 = len(FIRST50) + len(LARGE)
    mo.md(
        f"""
        **The corpus**: {len(FIRST50)} hand-curated + {len(LARGE)} script-fetched
        = **{n4} documents**, ten categories:

        {", ".join(f"`{c}`" for c in sorted({d["cat"] for d in FIRST50}))}
        """
    )
    return FIRST50, LARGE, n4


@app.cell
def _(Counter, FIRST50, LARGE, mo, plt4):
    cats4 = Counter(d["cat"] for d in FIRST50) + Counter(d["cat"] for d in LARGE)
    order4 = sorted(cats4, key=cats4.get)
    fig_cat4, ax_cat4 = plt4.subplots(figsize=(8, 4))
    ax_cat4.barh(order4, [cats4[c] for c in order4], color="#4a7ebb")
    ax_cat4.set_xlabel("documents")
    ax_cat4.set_title("ten categories, all real sources except the marked spam")
    fig_cat4.tight_layout()
    mo.vstack([fig_cat4])
    return (fig_cat4,)


@app.cell
def _(mo):
    mo.md("## Browse the corpus")
    return


@app.cell
def _(FIRST50, LARGE, mo):
    ALL_DOCS4 = FIRST50 + LARGE
    dropdown4 = mo.ui.dropdown(
        options=sorted({d["cat"] for d in ALL_DOCS4}),
        value="inspection_finding",
        label="category: ",
    )
    return ALL_DOCS4, dropdown4


@app.cell
def _(ALL_DOCS4, dropdown4, mo):
    chosen4 = [d for d in ALL_DOCS4 if d["cat"] == dropdown4.value]
    rows4 = [
        {
            "id": d["id"],
            "source": d["source"],
            "title": d["title"][:80],
            "text": d["text"][:140] + ("..." if len(d["text"]) > 140 else ""),
            "url": d["url"],
        }
        for d in chosen4
    ]
    mo.vstack(
        [
            dropdown4,
            mo.md(f"{len(chosen4)} documents in `{dropdown4.value}`"),
            mo.ui.table(rows4, selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Zero-shot on the first 50: the sweep

        Same schema as the fine-tuning experiment: primary category,
        secondary, owning team, actionability, urgency. But raw, before any
        training on this distribution.
        """
    )
    return


@app.cell
def _(FIRST50, mo):
    OPERATIONAL4 = {
        "regulatory_guidance",
        "inspection_finding",
        "safety_communication",
        "scientific_publication",
    }
    qmap4 = [
        ("primary category", "cat", "doc_type"),
        ("secondary category", "cat2", "secondary_type"),
        ("owning team", "team", "route_to"),
        ("urgency (ordinal)", "urg", "urgency"),
    ]
    sweep4 = []
    for label4, gt4, pred4 in qmap4:
        pairs4 = [(d[gt4], d["pred"][pred4]) for d in FIRST50 if d[gt4] is not None]
        ops4 = [
            (g4, p4)
            for g4, p4 in pairs4
            if gt4 != "cat" or g4 in OPERATIONAL4
        ]
        rest4 = [
            (g4, p4)
            for g4, p4 in pairs4
            if gt4 == "cat" and g4 not in OPERATIONAL4
        ]
        if gt4 != "cat":
            ops4, rest4 = pairs4, []
        op_ok4 = sum(1 for g4, p4 in ops4 if g4 == p4)
        rest_ok4 = sum(1 for g4, p4 in rest4 if g4 == p4)
        sweep4.append(
            {
                "question": label4,
                "operational 4 cats": (
                    f"{op_ok4}/{len(ops4)}" if ops4 else "-"
                ),
                "other 6 cats": (f"{rest_ok4}/{len(rest4)}" if rest4 else "-"),
            }
        )
    action_ops4 = [
        (d["action"], d["pred"]["action_required"] >= 0.5)
        for d in FIRST50
        if d["cat"] in OPERATIONAL4
    ]
    action_rest4 = [
        (d["action"], d["pred"]["action_required"] >= 0.5)
        for d in FIRST50
        if d["cat"] not in OPERATIONAL4
    ]
    sweep4.append(
        {
            "question": "action required (noul @ 0.5)",
            "operational 4 cats": f"{sum(1 for g4, p4 in action_ops4 if g4 == p4)}/{len(action_ops4)}",
            "other 6 cats": f"{sum(1 for g4, p4 in action_rest4 if g4 == p4)}/{len(action_rest4)}",
        }
    )
    mo.vstack(
        [
            mo.ui.table(sweep4, selection=None),
            mo.md(
                """
                Zero-shot, no fine-tuning. The four operational categories -
                regulatory guidance, inspection findings, safety
                communications, scientific publications - are perfect: 20/20
                on primary category. Everything else is where the edges
                live.
                """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        Five operational categories, zero-shot: all 20 inspection findings,
        safety communications, regulatory guidance, and publications land.
        Now the part that matters more than the headline.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## The failure gallery")
    return


@app.cell
def _(FIRST50, mo):
    # Where zero-shot struggles, measured on the 50 docs that carry predictions.
    wrong4 = []
    for d in FIRST50:
        p4 = d["pred"]["doc_type"]
        if p4 != d["cat"]:
            wrong4.append(
                {
                    "id": d["id"],
                    "source": d["source"],
                    "label": d["cat"],
                    "predicted": p4,
                    "P(pred)": round(d["pred"]["type_probs"].get(p4, 0), 3),
                    "title": d["title"][:70],
                }
            )
    gallery4 = mo.ui.table(wrong4, selection=None)
    mo.vstack(
        [
            mo.md(
                f"**{len(wrong4)} primary-category miss in 50, zero-shot** (the large "
                "corpus carries labels for fine-tuning, not zero-shot predictions). "
                "The misses concentrate where the *genre* matches better than the "
                "content: press articles about warning letters that read as warning "
                "letters, spam dressed as legitimate announcements. Click through:"
            ),
            gallery4,
        ]
    )
    return (gallery4,)


@app.cell
def _(FIRST50, LARGE, mo):
    spam4 = [doc4 for doc4 in FIRST50 if doc4["cat"] == "spam_or_irrelevant"]
    examples4 = []
    for ds4 in spam4[:6]:
        top4 = sorted(
            ds4["pred"]["type_probs"].items(), key=lambda kv: -kv[1]
        )[0]
        examples4.append(
            {
                "id": ds4["id"],
                "source": ds4["source"],
                "what it is": ds4["title"][:60],
                "top prediction": f"{top4[0]} ({top4[1]:.2f})",
                "text": ds4["text"][:120] + "...",
            }
        )
    mo.vstack(
        [
            mo.md("### Spam, wearing suits"),
            mo.ui.table(examples4, selection=None),
            mo.md(
                """
                The spam never says "I am spam". It dresses as a press release
                or a publication, and a zero-shot triage front half lets it
                through.
                """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## What to carry out of Act 4

        1. Zero-shot covers the boring 90%: well-separated operational
           categories classify perfectly on real documents.
        2. The edges are predictable: genre-vs-content collisions and spam.
           Exactly the examples above.
        3. Both edges have the same fix: labeled examples from your own
           distribution. Which is Act 5 - 212 labeled documents and 20
           minutes of fine-tuning.
        """
    )
    return


if __name__ == "__main__":
    app.run()
