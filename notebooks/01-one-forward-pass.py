# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo==0.25.1",
#     "laya==0.3.5",
#     "matplotlib",
# ]
# ///
"""Act 1: One forward pass.

The claim: a 421M-parameter model answers four typed questions about an email
in a single forward pass, no text generation, tens of milliseconds. This
notebook shows it live. First run downloads the checkpoint from HuggingFace.
"""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import time
    import matplotlib.pyplot as plt

    plt.rcParams.update(
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

    return mo, time


@app.cell
def _(mo):
    mo.md(
        """
        # Act 1 - One forward pass

        > **The Laya pharma surveillance series · Act 1 of 7**
        > Run me: `uvx marimo edit --sandbox notebooks/01-one-forward-pass.py`

        Laya is a decision model: typed questions in, probability distributions
        out, **one forward pass, no text generated**. Below: one support email,
        four questions, one pass. Watch the timer.

        *First cell run downloads the checkpoint from HuggingFace (~840 MB); after
        that it loads from cache in seconds.*
        """
    )
    return


@app.cell
def _(mo):
    import contextlib
    import os

    os.environ.setdefault("USE_TF", "0")

    import laya

    load_ctx1 = (
        mo.spinner("loading convaiinnovations/laya ...")
        if hasattr(mo, "spinner")
        else contextlib.nullcontext()
    )
    with load_ctx1:
        agent1 = laya.load("convaiinnovations/laya")

    n_params1 = sum(p.numel() for p in agent1.model.parameters())
    meta1 = (
        f"checkpoint `convaiinnovations/laya` (Apache 2.0) · "
        f"{n_params1 / 1e6:.0f}M params · device `{agent1.device}` · "
        f"{agent1.cfg.get('max_len')}-token budget per question"
    )
    return agent1, meta1


@app.cell
def _(meta1, mo):
    mo.md(f"**Model**: {meta1}")
    return


@app.cell
def _(mo):
    mo.md("## The ticket and the schema")
    return


@app.cell
def _():
    EMAIL = {
        "subject": "Duplicate charge on invoice 4411",
        "body": (
            "We were billed twice for March. Please refund the duplicate. "
            "If this isn't fixed this week we will cancel our account."
        ),
    }

    # three primitives, four questions, one schema:
    #   choice -> pick from options defined at request time
    #   score  -> place on an ordinal rubric
    #   noul   -> calibrated P(yes/no statement holds)
    TRIAGE_Q = {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this email?",
            "criteria": {
                "billing": "invoices, payments, refunds, charges",
                "technical": "bugs, outages, system errors",
                "sales": "pricing, new contracts, upgrades",
                "other": "everything else",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": [
                "not urgent",
                "soon",
                "critical deadline or blocking issue",
            ],
        },
        "is_phishing": {
            "type": "noul",
            "instructions": "Is this email a phishing or scam attempt?",
        },
        "churn_risk": {
            "type": "noul",
            "instructions": "Does the user threaten to cancel or leave?",
        },
    }
    return EMAIL, TRIAGE_Q


@app.cell
def _(EMAIL, mo):
    mo.md(
        f"""**Email**

        > **{EMAIL["subject"]}**
        >
        > {EMAIL["body"]}
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## Run it")
    return


@app.cell
def _(EMAIL, TRIAGE_Q, agent1, contextlib, mo, time):
    run_ctx1 = (
        mo.spinner("one forward pass ...")
        if hasattr(mo, "spinner")
        else contextlib.nullcontext()
    )
    with run_ctx1:
        t01 = time.perf_counter()
        result1 = agent1.predict(EMAIL, TRIAGE_Q)
        elapsed_ms1 = (time.perf_counter() - t01) * 1000
    a1 = result1["answers"]
    return a1, elapsed_ms1


@app.cell
def _(a1, elapsed_ms1, mo):
    mo.md(
        f"""
        ## What came back: {elapsed_ms1:.0f} ms, four answers, zero generated text

        | primitive | question | answer |
        |---|---|---|
        | `choice` | which department | **{a1["department"]["choice"]}** ({a1["department"]["confidence"]:.2f}) |
        | `score` | urgency | **{a1["urgency"]["score"]:.2f}** on 0-2 |
        | `noul` | is this phishing | **{a1["is_phishing"]["noul"]:.3f}** |
        | `noul` | churn risk | **{a1["churn_risk"]["noul"]:.3f}** |
        """
    )
    return


@app.cell
def _(a1, mo):
    dept_probs1 = a1["department"]["probabilities"]
    import matplotlib.pyplot as plt_dept

    order1 = sorted(dept_probs1, key=dept_probs1.get)
    fig_dept1, ax_dept1 = plt_dept.subplots(figsize=(7, 2.5))
    ax_dept1.barh(order1, [dept_probs1[k] for k in order1], color="#4a7ebb")
    ax_dept1.set_title("P(department) - the full choice distribution")
    ax_dept1.axvline(1.0, color="grey", linewidth=0.5)
    fig_dept1.tight_layout()
    mo.vstack([fig_dept1])
    return (fig_dept1,)


@app.cell
def _(a1, mo):
    urg1 = a1["urgency"]["probabilities"]
    import matplotlib.pyplot as plt_urg

    labels1 = ["not urgent", "soon", "critical"]
    fig_urg1, ax_urg1 = plt_urg.subplots(figsize=(7, 2.2))
    ax_urg1.bar(labels1, [urg1[k] for k in ("0", "1", "2")], color="#4ade80")
    ax_urg1.set_title(f"P(urgency) - score head, mean {a1['urgency']['score']:.2f}")
    fig_urg1.tight_layout()
    mo.vstack([fig_urg1])
    return (fig_urg1,)


@app.cell
def _(mo):
    mo.md(
        """
        Nothing to parse. The output space is numeric and typed, so there is
        no JSON to repair and nothing to hallucinate. The same four questions
        cost one forward pass, not four.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## Now five emails, same schema")
    return


@app.cell
def _(TRIAGE_Q, agent1, time):
    EMAILS1 = {
        "duplicate charge": {
            "subject": "Duplicate charge on invoice 4411",
            "body": "We were billed twice for March. Please refund the duplicate. If this isn't fixed this week we will cancel our account.",
        },
        "obvious phishing": {
            "subject": "URGENT: Your account will be suspended",
            "body": "Dear customer, we detected unusual activity. Click http://paypal-secure-verify.example-login.com immediately to confirm your password and SSN, or your account will be permanently deleted within 24 hours.",
        },
        "prod outage": {
            "subject": "Production down - checkout failing for all customers",
            "body": "Since 9am UTC every checkout returns a 500 error. We are losing orders every minute this stays broken. Please treat this as a P1.",
        },
        "happy customer": {
            "subject": "Loving the new dashboard!",
            "body": "Just wanted to say the new analytics dashboard is fantastic. Any chance you'll add CSV export next quarter? No rush at all.",
        },
        "BEC scam": {
            "subject": "Re: Q3 invoice - updated banking details",
            "body": "Hi, following our conversation last week, please note our accounts payable email has changed. Kindly update your records and process the attached invoice to the new details by Friday.",
        },
    }
    sweep_rows1 = []
    for name1, email1 in EMAILS1.items():
        t1 = time.perf_counter()
        r1 = agent1.predict(email1, TRIAGE_Q)["answers"]
        ms1 = (time.perf_counter() - t1) * 1000
        sweep_rows1.append(
            {
                "email": name1,
                "department": f'{r1["department"]["choice"]} ({r1["department"]["confidence"]:.2f})',
                "urgency": f'{r1["urgency"]["score"]:.2f}',
                "P(phishing)": f'{r1["is_phishing"]["noul"]:.3f}',
                "P(churn)": f'{r1["churn_risk"]["noul"]:.3f}',
                "ms": f"{ms1:.0f}",
            }
        )
    return sweep_rows1


@app.cell
def _(mo, sweep_rows1):
    mo.ui.table(sweep_rows1, selection=None)
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## The same job with an LLM

        Prompt it to return JSON, generate the tokens, parse the output, retry
        when the JSON is malformed, sample the temperature and hope the
        distribution means something. Seconds per email instead of
        milliseconds, and no native probabilities.

        Here: one forward pass per email, and the probabilities are the
        native output, not a sampled string you have to trust.

        **But should you trust those confidence numbers?** Act 2: feed this
        model a language it has never seen, and watch the confidence score
        stay perfectly silent.
        """
    )
    return


if __name__ == "__main__":
    app.run()
