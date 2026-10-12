# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo==0.25.1",
#     "laya==0.3.5",
#     "plotly==7.1.0",
#     "anywidget==0.11.0",
#     "marimo-lens==0.2.3",
#     "marimo-studio==0.3.0",
# ]
# ///
"""Act 1: One forward pass.

The claim: a 421M-parameter model answers four typed questions about an email
in a single forward pass, no text generation, tens of milliseconds. This
notebook shows it live. First run downloads the checkpoint from HuggingFace.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import contextlib
    import os
    import sys
    from pathlib import Path

    os.environ.setdefault("USE_TF", "0")

    import laya
    import marimo as mo
    import plotly.graph_objects as go
    import time

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mail_viewer import MailViewer

    return MailViewer, contextlib, go, laya, mo, time


@app.cell(hide_code=True)
def banner(mo):
    mo.Html("""
    <div style="line-height:0;border-radius:16px;overflow:hidden;">
    <svg viewBox="0 0 960 210" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Act 1 - One forward pass">
      <defs>
        <linearGradient id="bg1" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="#0b1524"/><stop offset="1" stop-color="#13253f"/>
        </linearGradient>
        <linearGradient id="acc1" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stop-color="#4a7ebb"/><stop offset="0.55" stop-color="#4ade80"/><stop offset="1" stop-color="#d9534f"/>
        </linearGradient>
      </defs>
      <rect x="1" y="1" width="958" height="208" rx="16" fill="url(#bg1)" stroke="#22364f"/>
      <text x="700" y="185" font-family="ui-monospace,Menlo,monospace" font-size="170" font-weight="700" fill="none" stroke="#26405f" stroke-width="2" opacity="0.5">01</text>
      <rect x="798" y="34" width="122" height="26" rx="13" fill="#1b2f4a" stroke="#33507a"/>
      <text x="813" y="51" font-family="ui-monospace,Menlo,monospace" font-size="11" letter-spacing="2" fill="#9fc0e8">ACT 1 OF 7</text>
      <text x="40" y="54" font-family="ui-monospace,Menlo,monospace" font-size="12" letter-spacing="3.5" fill="#7f9cc0">THE LAYA PHARMA SURVEILLANCE SERIES</text>
      <text x="40" y="112" font-family="-apple-system,Segoe UI,sans-serif" font-size="42" font-weight="800" fill="#eef4fb">One forward pass</text>
      <text x="40" y="148" font-family="ui-monospace,Menlo,monospace" font-size="14.5" fill="#a8b2bd">Four typed questions &#183; one ~40 ms pass &#183; zero generated text</text>
      <g font-family="ui-monospace,Menlo,monospace" font-size="11">
        <rect x="596" y="52" width="74" height="21" rx="6" fill="#1b2f4a" stroke="#33507a"/><text x="606" y="66" fill="#9fc0e8">choice</text>
        <rect x="596" y="82" width="74" height="21" rx="6" fill="#1b2f4a" stroke="#33507a"/><text x="606" y="96" fill="#9fc0e8">score</text>
        <rect x="596" y="112" width="74" height="21" rx="6" fill="#1b2f4a" stroke="#33507a"/><text x="606" y="126" fill="#9fc0e8">noul</text>
        <rect x="596" y="142" width="74" height="21" rx="6" fill="#1b2f4a" stroke="#33507a"/><text x="606" y="156" fill="#9fc0e8">noul</text>
      </g>
      <path d="M670 62 C 710 62 726 100 756 101" stroke="#33507a" fill="none"/>
      <path d="M670 92 C 706 92 724 101 756 101" stroke="#33507a" fill="none"/>
      <path d="M670 122 C 706 122 724 103 756 103" stroke="#33507a" fill="none"/>
      <path d="M670 152 C 710 152 726 104 756 103" stroke="#33507a" fill="none"/>
      <circle cx="772" cy="102" r="15" fill="#13253f" stroke="#4a7ebb" stroke-width="2"/>
      <text x="764" y="107" font-family="ui-monospace,Menlo,monospace" font-size="12" fill="#eef4fb">1&#215;</text>
      <g>
        <rect x="806" y="70" width="98" height="9" rx="3" fill="#4a7ebb"/>
        <rect x="806" y="92" width="61" height="9" rx="3" fill="#4a7ebb" opacity="0.65"/>
        <rect x="806" y="114" width="34" height="9" rx="3" fill="#4a7ebb" opacity="0.4"/>
        <rect x="806" y="136" width="16" height="9" rx="3" fill="#4a7ebb" opacity="0.25"/>
      </g>
      <text x="806" y="166" font-family="ui-monospace,Menlo,monospace" font-size="10" fill="#5f7ea6">P(answer) per question</text>
      <rect x="24" y="186" width="620" height="4" rx="2" fill="url(#acc1)"/>
    </svg>
    </div>
    """)
    return


@app.cell(hide_code=True)
def goal(mo):
    mo.md("""
    In this notebook, we will show how Laya performs on a simple task: tagging
    and classifying support emails.

    By the end of the notebook, you will:

    - see the Laya model in action in a zero-shot setting, and
    - see the code that is necessary to run the model zero-shot.
    """)
    return


@app.cell(hide_code=True)
def pre_load(mo):
    mo.md("""
    ## First, the model

    Everything below runs on `convaiinnovations/laya`, an open-weights decision
    model from ConvAI Innovations (Apache 2.0). Under the hood it is a
    ModernBERT-large encoder with a decision head: instead of generating the
    next token, it scores answer options you define at request time. The first
    run downloads the checkpoint (about 840 MB); after that it loads from
    cache. One forward pass for all four questions is about to matter, so the
    timer in a later cell is worth watching.
    """)
    return


@app.cell(hide_code=True)
def load_model(contextlib, laya, mo):
    load_ctx1 = (
        mo.spinner("loading convaiinnovations/laya ...")
        if hasattr(mo, "spinner")
        else contextlib.nullcontext()
    )
    with load_ctx1:
        agent1 = laya.load("convaiinnovations/laya")

    n_params1 = sum(p.numel() for p in agent1.model.parameters())
    return agent1, n_params1


@app.cell(hide_code=True)
def model_card(agent1, mo, n_params1):
    mo.Html(
        f"""
        <div style="display:flex;flex-wrap:wrap;gap:30px;align-items:center;padding:16px 22px;border:1px solid #22364f;border-radius:12px;background:#0e1a2b;font-family:ui-monospace,Menlo,monospace;">
          <div><div style="font-size:10px;letter-spacing:2px;color:#7f9cc0;">CHECKPOINT</div><div style="font-size:14px;color:#eef4fb;">convaiinnovations/laya</div></div>
          <div><div style="font-size:10px;letter-spacing:2px;color:#7f9cc0;">PARAMETERS</div><div style="font-size:14px;color:#eef4fb;">{n_params1 / 1e6:.0f}M</div></div>
          <div><div style="font-size:10px;letter-spacing:2px;color:#7f9cc0;">DEVICE</div><div style="font-size:14px;color:#eef4fb;">{agent1.device}</div></div>
          <div><div style="font-size:10px;letter-spacing:2px;color:#7f9cc0;">CONTEXT BUDGET</div><div style="font-size:14px;color:#eef4fb;">{agent1.cfg.get("max_len")} tokens / question</div></div>
          <div style="margin-left:auto;"><div style="font-size:10px;letter-spacing:2px;color:#7f9cc0;">LICENSE</div><div style="font-size:14px;color:#4ade80;">Apache 2.0</div></div>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def model_card_reading(mo):
    mo.md("""
    Read the card. The number that matters is **421M parameters**, roughly a
    thousandth of a frontier LLM, small enough that every inference in this
    notebook runs on a laptop CPU. The 512-token budget is per question and it
    is configurable at call time, which becomes a plot point later in the
    series.
    """)
    return


@app.cell(hide_code=True)
def schema_intro(mo):
    mo.md("""
    ## The email and the schema

    A decision model works in two parts: a **state** (here, an email) and a
    **schema** of typed questions. Three primitives cover most triage work:

    - `choice` picks one option from a list you define at request time and
      returns the full probability distribution over every option.
    - `score` places the email on an ordinal rubric you describe in words.
    - `noul` returns a calibrated probability that a yes/no statement holds.

    Our support-triage schema asks four questions in one pass: which department
    (choice), how urgent (score), is this phishing (noul), and is this customer
    about to churn (noul). The schema is code you write; the model fills in the
    answers.
    """)
    return


@app.cell(hide_code=True)
def email_and_schema():
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


@app.cell(hide_code=True)
def schema_table(mo):
    mo.md("""
    The schema we hand the model is as follows:

    | primitive | question | options |
    |---|---|---|
    | `choice` | which department? | billing · technical · sales · other |
    | `score` | how urgent? | 0 not urgent to 2 critical |
    | `noul` | is this a phishing attempt? | P(yes) |
    | `noul` | will this customer churn? | P(yes) |

    Nothing here is a prompt. Each question is a typed declaration, and the
    model answers all four in one forward pass.
    """)
    return


@app.cell(hide_code=True)
def email_setup(mo):
    mo.md("""
    Here is the email, in the inbox. Read it the way a triager would: who is
    this, what do they want, how bad is it, and is anyone at risk?
    """)
    return


@app.cell(hide_code=True)
def email_preview(EMAIL, MailViewer):
    MailViewer(
        emails=[
            {
                "sender_name": "Maya Chen",
                "sender_email": "maya@northwind-clinic.com",
                "initials": "MC",
                "subject": EMAIL["subject"],
                "body": EMAIL["body"],
                "time": "8:58 AM",
                "preview": EMAIL["body"][:70],
                "answers": None,
            }
        ]
    )
    return


@app.cell(hide_code=True)
def run_intro(mo):
    mo.vstack(
        [
            mo.md(
                """
                ## Run it

                One call sends the email and the schema through the model, and a
                single forward pass scores every answer option at once.
                """
            ),
            mo.accordion(
                {
                    "Click here to see the code that runs Laya": mo.md(
                        """
                        ```python
                        t01 = time.perf_counter()
                        result1 = agent1.predict(EMAIL, TRIAGE_Q)
                        elapsed_ms1 = (time.perf_counter() - t01) * 1000
                        a1 = result1["answers"]
                        ```
                        """
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def run_inference(EMAIL, TRIAGE_Q, agent1, contextlib, mo, time):
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


@app.cell(hide_code=True)
def results_viewer(EMAIL, MailViewer, a1, elapsed_ms1, mo):
    mo.vstack(
        [
            mo.md(
                f"""
                ## What came back: {elapsed_ms1:.0f} ms, four answers, zero generated text
                """
            ),
            MailViewer(
                emails=[
                    {
                        "sender_name": "Maya Chen",
                        "sender_email": "maya@northwind-clinic.com",
                        "initials": "MC",
                        "subject": EMAIL["subject"],
                        "body": EMAIL["body"],
                        "time": "8:58 AM",
                        "answers": a1,
                        "ms": round(elapsed_ms1),
                    }
                ]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def results_reading(mo):
    mo.md("""
    Read the chips.

    - **Department** came back `billing` with most of the probability mass. A
      duplicate charge is about as clean a billing case as they come.
    - **Urgency** landed mid-scale: there is a deadline ("this week"), but
      nobody is bleeding.
    - **P(phishing)** sits near zero. Nothing here asks for credentials or a
      click.
    - **P(churn)** is the number an account-management queue would page on:
      the customer names the exit door.

    Four answers, one forward pass, each with its uncertainty attached.
    """)
    return


@app.cell(hide_code=True)
def dept_plot_intro(mo):
    mo.md("""
    The choice head does not just pick `billing`; it distributes probability
    over all four options. Here is the full distribution, which is what the
    confidence number summarizes.
    """)
    return


@app.cell(hide_code=True)
def dept_plot(a1, go):
    dept_probs1 = a1["department"]["probabilities"]
    order1 = sorted(dept_probs1, key=dept_probs1.get)
    fig_dept1 = go.Figure(
        go.Bar(
            x=[dept_probs1[k] for k in order1],
            y=order1,
            orientation="h",
            marker_color="#4a7ebb",
            text=[f"{dept_probs1[k]:.3f}" for k in order1],
            textposition="outside",
        )
    )
    fig_dept1.update_layout(
        title=dict(
            text="P(department) - the full choice distribution",
            font=dict(size=13),
        ),
        height=280,
        margin=dict(l=10, r=50, t=46, b=10),
        template="plotly_white",
        xaxis=dict(range=[0, 1.12]),
    )
    fig_dept1
    return


@app.cell(hide_code=True)
def dept_plot_reading(mo):
    mo.md("""
    `billing` wins with most of the probability mass; the other three options
    split the remainder. When two options split the mass instead, the model is
    telling you the email sits on a boundary, and a confidence gate should send
    it to a human.
    """)
    return


@app.cell(hide_code=True)
def urgency_plot_intro(mo):
    mo.md("""
    The score head works differently: it places the email on the 0-2 urgency
    rubric and shows its mass per level.
    """)
    return


@app.cell(hide_code=True)
def urgency_plot(a1, go):
    urg1 = a1["urgency"]["probabilities"]
    fig_urg1 = go.Figure(
        go.Bar(
            x=["not urgent", "soon", "critical"],
            y=[
                float(urg1.get("0", 0)),
                float(urg1.get("1", 0)),
                float(urg1.get("2", 0)),
            ],
            marker_color="#4ade80",
            text=[
                f"{float(urg1.get('0', 0)):.2f}",
                f"{float(urg1.get('1', 0)):.2f}",
                f"{float(urg1.get('2', 0)):.2f}",
            ],
            textposition="outside",
        )
    )
    fig_urg1.update_layout(
        title=dict(
            text="P(urgency) - score head, mean "
            + f"{a1['urgency']['score']:.2f}",
            font=dict(size=13),
        ),
        height=260,
        margin=dict(l=10, r=50, t=46, b=10),
        template="plotly_white",
        yaxis=dict(range=[0, 0.7]),
    )
    fig_urg1
    return


@app.cell(hide_code=True)
def urgency_plot_reading(mo):
    mo.md("""
    Most of the mass sits on level 1, "soon", with the remainder split between
    the extremes. A duplicate charge with a one-week fuse is a "soon", not a
    page-the-on-call "critical", and the head agrees. Ordinal scores like this
    are handy for sorting queues without hand-inventing numeric urgency
    policies.
    """)
    return


@app.cell(hide_code=True)
def nothing_to_parse(mo):
    mo.md("""
    ## What just happened, in one line

    The email went in as state, the schema fixed the shape of the answer, and
    the outputs arrived as numbers and labels: no JSON to repair, nothing to
    hallucinate, and all four questions answered in one forward pass.
    """)
    return


@app.cell(hide_code=True)
def sweep_intro(mo):
    mo.md("""
    ## Five emails, same schema

    One email is a demo; an inbox is the job. These five hit the same support
    address in one morning: a production outage, a payment dispute, a happy
    customer, a phishing blast, and a business-email-compromise attempt that
    looks like routine accounts-payable housekeeping. Click through the inbox
    on the left; the chips in the reader pane are the model's answers.
    """)
    return


@app.cell(hide_code=True)
def sweep_emails(TRIAGE_Q, agent1, time):
    EMAILS1 = [
        {
            "key": "prod outage",
            "sender_name": "Engineering On-Call",
            "sender_email": "oncall@acme-health.com",
            "initials": "EO",
            "subject": "Production down - checkout failing for all customers",
            "body": "Since 9am UTC every checkout returns a 500 error. We are losing orders every minute this stays broken. Please treat this as a P1.",
            "time": "9:41 AM",
            "unread": True,
        },
        {
            "key": "BEC scam",
            "sender_name": "Vendor Finance Desk",
            "sender_email": "ap-updates@vendor-portal.example",
            "initials": "VF",
            "subject": "Re: Q3 invoice - updated banking details",
            "body": "Hi, following our conversation last week, please note our accounts payable email has changed. Kindly update your records and process the attached invoice to the new details by Friday.",
            "time": "9:07 AM",
            "unread": True,
        },
        {
            "key": "duplicate charge",
            "sender_name": "Maya Chen",
            "sender_email": "maya@northwind-clinic.com",
            "initials": "MC",
            "subject": "Duplicate charge on invoice 4411",
            "body": "We were billed twice for March. Please refund the duplicate. If this isn't fixed this week we will cancel our account.",
            "time": "8:58 AM",
        },
        {
            "key": "happy customer",
            "sender_name": "Devon Park",
            "sender_email": "devon@parkanalytics.io",
            "initials": "DP",
            "subject": "Loving the new dashboard!",
            "body": "Just wanted to say the new analytics dashboard is fantastic. Any chance you'll add CSV export next quarter? No rush at all.",
            "time": "8:15 AM",
        },
        {
            "key": "obvious phishing",
            "sender_name": "Account Security",
            "sender_email": "security@paypal-secure-verify.example-login.com",
            "initials": "AS",
            "subject": "URGENT: Your account will be suspended",
            "body": "Dear customer, we detected unusual activity. Click http://paypal-secure-verify.example-login.com immediately to confirm your password and SSN, or your account will be permanently deleted within 24 hours.",
            "time": "7:30 AM",
            "unread": True,
        },
    ]

    def clean_answers1(ax1):
        return {
            "department": {
                "choice": ax1["department"]["choice"],
                "confidence": float(ax1["department"]["confidence"]),
            },
            "urgency": {"score": float(ax1["urgency"]["score"])},
            "is_phishing": {"noul": float(ax1["is_phishing"]["noul"])},
            "churn_risk": {"noul": float(ax1["churn_risk"]["noul"])},
        }

    MAIL_DATA1 = []
    for m1 in EMAILS1:
        t1 = time.perf_counter()
        r1 = agent1.predict(
            {"subject": m1["subject"], "body": m1["body"]}, TRIAGE_Q
        )["answers"]
        ms1 = round((time.perf_counter() - t1) * 1000)
        MAIL_DATA1.append(
            {
                "sender_name": m1["sender_name"],
                "sender_email": m1["sender_email"],
                "initials": m1["initials"],
                "subject": m1["subject"],
                "body": m1["body"],
                "preview": m1["body"][:70],
                "time": m1["time"],
                "unread": bool(m1.get("unread", False)),
                "ms": ms1,
                "answers": clean_answers1(r1),
            }
        )
    return (MAIL_DATA1,)


@app.cell(hide_code=True)
def inbox_viewer(MAIL_DATA1, MailViewer):
    MailViewer(emails=MAIL_DATA1, selected=1)
    return


@app.cell(hide_code=True)
def sweep_reading(mo):
    mo.md("""
    Click through and the pattern holds: the outage routes to technical at high
    urgency, the phishing blast scores P(phishing) near 1.0, the happy customer
    files under sales.

    The banking-details email is the one to study. It reads like routine
    housekeeping and the model scored it that way; it is actually business
    email compromise, the most expensive email fraud on the books. The model
    could not see it, and its confidence said nothing. Hold that thought; the
    series keeps coming back to it.
    """)
    return


@app.cell(hide_code=True)
def llm_contrast(mo):
    mo.md("""
    ## The same job with an LLM

    To do this with an LLM, you prompt it to return JSON, generate the tokens,
    parse the output, and retry when the JSON is malformed. That is seconds per
    email instead of milliseconds, and the probabilities are not native; you
    sample or you trust prose.

    Here, one forward pass per email, and the probabilities are the native
    output, not a sampled string you have to trust.

    But should you trust those confidence numbers? That is what notebook 2 is
    all about: feed this model a language it has never seen, and watch the
    confidence score stay perfectly silent.
    """)
    return


if __name__ == "__main__":
    app.run()
