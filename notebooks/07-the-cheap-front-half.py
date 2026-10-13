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
"""Act 7: The cheap front half.

Wrap the English checkpoint as a serverless API on Modal, scale to zero,
deploy only from CI. Two profiles measured: CPU and T4 GPU. The economics say
batch triage belongs on CPU; the GPU buys latency, not cost. And the closing
example shows why the cheap fast front half still needs a System 2 behind it.
Measurements are inlined from the deployment benchmarks in nll-ai/laya-demo.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def banner(CRIMSON, GREEN, INK, INK2, KICK, MIST, SLATE, TEAL, mo):
    mo.Html(
        f"""
        <div style="line-height:0;border-radius:16px;overflow:hidden;">
        <svg viewBox="0 0 960 210" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Notebook 7 - The deployment bill">
          <defs>
            <linearGradient id="bg7" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stop-color="{INK}"/><stop offset="1" stop-color="{INK2}"/>
            </linearGradient>
            <linearGradient id="acc7" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0" stop-color="{TEAL}"/><stop offset="0.55" stop-color="{GREEN}"/><stop offset="1" stop-color="{CRIMSON}"/>
            </linearGradient>
          </defs>
          <rect x="1" y="1" width="958" height="208" rx="16" fill="url(#bg7)" stroke="#24404F"/>
          <text x="700" y="185" font-family="ui-monospace,Menlo,monospace" font-size="170" font-weight="700" fill="none" stroke="#26404F" stroke-width="2" opacity="0.5">07</text>
          <rect x="798" y="34" width="122" height="26" rx="13" fill="{INK2}" stroke="#2E5566"/>
          <text x="813" y="51" font-family="ui-monospace,Menlo,monospace" font-size="11" letter-spacing="2" fill="{KICK}">NOTEBOOK 7 OF 7</text>
          <text x="40" y="54" font-family="ui-monospace,Menlo,monospace" font-size="12" letter-spacing="3.5" fill="{KICK}">THE LAYA PHARMA SURVEILLANCE SERIES</text>
          <text x="40" y="112" font-family="-apple-system,Segoe UI,sans-serif" font-size="42" font-weight="800" fill="#EEF4FB">The deployment bill</text>
          <text x="40" y="148" font-family="ui-monospace,Menlo,monospace" font-size="13.5" fill="{MIST}">CPU wins on cost &#183; the GPU buys latency &#183; a System 2 stays on the funnel</text>
          <g font-family="ui-monospace,Menlo,monospace" font-size="11">
            <rect x="640" y="52" width="74" height="21" rx="6" fill="{INK2}" stroke="#2E5566"/><text x="650" y="66" fill="{KICK}">cost</text>
            <rect x="640" y="82" width="74" height="21" rx="6" fill="{INK2}" stroke="#2E5566"/><text x="650" y="96" fill="{KICK}">latency</text>
            <rect x="640" y="112" width="74" height="21" rx="6" fill="{INK2}" stroke="#2E5566"/><text x="650" y="126" fill="{KICK}">scale 0</text>
            <rect x="640" y="142" width="74" height="21" rx="6" fill="{INK2}" stroke="#2E5566"/><text x="650" y="156" fill="{KICK}">ci gate</text>
          </g>
          <path d="M714 62 C 736 62 746 100 756 101" stroke="#2E5566" fill="none"/>
          <path d="M714 92 C 732 92 744 101 756 101" stroke="#2E5566" fill="none"/>
          <path d="M714 122 C 732 122 744 103 756 103" stroke="#2E5566" fill="none"/>
          <path d="M714 152 C 736 152 746 104 756 103" stroke="#2E5566" fill="none"/>
          <circle cx="772" cy="102" r="15" fill="{INK}" stroke="{TEAL}" stroke-width="2"/>
          <text x="764" y="107" font-family="ui-monospace,Menlo,monospace" font-size="12" fill="#EEF4FB">7&#215;</text>
          <g>
            <rect x="806" y="82" width="98" height="9" rx="3" fill="{TEAL}"/>
            <rect x="806" y="112" width="14" height="9" rx="3" fill="{SLATE}"/>
          </g>
          <text x="806" y="166" font-family="ui-monospace,Menlo,monospace" font-size="10" fill="#6E93A0">p50 seconds by profile</text>
          <rect x="24" y="186" width="620" height="4" rx="2" fill="url(#acc7)"/>
        </svg>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def goal(mo):
    mo.md("""
    In this notebook, we will show what it costs to serve the English
    checkpoint as a serverless API, and why the cheap profile is the right
    home for batch triage.

    By the end of the notebook, you will:

    - see the two measured deployment profiles, CPU and T4 GPU, side by
      side,
    - see the one line of arithmetic that turns an hourly rate and a
      throughput into a cost per million requests, and
    - see the business email compromise attempt that the model waves
      straight through, the reason a fast, cheap model
      still needs a System 2 behind it.
    """)
    return


@app.cell(hide_code=True)
def pre_imports(mo):
    mo.md("""
    ## Benchmarks, inlined

    Notebooks 1 through 6 ran on a laptop. This notebook prices the model
    as a service: the English checkpoint behind an HTTP endpoint on Modal.
    The image ships with the weights baked in, both profiles scale to zero
    after 60 seconds idle, and CI gates every deploy with smoke tests. I
    measured every number below with the benchmark harness in
    `nll-ai/laya-demo/modal/benchmark.py` and inlined them as constants.
    """)
    return


@app.cell(hide_code=True)
def imports():
    import json
    import sys
    from pathlib import Path

    import marimo as mo
    import plotly.graph_objects as go

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mail_viewer import MailViewer
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
    DLR = "\$"  # escaped dollar sign: a bare $...$ pair renders as math in markdown
    return (
        CRIMSON,
        DLR,
        GREEN,
        INK,
        INK2,
        KICK,
        MIST,
        MailViewer,
        Path,
        SLATE,
        TEAL,
        go,
        json,
        make_subplots,
        mo,
    )


@app.cell(hide_code=True)
def profiles_intro(mo):
    mo.md("""
    ## Two profiles, measured

    The deployment is serverless. A Modal app wraps the English checkpoint
    behind one POST endpoint and shuts each container down when traffic stops,
    so the bill tracks requests rather than machines. I benchmarked two hardware profiles
    on the same endpoint: a CPU profile with 1 core and 3 GiB of
    memory, and a T4 GPU profile with 2 cores and 8 GiB. The next cell lays the
    measurements side by side: cold start, median end-to-end latency,
    server-side inference time, sustained throughput, and the billable hourly
    rate.
    """)
    return


@app.cell(hide_code=True)
def endpoint_call(mo):
    mo.vstack(
        [
            mo.md(
                """
                Calling the endpoint is just a POST request. You send the email
                and the questions as JSON, and you get the answers back as JSON:
                """
            ),
            mo.accordion(
                {
                    "Click here to see how to call the endpoint": mo.md(
                        """
                        ```python
                        import json
                        import urllib.request

                        URL = "https://<workspace>--laya-api-<endpoint>.modal.run"

                        payload = {
                            "state": {"subject": "...", "body": "..."},
                            "questions": {
                                "is_phishing": {
                                    "type": "noul",
                                    "instructions": "Is this email a phishing or scam attempt?",
                                }
                            },
                        }
                        request = urllib.request.Request(
                            URL,
                            data=json.dumps(payload).encode(),
                            headers={"Content-Type": "application/json"},
                        )
                        result = json.loads(urllib.request.urlopen(request, timeout=60).read())
                        ```
                        """
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def benchmarks(Path, SLATE, TEAL, json, make_subplots):
    DATA7 = Path(__file__).resolve().parent.parent / "data"
    with open(DATA7 / "modal_benchmarks.json", encoding="utf-8") as f7:
        _raw7 = json.load(f7)["profiles"]

    def _normalize7(raw7, name7, thr_key7):
        normalized7 = dict(raw7)
        normalized7["name"] = name7
        normalized7["cold start s"] = normalized7.pop("cold_start_s")
        normalized7["p50 s"] = normalized7.pop("p50_s")
        normalized7["server inference s"] = normalized7.pop("server_inference_s")
        normalized7["throughput req/s"] = normalized7.pop(thr_key7)
        normalized7["$ per hour"] = normalized7.pop("usd_per_hour")
        return normalized7

    CPU7 = _normalize7(_raw7["cpu"], "CPU", "throughput_req_s_serial")
    GPU7 = _normalize7(_raw7["gpu_t4"], "T4 GPU", "throughput_req_s_8_concurrent")

    fig_prof4 = make_subplots(
        rows=5,
        cols=1,
        subplot_titles=(
            "cold start (seconds)",
            "median end-to-end latency, p50 (seconds)",
            "server-side inference time (seconds)",
            "sustained throughput (req/s)",
            "billable rate ($/hour)",
        ),
    )
    for row7, key7 in enumerate(
        [
            "cold start s",
            "p50 s",
            "server inference s",
            "throughput req/s",
            "$ per hour",
        ],
        start=1,
    ):
        fig_prof4.add_bar(
            y=["CPU", "T4 GPU"],
            x=[CPU7[key7], GPU7[key7]],
            orientation="h",
            marker_color=[SLATE, TEAL],
            text=[f"{CPU7[key7]:g}", f"{GPU7[key7]:g}"],
            textposition="outside",
            row=row7,
            col=1,
        )

    fig_prof4.update_layout(
        height=760,
        margin=dict(l=10, r=60, t=70, b=10),
        template="plotly_white",
        showlegend=False,
    )
    fig_prof4.update_xaxes(range=[0, None])
    fig_prof4
    return CPU7, GPU7


@app.cell(hide_code=True)
def profiles_reading(CPU7, DLR, GPU7, mo):
    mo.md(f"""
    CPU against T4, measurement by measurement:

    - **Cold start**: {CPU7["cold start s"]:.1f} s on CPU against
      {GPU7["cold start s"]:.1f} s on the T4, about
      {GPU7["cold start s"] / CPU7["cold start s"]:.0f} times longer to wait for
      the first byte. The GPU image is heavier and the hardware is rarer.
    - **p50 latency**: {GPU7["p50 s"]:.2f} s on the T4 against
      {CPU7["p50 s"]:.2f} s on CPU. here the picture flips.
    - **Server-side inference**: {GPU7["server inference s"] * 1000:.0f} ms
      inside the model on the T4 against {CPU7["server inference s"]:.1f} s on
      CPU.
    - **Throughput and bill**: {GPU7["throughput req/s"]:.1f} req/s against
      {CPU7["throughput req/s"]:.1f} req/s, at {DLR}{GPU7["$ per hour"]:.3f}/h
      against {DLR}{CPU7["$ per hour"]:.3f}/h, about
      {GPU7["$ per hour"] / CPU7["$ per hour"]:.0f} times the hourly rate.

    Both rows come from one benchmark harness, so everything that follows is
    arithmetic on these numbers.
    """)
    return


@app.cell(hide_code=True)
def cost_intro(mo):
    mo.md("""
    ## What a million requests cost

    The arithmetic is simple: take the hourly rate, divide by the
    throughput, and scale up to a million requests.
    """)
    return


@app.cell(hide_code=True)
def cost_per_million(CPU7, DLR, GPU7, mo):
    def cost_per_million7(rate7, req_s7):
        # $ per 1M requests = ($/h) / (req/s) * 1e6 / 3600
        return rate7 / req_s7 * 1e6 / 3600

    cpu_cost7 = cost_per_million7(CPU7["$ per hour"], CPU7["throughput req/s"])
    gpu_cost7 = cost_per_million7(GPU7["$ per hour"], GPU7["throughput req/s"])
    mo.md(
        f"""
        At full utilization, the whole arithmetic per profile:

        - CPU: {DLR}{CPU7["$ per hour"]:.3f}/h divided by
          {CPU7["throughput req/s"]:.1f} req/s, which is
          **{DLR}{cpu_cost7:.0f} per 1M requests**
        - T4 GPU: {DLR}{GPU7["$ per hour"]:.3f}/h divided by
          {GPU7["throughput req/s"]:.1f} req/s, which is
          **{DLR}{gpu_cost7:.0f} per 1M requests**
        """
    )
    return cpu_cost7, gpu_cost7


@app.cell(hide_code=True)
def cost_plot_intro(DLR, cpu_cost7, gpu_cost7, mo):
    mo.md(f"""
    Plug the numbers in and a million requests cost about
    {DLR}{cpu_cost7:.0f} on CPU versus {DLR}{gpu_cost7:.0f} on the T4:
    """)
    return


@app.cell(hide_code=True)
def cost_plot(SLATE, TEAL, cpu_cost7, go, gpu_cost7):
    fig_cost7 = go.Figure(
        go.Bar(
            x=["CPU (1 core, 3 GiB)", "T4 GPU (2 cores, 8 GiB)"],
            y=[cpu_cost7, gpu_cost7],
            marker_color=[TEAL, SLATE],
            text=[f"${cpu_cost7:.0f}", f"${gpu_cost7:.0f}"],
            textposition="outside",
        )
    )
    fig_cost7.update_layout(
        title=dict(
            text="batch triage: $ per 1M requests at full utilization",
            font=dict(size=13),
        ),
        height=300,
        margin=dict(l=10, r=20, t=46, b=10),
        template="plotly_white",
        yaxis=dict(title="$ per 1M requests", range=[0, gpu_cost7 * 1.2]),
    )
    fig_cost7
    return


@app.cell(hide_code=True)
def cost_plot_reading(DLR, cpu_cost7, gpu_cost7, mo):
    mo.md(f"""
    At full utilization, a million requests cost about
    {DLR}{cpu_cost7:.0f} on CPU against {DLR}{gpu_cost7:.0f} on the T4.
    The next chart plots the monthly bill as request volume grows.
    """)
    return


@app.cell(hide_code=True)
def cost_crossover(SLATE, TEAL, cpu_cost7, go, gpu_cost7):
    cpu_per_req7 = cpu_cost7 / 1e6
    gpu_per_req7 = gpu_cost7 / 1e6
    volumes7 = [0, 250e3, 500e3, 1e6, 2e6, 4e6, 8e6]

    fig_cross4 = go.Figure()
    fig_cross4.add_trace(
        go.Scatter(
            x=volumes7,
            y=[v7 * cpu_per_req7 for v7 in volumes7],
            mode="lines+markers",
            name="CPU",
            line=dict(color=TEAL, width=3),
        )
    )
    fig_cross4.add_trace(
        go.Scatter(
            x=volumes7,
            y=[v7 * gpu_per_req7 for v7 in volumes7],
            mode="lines+markers",
            name="T4 GPU",
            line=dict(color=SLATE, width=3),
        )
    )
    fig_cross4.update_layout(
        title=dict(text="monthly bill vs monthly requests", font=dict(size=13)),
        height=380,
        margin=dict(l=10, r=20, t=46, b=10),
        template="plotly_white",
        xaxis=dict(title="requests per month", tickformat=",.0f"),
        yaxis=dict(title="dollars per month", tickprefix="$"),
    )
    fig_cross4
    return cpu_per_req7, gpu_per_req7


@app.cell(hide_code=True)
def cost_crossover_reading(DLR, cpu_per_req7, gpu_per_req7, mo):
    mo.md(f"""
    The cost lines never cross. Per request, CPU costs about
    {DLR}{cpu_per_req7 * 1000:.3f} per thousand and the T4 about
    {DLR}{gpu_per_req7 * 1000:.3f} per thousand, and both scale linearly,
    so batch triage is cheaper on CPU at every volume. The GPU is a
    latency purchase, not a cost decision: buy it when interactive
    response time matters, not when volume does.
    """)
    return


@app.cell(hide_code=True)
def cost_reading(DLR, cpu_cost7, gpu_cost7, mo):
    mo.md(f"""
    The ordering is not close, and it never flips. At full utilization the
    T4 costs about {gpu_cost7 / cpu_cost7:.1f} times as much per request as
    the CPU profile. Below full utilization the ranges widen, because idle
    seconds land on the bill: the blog post measured {DLR}17-68 per 1M for
    CPU and {DLR}26-115 per 1M for T4 across real concurrency levels.
    Stretch the ranges as wide as you like; at every concurrency measured,
    batch triage is cheaper on the CPU profile. Cost is not the reason
    anyone buys the GPU.
    """)
    return


@app.cell(hide_code=True)
def cost_crossover_intro(mo):
    mo.md("""
    ## The monthly bill at scale

    What does the difference mean at real volumes? The next chart plots
    the monthly bill as request count grows.
    """)
    return


@app.cell(hide_code=True)
def latency_intro(mo):
    mo.md("""
    ## Why would we consider using the GPU?

    The real answer is latency. The same benchmark ran the same request
    against both profiles and timed two things: the p50 a caller
    experiences end to end, and the time the server spends inside the
    model. The next figure puts both on a log scale, because the two
    profiles live an order of magnitude apart.
    """)
    return


@app.cell(hide_code=True)
def latency_plot(CPU7, GPU7, SLATE, TEAL, go):
    names7 = ["CPU (1 core, 3 GiB)", "T4 GPU (2 cores, 8 GiB)"]
    fig_lat7 = go.Figure()
    fig_lat7.add_bar(
        name="p50, end to end",
        x=names7,
        y=[CPU7["p50 s"], GPU7["p50 s"]],
        marker_color=TEAL,
        text=[f"{CPU7['p50 s']:.2f} s", f"{GPU7['p50 s']:.2f} s"],
        textposition="outside",
    )
    fig_lat7.add_bar(
        name="server-side inference only",
        x=names7,
        y=[CPU7["server inference s"], GPU7["server inference s"]],
        marker_color=SLATE,
        text=[
            f"{CPU7['server inference s']:.1f} s",
            f"{GPU7['server inference s'] * 1000:.0f} ms",
        ],
        textposition="outside",
    )
    fig_lat7.update_layout(
        barmode="group",
        title=dict(
            text="the T4 buys latency, not cost (log scale)",
            font=dict(size=13),
        ),
        height=340,
        margin=dict(l=10, r=20, t=60, b=10),
        template="plotly_white",
        yaxis=dict(title="seconds (log scale)", type="log"),
    )
    fig_lat7
    return


@app.cell(hide_code=True)
def latency_reading(CPU7, DLR, GPU7, mo):
    mo.md(f"""
    The median request finishes in {GPU7["p50 s"]:.2f} s on the T4 against
    {CPU7["p50 s"]:.2f} s on CPU: a {CPU7["p50 s"] / GPU7["p50 s"]:.0f}x
    latency win, which is the entire sales pitch for the GPU profile. Inside
    the model the gap is wider still,
    {GPU7["server inference s"] * 1000:.0f} ms against
    {CPU7["server inference s"]:.1f} s. That is what
    {DLR}{GPU7["$ per hour"]:.3f}/h buys: responsiveness for interactive
    traffic. The cost ordering from the last figure survives untouched, so
    the honest summary is the tagline: the CPU wins on cost, and the GPU
    buys latency.
    """)
    return


@app.cell(hide_code=True)
def ops_intro(CPU7, GPU7, mo):
    mo.md(f"""
    ## Cold starts and the shape of the bill

    Both profiles scale to zero after 60 seconds idle, which is what makes
    the CPU profile nearly free for sporadic traffic: a quiet overnight
    surveillance feed bills for almost nothing and wakes in
    {CPU7["cold start s"]:.1f} s when a post arrives. The T4 pays for its
    latency with a {GPU7["cold start s"]:.1f} s cold start, so it fits
    steady interactive traffic rather than occasional batch.

    Deploys run only from GitHub Actions on the `nll-ai` workspace, and CI
    smoke-tests both live profiles before declaring a deploy green. A push
    to `main` is the whole ops story.
    """)
    return


@app.cell(hide_code=True)
def bec_intro(mo):
    mo.md("""
    ## Where the fast, cheap deployment breaks

    The endpoint is fast and cheap. Now let's point it at one specific
    email and see how it does. This is one of the emails from the
    earlier notebooks, scored with the phishing question:
    """)
    return


@app.cell(hide_code=True)
def bec_load(mo):
    import contextlib
    import os

    os.environ.setdefault("USE_TF", "0")

    import laya

    bec_load_ctx7 = (
        mo.spinner("loading the English checkpoint (cached after Notebook 1) ...")
        if hasattr(mo, "spinner")
        else contextlib.nullcontext()
    )
    with bec_load_ctx7:
        agent7 = laya.load("convaiinnovations/laya")
    return (agent7,)


@app.cell(hide_code=True)
def bec_beat(MailViewer, agent7):
    BEC_SUBJECT7 = "Re: Q3 invoice - updated banking details"
    BEC_FROM7 = "Vendor Finance Desk <ap-updates@vendor-portal.example>"
    BEC_BODY7 = (
        "Hi, following our conversation last week, please note our accounts "
        "payable email has changed. Kindly update your records and process the "
        "attached invoice to the new details by Friday."
    )

    phish_q7 = {
        "is_phishing": {
            "type": "noul",
            "instructions": "Is this email a phishing or scam attempt?",
        }
    }
    r_bec7 = agent7.predict({"subject": BEC_SUBJECT7, "body": BEC_BODY7}, phish_q7)[
        "answers"
    ]
    BEC_PHISH7 = float(r_bec7["is_phishing"]["noul"])

    MailViewer(
        emails=[
            {
                "sender_name": "Vendor Finance Desk",
                "sender_email": "ap-updates@vendor-portal.example",
                "initials": "VF",
                "subject": BEC_SUBJECT7,
                "body": BEC_BODY7,
                "time": "9:07 AM",
                "answers": {"is_phishing": {"noul": BEC_PHISH7}},
                "note": "Scored live by this notebook: English checkpoint, phishing question, one forward pass.",
            }
        ]
    )
    return (BEC_PHISH7,)


@app.cell(hide_code=True)
def bec_reading(BEC_PHISH7, mo):
    mo.md(f"""
    The model scored it **P(phishing) = {BEC_PHISH7:.3f}** - completely
    safe.

    Read it again. Someone impersonating one of our vendors is asking us
    to send payments to a new bank account. No links, no "urgent!!!"
    subject line, nothing to trip a filter - and that's exactly why this
    kind of scam works. It looks like something the finance team gets
    every week.

    So for a few dollars a day, this deployment will triage the inbox all
    day long, and it will still wave this email straight through to the
    finance team. Catching it takes something slower - a person who knows
    the vendors, or a model that does.
    """)
    return


@app.cell(hide_code=True)
def series_recap(BEC_PHISH7, mo):
    mo.md(f"""
    ## The series, closed

    1. One forward pass answers typed questions in milliseconds.
    2. The confidence score is silent exactly when the model is wrong.
    3. One badly-phrased clause can invert a question head; schemas are
       code.
    4. Zero-shot covers the operational core on real documents.
    5. 212 labels and 20 minutes close the edges (0.26-0.58 to 0.89-0.96).
    6. Calibration is two steps: proper scoring rules, then a per-bucket
       refit, and it beats the closed competitor's shipped number.
    7. The cheap deployment wins on cost; latency is the only reason to buy
       the GPU.

    Laya earns its place as the fast, cheap, auditable first pass over an
    inbox. Catching what it misses - the scam that looks like bookkeeping,
    at P = {BEC_PHISH7:.3f} - still needs a System 2.
    """)
    return


if __name__ == "__main__":
    app.run()
