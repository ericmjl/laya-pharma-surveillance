# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo==0.25.1",
#     "plotly==7.1.0",
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
        <svg viewBox="0 0 960 210" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Act 7 - The cheap front half">
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
          <text x="813" y="51" font-family="ui-monospace,Menlo,monospace" font-size="11" letter-spacing="2" fill="{KICK}">ACT 7 OF 7</text>
          <text x="40" y="54" font-family="ui-monospace,Menlo,monospace" font-size="12" letter-spacing="3.5" fill="{KICK}">THE LAYA PHARMA SURVEILLANCE SERIES</text>
          <text x="40" y="112" font-family="-apple-system,Segoe UI,sans-serif" font-size="42" font-weight="800" fill="#EEF4FB">The cheap front half</text>
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
    checkpoint as a serverless API, and why the cheap profile is the right home
    for batch triage.

    By the end of the notebook, you will:

    - see the two measured deployment profiles, CPU and T4 GPU, side by side,
    - see the one line of arithmetic that turns an hourly rate and a throughput
      into a cost per million requests, and
    - see the business email compromise attempt that scored P(phishing) = 0.000,
      the reason the fast, cheap, auditable front half still needs a System 2
      behind it.
    """)
    return


@app.cell(hide_code=True)
def pre_imports(mo):
    mo.md("""
    ## Benchmarks, inlined

    Acts 1 through 6 ran on a laptop. This act prices the model as a service:
    the English checkpoint behind an HTTP endpoint on Modal, weights baked into
    the image, both profiles scaling to zero, deploys gated on CI smoke tests.
    Every number below was measured by the benchmark harness in
    `nll-ai/laya-demo/modal/benchmark.py` and is inlined here as constants, so
    this notebook loads no model and downloads nothing; the economics are the
    story and the numbers stay stable.
    """)
    return


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import plotly.graph_objects as go

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
    DLR = "\\$"  # escaped dollar sign: a bare $...$ pair renders as math in markdown
    return CRIMSON, DLR, GREEN, INK, INK2, KICK, MIST, SLATE, TEAL, go, mo


@app.cell(hide_code=True)
def profiles_intro(mo):
    mo.md("""
    ## Two profiles, measured

    The deployment is serverless. A Modal app wraps the English checkpoint
    behind one POST endpoint and shuts each container down when traffic stops,
    so the bill tracks requests rather than machines. Two hardware profiles were
    benchmarked on the same endpoint: a CPU profile with 1 core and 3 GiB of
    memory, and a T4 GPU profile with 2 cores and 8 GiB. The next cell lays the
    measurements side by side: cold start, median end-to-end latency,
    server-side inference time, sustained throughput, and the billable hourly
    rate.
    """)
    return


@app.cell(hide_code=True)
def benchmarks(mo):
    # Measured by the benchmark harness in nll-ai/laya-demo/modal/benchmark.py
    PROFILES7 = {
        "CPU (1 core, 3 GiB)": {
            "cold start s": 3.5,
            "p50 s": 3.31,
            "server inference s": 2.9,
            "throughput req/s": 1.2,
            "$ per hour": 0.071,
        },
        "T4 GPU (2 cores, 8 GiB)": {
            "cold start s": 44.3,
            "p50 s": 0.48,
            "server inference s": 0.049,
            "throughput req/s": 8.0,
            "$ per hour": 0.749,
        },
    }
    CPU7 = PROFILES7["CPU (1 core, 3 GiB)"]
    GPU7 = PROFILES7["T4 GPU (2 cores, 8 GiB)"]
    rows7 = [
        {
            "profile": name7,
            "cold start": f"{v7['cold start s']:.1f} s",
            "p50 latency": f"{v7['p50 s']:.2f} s",
            "server inference": (
                f"{v7['server inference s'] * 1000:.0f} ms"
                if v7["server inference s"] < 1
                else f"{v7['server inference s']:.1f} s"
            ),
            "throughput": f"{v7['throughput req/s']:.1f} req/s",
            "billable": f"${v7['$ per hour']:.3f}/h",
        }
        for name7, v7 in PROFILES7.items()
    ]
    mo.ui.table(rows7, selection=None)
    return CPU7, GPU7


@app.cell(hide_code=True)
def profiles_reading(CPU7, DLR, GPU7, mo):
    mo.md(f"""
    Read the table top to bottom.

    - **Cold start**: {CPU7["cold start s"]:.1f} s on CPU against
      {GPU7["cold start s"]:.1f} s on the T4, about
      {GPU7["cold start s"] / CPU7["cold start s"]:.0f} times longer to wait for
      the first byte. The GPU image is heavier and the hardware is scarcer.
    - **p50 latency**: {GPU7["p50 s"]:.2f} s on the T4 against
      {CPU7["p50 s"]:.2f} s on CPU. Here the picture flips; hold that for two
      cells.
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
    mo.vstack(
        [
            mo.md(
                """
                ## What a million requests cost

                One line of arithmetic turns a measured hourly rate and a
                measured throughput into a price per million requests: hourly
                rate divided by throughput, times a million, divided by the
                seconds in an hour.
                """
            ),
            mo.accordion(
                {
                    "Click here to see the one line of arithmetic behind these numbers": mo.md(
                        """
                        ```python
                        def cost_per_million7(rate7, req_s7):
                            return rate7 / req_s7 * 1e6 / 3600

                        cpu_cost7 = cost_per_million7(
                            CPU7["$ per hour"], CPU7["throughput req/s"]
                        )
                        gpu_cost7 = cost_per_million7(
                            GPU7["$ per hour"], GPU7["throughput req/s"]
                        )
                        ```
                        """
                    )
                }
            ),
        ]
    )
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
    mo.md(
        f"""
        Here is the price comparison the arithmetic implies: at full utilization,
        a million requests cost about {DLR}{cpu_cost7:.0f} on the CPU profile and
        about {DLR}{gpu_cost7:.0f} on the T4.
        """
    )
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
def cost_reading(DLR, cpu_cost7, gpu_cost7, mo):
    mo.md(
        f"""
        The ordering is not close, and it never flips. At full utilization the
        T4 costs about {gpu_cost7 / cpu_cost7:.1f} times as much per request as
        the CPU profile. Below full utilization the ranges widen, because idle
        seconds land on the bill: the blog post measured {DLR}17-68 per 1M for
        CPU and {DLR}26-115 per 1M for T4 across real concurrency levels.
        Stretch the ranges as wide as you like; at every concurrency measured,
        batch triage is cheaper on the CPU profile. Cost is not the reason
        anyone buys the GPU.
        """
    )
    return


@app.cell(hide_code=True)
def latency_intro(mo):
    mo.md("""
    ## The GPU buys latency, not cost

    If the CPU wins on cost, why does the GPU profile exist? Latency. The same
    benchmark ran the same request against both profiles and timed two things:
    the p50 a caller experiences end to end, and the time the server spends
    inside the model. The next figure puts both on a log scale, because the two
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
    mo.md(
        f"""
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
        """
    )
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
    ## Why the front half is not the whole pipeline

    Cost and latency describe the pipe, not the judgment inside it. Here is one
    email from Act 1's sweep, run through the English checkpoint's phishing
    question. Read it the way a busy accounts-payable clerk would, on a Friday,
    with the invoice attached.
    """)
    return


@app.cell(hide_code=True)
def bec_beat(CRIMSON, INK, KICK, MIST, mo):
    BEC_SUBJECT7 = "Re: Q3 invoice - updated banking details"
    BEC_FROM7 = "Vendor Finance Desk <ap-updates@vendor-portal.example>"
    BEC_BODY7 = (
        "Hi, following our conversation last week, please note our accounts "
        "payable email has changed. Kindly update your records and process the "
        "attached invoice to the new details by Friday."
    )
    BEC_PHISH_P7 = 0.000

    mo.Html(
        f"""
        <div style="background:{INK};border:1px solid #24404F;border-radius:12px;padding:18px 22px;max-width:660px;">
          <div style="font-family:ui-monospace,Menlo,monospace;font-size:11px;letter-spacing:2px;color:{KICK};margin-bottom:8px;">ACT 1 SWEEP &#183; ENGLISH CHECKPOINT &#183; PHISHING QUESTION</div>
          <div style="font-family:-apple-system,Segoe UI,sans-serif;font-size:16px;font-weight:700;color:#EEF4FB;margin-bottom:4px;">{BEC_SUBJECT7}</div>
          <div style="font-family:ui-monospace,Menlo,monospace;font-size:12px;color:{KICK};margin-bottom:10px;">{BEC_FROM7}</div>
          <div style="font-family:-apple-system,Segoe UI,sans-serif;font-size:13.5px;line-height:1.55;color:{MIST};margin-bottom:14px;">{BEC_BODY7}</div>
          <div style="display:inline-block;background:{CRIMSON};border-radius:999px;padding:5px 14px;font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#EEF4FB;">P(phishing) = {BEC_PHISH_P7:.3f}</div>
        </div>
        """
    )
    return (BEC_PHISH_P7,)


@app.cell(hide_code=True)
def bec_reading(BEC_PHISH_P7, DLR, cpu_cost7, mo):
    mo.md(
        f"""
        This is business email compromise: no link, no urgency theater, just a
        plausible voice asking for a routing change. The English checkpoint
        scored it **P(phishing) = {BEC_PHISH_P7:.3f}**, and the confidence
        score, as always in this series, said nothing.

        That is the honest boundary of this act. The front half can be fast,
        cheap, and auditable, serve a million requests for about
        {DLR}{cpu_cost7:.0f}, and still wave the most expensive email fraud on
        the books straight through to the finance queue.
        """
    )
    return


@app.cell(hide_code=True)
def series_recap(BEC_PHISH_P7, mo):
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

    Laya earns its place as the fast, cheap, auditable front half of a
    pipeline. The back half, the part that catches business email compromise
    at P = {BEC_PHISH_P7:.3f}, still needs a System 2.
    """)
    return


if __name__ == "__main__":
    app.run()
