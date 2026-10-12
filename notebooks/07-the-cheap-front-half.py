# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo==0.25.1",
#     "matplotlib",
#     "numpy",
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

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt7
    import numpy as np

    plt7.rcParams.update(
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

    return mo, np, plt7


@app.cell
def _(mo):
    mo.md(
        """
        # Act 7 - The cheap front half

        > **The Laya pharma surveillance series · Act 7 of 7**
        > Runs on committed data, no model downloads.

        Everything so far ran on a laptop. This act prices it as a service:
        the English checkpoint behind an HTTP endpoint on Modal, weights baked
        into the image, both profiles scaling to zero, deploys gated on CI
        smoke tests. The measurements below come from the benchmark harness in
        `nll-ai/laya-demo/modal/benchmark.py`.
        """
    )
    return


@app.cell
def _(mo):
    PROFILES7 = {
        "CPU (1 core, 3 GiB)": {
            "cold start s": 3.5,
            "p50 s": 3.31,
            "server inference s": 2.9,
            "throughput req/s (serial)": 1.2,
            "$ per hour": 0.071,
        },
        "T4 GPU (2 cores, 8 GiB)": {
            "cold start s": 44.3,
            "p50 s": 0.48,
            "server inference s": 0.049,
            "throughput req/s (8 concurrent)": 8.0,
            "$ per hour": 0.749,
        },
    }
    rows7 = [
        {
            "profile": name7,
            "cold start": f"{v7['cold start s']:.1f} s",
            "p50 latency": f"{v7['p50 s']:.2f} s",
            "server-side inference": (
                f"{v7['server inference s'] * 1000:.0f} ms"
                if v7["server inference s"] < 1
                else f"{v7['server inference s']:.1f} s"
            ),
            "billable": f"${v7['$ per hour']:.3f}/h",
        }
        for name7, v7 in PROFILES7.items()
    ]
    mo.ui.table(rows7, selection=None)
    return (PROFILES7,)


@app.cell
def _(mo):
    mo.md("## What a million requests cost")
    return


@app.cell
def _(PROFILES7, mo):
    # $ per 1M requests = ($/h) / (req/s) * 1e6 / 3600
    def cost_per_million7(rate7, req_s7):
        return rate7 / req_s7 * 1e6 / 3600

    cpu7 = PROFILES7["CPU (1 core, 3 GiB)"]
    gpu7 = PROFILES7["T4 GPU (2 cores, 8 GiB)"]
    cpu_cost7 = cost_per_million7(cpu7["$ per hour"], cpu7["throughput req/s (serial)"])
    gpu_cost7 = cost_per_million7(
        gpu7["$ per hour"], gpu7["throughput req/s (8 concurrent)"]
    )
    mo.md(
        f"""
        One line of arithmetic, per profile: hourly rate ÷ throughput ×
        seconds in an hour.

        - CPU: $0.071/h ÷ 1.2 req/s → **${cpu_cost7:.0f} per 1M requests**
        - T4: $0.749/h ÷ 8.0 req/s → **${gpu_cost7:.0f} per 1M requests**

        Below full utilization the ranges widen (the blog post measured
        $17-68 for CPU, $26-115 for T4), and the ordering never flips: the
        CPU profile wins on cost for batch work at every concurrency
        measured.
        """
    )
    return cpu_cost7, gpu_cost7


@app.cell
def _(cpu_cost7, gpu_cost7, mo, plt7):
    fig_cost7, ax_cost7 = plt7.subplots(figsize=(7, 3))
    bars7 = ax_cost7.bar(
        ["CPU", "T4 GPU"],
        [cpu_cost7, gpu_cost7],
        color=["#4a7ebb", "#4ade80"],
        width=0.5,
    )
    ax_cost7.set_ylabel("$ per 1M requests")
    ax_cost7.set_title("batch triage: the CPU profile wins on cost")
    for b7 in bars7:
        ax_cost7.text(
            b7.get_x() + b7.get_width() / 2,
            b7.get_height() + 1,
            f"${b7.get_height():.0f}",
            ha="center",
            fontsize=10,
        )
    fig_cost7.tight_layout()
    mo.vstack([fig_cost7])
    return (fig_cost7,)


@app.cell
def _(PROFILES7, mo, np, plt7):
    lat_p50_7 = [PROFILES7[p7]["p50 s"] for p7 in PROFILES7]
    lat_inf_7 = [PROFILES7[p7]["server inference s"] for p7 in PROFILES7]
    x8 = np.arange(2)
    fig_lat7, ax_lat7 = plt7.subplots(figsize=(7.5, 3.4))
    ax_lat7.bar(x8 - 0.19, lat_p50_7, 0.38, label="p50, end to end", color="#a8b2bd")
    ax_lat7.bar(
        x8 + 0.19, lat_inf_7, 0.38, label="server-side inference only", color="#4a7ebb"
    )
    ax_lat7.set_yscale("log")
    ax_lat7.set_ylabel("seconds (log scale)")
    ax_lat7.set_xticks(x8, ["CPU", "T4 GPU"])
    ax_lat7.set_title("the T4 buys 7x latency, not cost")
    ax_lat7.legend()
    fig_lat7.tight_layout()
    mo.vstack([fig_lat7])
    return (fig_lat7,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Cold starts and the shape of the bill

        CPU cold start: 3.5 seconds. T4 cold start: 44.3 seconds. Both
        profiles scale to zero after 60 seconds idle, so sporadic CPU traffic
        costs pennies - a quiet overnight surveillance feed bills for
        almost nothing and wakes in 3.5 seconds when a post arrives.

        Deploys run only from GitHub Actions on the `nll-ai` workspace, and
        CI smoke-tests both live profiles before declaring a deploy green.
        A push to `main` is the whole ops story.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## Why the front half is not the whole pipeline")
    return


@app.cell
def _(mo):
    mo.md(
        """
        One email from Act 1's sweep, run through the English checkpoint's
        phishing question:

        > **Re: Q3 invoice - updated banking details**
        >
        > Hi, following our conversation last week, please note our accounts
        > payable email has changed. Kindly update your records and process
        > the attached invoice to the new details by Friday.

        This is business email compromise: no link, no urgency theater, just
        a plausible voice asking for a routing change. Laya scored
        **P(phishing) = 0.000**. The confidence score, as always, said
        nothing.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## The series, closed

        1. One forward pass answers typed questions in milliseconds.
        2. The confidence score is silent exactly when the model is wrong.
        3. One badly-phrased clause can invert a question head; schemas are
           code.
        4. Zero-shot covers the operational core on real documents.
        5. 212 labels and 20 minutes close the edges (0.26-0.58 → 0.89-0.96).
        6. Calibration is two steps: proper scoring rules, then a per-bucket
           refit - and it beats the closed competitor's shipped number.
        7. The cheap deployment wins on cost; latency is the only reason to
           buy the GPU.

        Laya earns its place as the fast, cheap, auditable front half of a
        pipeline. The back half - the part that catches business email
        compromise at P = 0.000 - still needs a System 2.
        """
    )
    return


if __name__ == "__main__":
    app.run()
