# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo==0.25.1",
#     "laya==0.3.5",
#     "matplotlib",
# ]
# ///
"""Act 2: The confidence trap.

A confidence score is supposed to warn you when a model is out of its depth.
Feed the English checkpoint Khmer text and watch it stay silent: 0.95+
confidence at essentially random accuracy. Then watch the Router catch the
same text BEFORE the forward pass, because the model cannot self-report
confusion. First run downloads both checkpoints from HuggingFace.
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
        # Act 2 - The confidence trap

        > **The Laya pharma surveillance series · Act 2 of 7**
        > Run me: `uvx marimo edit --sandbox notebooks/02-the-confidence-trap.py`

        Act 1 ended on a question: should you trust the confidence numbers?

        Here is the same billing complaint written in Khmer. One checkpoint
        reads it fluently. The other returns confident nonsense, and its
        confidence score says nothing is wrong.
        """
    )
    return


@app.cell
def _(mo):
    import contextlib
    import os

    os.environ.setdefault("USE_TF", "0")

    import laya
    from laya import Router

    load_ctx2 = (
        mo.spinner("loading both checkpoints (first run downloads ~1.5 GB) ...")
        if hasattr(mo, "spinner")
        else contextlib.nullcontext()
    )
    with load_ctx2:
        english2 = laya.load("convaiinnovations/laya")
        multi2 = laya.load("convaiinnovations/laya-multilingual")
        router2 = Router(max_loaded=2)
    return english2, multi2, router2


@app.cell
def _():
    # The Act 1 email, translated to Khmer by a human-quality translator.
    KHMER2 = {
        "subject": "វិក្កយបត្រ 4411 ត្រូវបានគិតថ្លៃពីរដង",
        "body": (
            "យើងត្រូវបានគិតថ្លៃពីរដងសម្រាប់ខែមីនា។ សូមប្រគល់ប្រាក់វិញនូវចំនួនដែលគិតពីរដង។ "
            "ប្រសិនបើមិនបានដោះស្រាយក្នុងសប្តាហ៍នេះ យើងនឹងលុបចោលគណនីរបស់យើង។"
        ),
    }

    TRIAGE_Q2 = {
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
            "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
        },
        "churn_risk": {
            "type": "noul",
            "instructions": "Does the user threaten to cancel or leave?",
        },
    }
    return KHMER2, TRIAGE_Q2


@app.cell
def _(mo):
    mo.md("## The English checkpoint meets Khmer")
    return


@app.cell
def _(KHMER2, TRIAGE_Q2, english2):
    r_en2 = english2.predict(KHMER2, TRIAGE_Q2)["answers"]
    return (r_en2,)


@app.cell
def _(KHMER2, mo, r_en2):
    import matplotlib.pyplot as plt_kh

    probs_en2 = r_en2["department"]["probabilities"]
    order_en2 = sorted(probs_en2, key=probs_en2.get)
    fig_kh2, (axL2, axR2) = plt_kh.subplots(1, 2, figsize=(10, 2.6))
    axL2.barh(order_en2, [probs_en2[k] for k in order_en2], color="#d9534f")
    axL2.set_title(
        f'laya (English ckpt): "{r_en2["department"]["choice"]}" '
        f'at {r_en2["department"]["confidence"]:.3f} confidence'
    )
    axR2.axis("off")
    axR2.text(
        0.0,
        0.75,
        "What the email actually says:",
        fontsize=10,
        weight="bold",
    )
    axR2.text(
        0.0,
        0.45,
        "\n".join(
            [
                '"Duplicate charge on invoice 4411"',
                '"We were billed twice for March. Please refund',
                'the duplicate. If this isn\'t fixed this week',
                'we will cancel our account."',
            ]
        ),
        fontsize=10,
        family="monospace",
    )
    axR2.text(
        0.0,
        0.08,
        f'churn P = {r_en2["churn_risk"]["noul"]:.3f}   urgency = {r_en2["urgency"]["score"]:.2f}',
        fontsize=10,
    )
    mo.vstack([KHMER2["subject"], fig_kh2])
    return (fig_kh2,)


@app.cell
def _(mo, r_en2):
    mo.md(
        f"""
        A billing complaint came back routed to **`{r_en2["department"]["choice"]}`**
        at **{r_en2["department"]["confidence"]:.3f} confidence**.

        The model card says what this actually is: the English checkpoint
        scores **0.000 accuracy at 0.952 mean confidence** on Khmer. Random
        answers, maximum confidence. The confidence score, the thing you would
        wire to a human-review queue, has nothing to say.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## Same email, multilingual checkpoint")
    return


@app.cell
def _(KHMER2, TRIAGE_Q2, mo, multi2):
    r_ml2 = multi2.predict(KHMER2, TRIAGE_Q2)["answers"]
    mo.md(
        f"""
        `laya-multilingual` (mmBERT-base, 256k vocab): department =
        **{r_ml2["department"]["choice"]}** at
        **{r_ml2["department"]["confidence"]:.3f}**, urgency =
        {r_ml2["urgency"]["score"]:.2f}, churn P = {r_ml2["churn_risk"]["noul"]:.3f}.
        Fluent. The failure was never the language; it was the checkpoint.
        """
    )
    return (r_ml2,)


@app.cell
def _(KHMER2, mo, router2):
    decision2 = router2.route(KHMER2)
    info2 = getattr(decision2, "__dict__", None) or decision2
    mo.md(
        f"""
        ## The fix sits before the model

        The library ships a `Router` that reads the script of the input
        (pure Python, under a millisecond) and dispatches to the right
        checkpoint. On the Khmer email it routes to:

        ```
        {info2}
        ```

        The design decision is the lesson: **put the check before the model,
        do not trust the model to self-report confusion.** The confidence
        score cannot warn you; it was silent the whole time.
        """
    )
    return


@app.cell
def _():
    TRANSLATIONS2 = {
        "en": ("Duplicate charge on invoice 4411", "We were billed twice for March. Please refund the duplicate. If this isn't fixed this week we will cancel our account."),
        "es": ("Cobro duplicado en la factura 4411", "Nos cobraron dos veces por marzo. Por favor, reembolsen el cargo duplicado. Si no se resuelve esta semana, cancelaremos nuestra cuenta."),
        "fr": ("Frais en double sur la facture 4411", "Nous avons été facturés deux fois pour mars. Merci de rembourser le montant en double. Si ce n'est pas réglé cette semaine, nous résilierons notre compte."),
        "de": ("Doppelte Abbuchung auf Rechnung 4411", "Uns wurde im März doppelt abgebucht. Bitte erstatten Sie den doppelten Betrag. Wenn das nicht diese Woche gelöst wird, kündigen wir unser Konto."),
        "pt": ("Cobrança duplicada na fatura 4411", "Fomos cobrados duas vezes em março. Por favor, reembolsem a cobrança duplicada. Se não for resolvido esta semana, cancelaremos nossa conta."),
        "ru": ("Двойное списание по счёту 4411", "С нас дважды списали платёж за март. Пожалуйста, верните лишнюю сумму. Если проблема не будет решена на этой неделе, мы отменим наш аккаунт."),
        "ar": ("رسوم مكررة على الفاتورة 4411", "تم خصم رسوم شهر مارس مرتين. يرجى إعادة المبلغ المكرر. إذا لم يتم حل المشكلة هذا الأسبوع، فسنقوم بإلغاء حسابنا."),
        "hi": ("इनवॉइस 4411 पर दोहरा शुल्क", "हमसे मार्च का शुल्क दो बार लिया गया है। कृपया दोहरा शुल्क वापस करें। यदि यह सप्ताह भर में ठीक नहीं होता, तो हम अपना खाता रद्द कर देंगे।"),
        "zh": ("发票 4411 重复扣费", "我们三月份的费用被重复扣取了。请退还重复扣款。如果本周内不解决，我们将取消账户。"),
        "ja": ("請求書4411の重複請求", "3月分の料金が二重に請求されました。重複分を返金してください。今週中に解決されない場合、アカウントを解約します。"),
        "ko": ("송장 4411 이중 청구", "3월 요금이 두 번 청구되었습니다. 중복 청구분을 환불해 주세요. 이번 주 안에 해결되지 않으면 계정을 해지하겠습니다."),
    }
    return (TRANSLATIONS2,)


@app.cell
def _(mo):
    mo.md("## The same email, eleven languages")
    return


@app.cell
def _(TRANSLATIONS2, TRIAGE_Q2, mo, multi2):
    grid_rows2 = []
    for lang2, (subj2, body2) in TRANSLATIONS2.items():
        a2 = multi2.predict({"subject": subj2, "body": body2}, TRIAGE_Q2)["answers"]
        grid_rows2.append(
            {
                "lang": lang2,
                "department": f'{a2["department"]["choice"]} ({a2["department"]["confidence"]:.2f})',
                "urgency": f'{a2["urgency"]["score"]:.2f}',
                "P(churn)": f'{a2["churn_risk"]["noul"]:.3f}',
            }
        )
    mo.ui.table(grid_rows2, selection=None)
    return (grid_rows2,)


@app.cell
def _(mo):
    mo.md("## Latency, measured live on this machine")
    return


@app.cell
def _(TRIAGE_Q2, english2, mo, multi2, time):
    STATE_EN2 = {
        "subject": "Duplicate charge on invoice 4411",
        "body": "We were billed twice for March. Please refund the duplicate. If this isn't fixed this week we will cancel our account.",
    }

    def bench2(agent2, state2, n2=10):
        for _ in range(3):
            agent2.predict(state2, TRIAGE_Q2)
        t2 = time.perf_counter()
        for _ in range(n2):
            agent2.predict(state2, TRIAGE_Q2)
        return (time.perf_counter() - t2) / n2 * 1000

    ms_en2 = bench2(english2, STATE_EN2)
    ms_ml2 = bench2(multi2, STATE_EN2)
    mo.md(
        f"""
        | checkpoint | ms per pass (4 questions) |
        |---|---|
        | laya (English, 421M) | {ms_en2:.1f} |
        | laya-multilingual (mmBERT, 322M) | {ms_ml2:.1f} |

        The multilingual checkpoint routed about twice as fast in the runs
        behind the blog post; your numbers above are the ones that count on
        your hardware.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## What to carry out of Act 2

        1. The confidence score stayed silent while the model was confidently
           wrong. Every time. It is not a safety rail; it is another output.
        2. Deterministic checks (script detection, routing rules) belong
           **before** the forward pass, not after it.
        3. Multilingual coverage is real when you use the right checkpoint -
           and cheap, at tens of milliseconds per pass.

        **Act 3:** the same silence, plus one badly-phrased sentence of mine
        that suppressed all 38 adverse-event detections in a surveillance
        set.
        """
    )
    return


if __name__ == "__main__":
    app.run()
