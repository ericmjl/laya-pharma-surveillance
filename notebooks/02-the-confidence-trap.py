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
"""Act 2: The confidence trap.

A confidence score is supposed to warn you when a model is out of its depth.
Feed the English checkpoint Khmer text and watch it stay silent: 0.95+
confidence at essentially random accuracy. Then watch the Router catch the
same text BEFORE the forward pass, because the model cannot self-report
confusion. First run downloads both checkpoints from HuggingFace.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def banner(mo):
    mo.Html("""
    <div style="line-height:0;border-radius:16px;overflow:hidden;">
    <svg viewBox="0 0 960 210" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Act 2 - The confidence trap">
      <defs>
        <linearGradient id="bg2" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="#0b1524"/><stop offset="1" stop-color="#13253f"/>
        </linearGradient>
        <linearGradient id="acc2" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stop-color="#4a7ebb"/><stop offset="0.55" stop-color="#4ade80"/><stop offset="1" stop-color="#d9534f"/>
        </linearGradient>
      </defs>
      <rect x="1" y="1" width="958" height="208" rx="16" fill="url(#bg2)" stroke="#22364f"/>
      <text x="690" y="188" font-family="ui-monospace,Menlo,monospace" font-size="160" font-weight="700" fill="none" stroke="#26405f" stroke-width="2" opacity="0.5">02</text>
      <rect x="798" y="34" width="122" height="26" rx="13" fill="#1b2f4a" stroke="#33507a"/>
      <text x="813" y="51" font-family="ui-monospace,Menlo,monospace" font-size="11" letter-spacing="2" fill="#9fc0e8">ACT 2 OF 7</text>
      <text x="40" y="54" font-family="ui-monospace,Menlo,monospace" font-size="12" letter-spacing="3.5" fill="#7f9cc0">THE LAYA PHARMA SURVEILLANCE SERIES</text>
      <text x="40" y="112" font-family="-apple-system,Segoe UI,sans-serif" font-size="42" font-weight="800" fill="#eef4fb">The confidence trap</text>
      <text x="40" y="148" font-family="ui-monospace,Menlo,monospace" font-size="14.5" fill="#a8b2bd">Degradation you can see. Collapse you cannot.</text>
      <g font-family="ui-monospace,Menlo,monospace" font-size="11">
        <rect x="620" y="52" width="74" height="21" rx="6" fill="#1b2f4a" stroke="#33507a"/><text x="630" y="66" fill="#9fc0e8">en ckpt</text>
        <rect x="620" y="82" width="74" height="21" rx="6" fill="#1b2f4a" stroke="#33507a"/><text x="630" y="96" fill="#9fc0e8">multi</text>
        <rect x="620" y="112" width="74" height="21" rx="6" fill="#1b2f4a" stroke="#33507a"/><text x="630" y="126" fill="#9fc0e8">router</text>
        <rect x="620" y="142" width="74" height="21" rx="6" fill="#1b2f4a" stroke="#33507a"/><text x="630" y="156" fill="#9fc0e8">11 langs</text>
      </g>
      <path d="M694 62 C 724 62 736 100 758 101" stroke="#33507a" fill="none"/>
      <path d="M694 92 C 720 92 734 101 758 101" stroke="#33507a" fill="none"/>
      <path d="M694 122 C 720 122 734 103 758 103" stroke="#33507a" fill="none"/>
      <path d="M694 152 C 724 152 736 104 758 103" stroke="#33507a" fill="none"/>
      <circle cx="772" cy="102" r="15" fill="#13253f" stroke="#d9534f" stroke-width="2"/>
      <text x="766" y="107" font-family="ui-monospace,Menlo,monospace" font-size="12" fill="#eef4fb">?</text>
      <g>
        <rect x="806" y="70" width="98" height="9" rx="3" fill="#d9534f"/>
        <rect x="806" y="92" width="61" height="9" rx="3" fill="#d9534f" opacity="0.6"/>
        <rect x="806" y="114" width="34" height="9" rx="3" fill="#4a7ebb" opacity="0.6"/>
        <rect x="806" y="136" width="16" height="9" rx="3" fill="#4a7ebb" opacity="0.35"/>
      </g>
      <text x="806" y="166" font-family="ui-monospace,Menlo,monospace" font-size="10" fill="#5f7ea6">P(option)</text>
      <rect x="24" y="186" width="560" height="4" rx="2" fill="url(#acc2)"/>
    </svg>
    </div>
    """)
    return


@app.cell(hide_code=True)
def goal(mo):
    mo.md("""
    In this notebook, we will show what a decision model's confidence score is
    worth when the model leaves its comfort zone.

    By the end of the notebook, you will:

    - see the English checkpoint degrade, visibly and verifiably, on a
      Mandarin email you can read yourself,
    - see the same checkpoint collapse with high confidence on a Khmer email
      that was machine-translated (I cannot read Khmer, so we reason assuming
      the translation is correct),
    - see the Router catch the script before the model runs, and
    - see the same email hold steady across eleven languages on the
      multilingual checkpoint.
    """)
    return


@app.cell(hide_code=True)
def pre_load(mo):
    mo.md("""
    ## Two checkpoints and a Router

    Everything below runs on two open-weights checkpoints from ConvAI
    Innovations: `laya` (English, 421M params, ModernBERT-large) and
    `laya-multilingual` (322M params, mmBERT-base, 256k vocab). The library
    also ships a Router that sends each email to the right checkpoint based
    on its script. The first run downloads both checkpoints (about 1.5 GB
    total). The load log shows a temperature-clamp warning; that becomes the
    story in notebook 6.
    """)
    return


@app.cell(hide_code=True)
def imports():
    import contextlib
    import os
    import statistics
    import sys
    from pathlib import Path

    os.environ.setdefault("USE_TF", "0")

    import laya
    import marimo as mo
    import plotly.graph_objects as go
    import time
    from laya import Router

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mail_viewer import MailViewer

    return MailViewer, Router, contextlib, go, laya, mo, statistics, time


@app.cell(hide_code=True)
def load_models(Router, contextlib, laya, mo):
    load_ctx2 = (
        mo.spinner(
            "loading both checkpoints (first run downloads ~1.5 GB) ..."
        )
        if hasattr(mo, "spinner")
        else contextlib.nullcontext()
    )
    with load_ctx2:
        english2 = laya.load("convaiinnovations/laya")
        multi2 = laya.load("convaiinnovations/laya-multilingual")
        router2 = Router(max_loaded=2)

    n_en2 = sum(p.numel() for p in english2.model.parameters())
    n_ml2 = sum(p.numel() for p in multi2.model.parameters())
    return english2, multi2, n_en2, n_ml2, router2


@app.cell(hide_code=True)
def load_reading(mo, n_en2, n_ml2):
    mo.md(f"""
    Both checkpoints are loaded: {n_en2 / 1e6:.0f}M params (English) and
    {n_ml2 / 1e6:.0f}M params (multilingual). Note the warning in the log:
    one English bucket ships a fitted temperature of 0.10, so far outside the
    sane range that the library clamps it and tells you the bucket is
    uncalibrated. Remember that when calibration gets its own act.
    """)
    return


@app.cell(hide_code=True)
def triage_schema():
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
            "criteria": [
                "not urgent",
                "soon",
                "critical deadline or blocking issue",
            ],
        },
        "churn_risk": {
            "type": "noul",
            "instructions": "Does the user threaten to cancel or leave?",
        },
    }
    return (TRIAGE_Q2,)


@app.cell(hide_code=True)
def mandarin_intro(mo):
    mo.md("""
    ## A Mandarin email I can verify

    Here is the Act 1 billing complaint, in Mandarin. I read Mandarin, so I
    can verify every word myself: a duplicate charge on invoice 4411, a
    refund demanded, an account cancelled if it is not fixed this week.
    Correct triage is department `billing`, urgency mid-scale, and a churn
    signal somewhere between "real" and "conditional boilerplate".
    """)
    return


@app.cell(hide_code=True)
def mandarin_email(MailViewer):
    ZH_EMAIL2 = {
        "subject": "发票 4411 重复扣费",
        "body": "我们三月份的费用被重复扣取了。请退还重复扣款。如果本周内不解决，我们将取消账户。",
    }

    MailViewer(
        emails=[
            {
                "sender_name": "Maya Chen",
                "sender_email": "maya@northwind-clinic.com",
                "initials": "MC",
                "subject": ZH_EMAIL2["subject"],
                "body": ZH_EMAIL2["body"],
                "time": "8:58 AM",
                "note": "English translation (I wrote it; the model never saw it): Duplicate charge on invoice 4411. Our March fees were charged twice; please refund the duplicate. If this is not fixed this week, we will cancel the account.",
            }
        ]
    )
    return (ZH_EMAIL2,)


@app.cell(hide_code=True)
def mandarin_test_intro(mo):
    mo.md("""
    Run it through both checkpoints. Same email, same schema, two opinions.
    """)
    return


@app.cell(hide_code=True)
def mandarin_test(TRIAGE_Q2, ZH_EMAIL2, english2, multi2):
    ZH_RESULTS2 = {}
    for name2, agent2 in [("english", english2), ("multilingual", multi2)]:
        a2 = agent2.predict(
            {"subject": ZH_EMAIL2["subject"], "body": ZH_EMAIL2["body"]},
            TRIAGE_Q2,
        )["answers"]
        ZH_RESULTS2[name2] = {
            "dept": a2["department"]["choice"],
            "conf": float(a2["department"]["confidence"]),
            "probs": {
                k: float(v)
                for k, v in a2["department"]["probabilities"].items()
            },
            "urgency": float(a2["urgency"]["score"]),
            "churn": float(a2["churn_risk"]["noul"]),
        }
    ZH_RESULTS2
    return (ZH_RESULTS2,)


@app.cell(hide_code=True)
def mandarin_plot(ZH_RESULTS2, go):
    opts2 = ["billing", "technical", "sales", "other"]
    fig_zh2 = go.Figure()
    fig_zh2.add_bar(
        name="English checkpoint",
        x=opts2,
        y=[ZH_RESULTS2["english"]["probs"].get(o2, 0) for o2 in opts2],
        marker_color="#a8b2bd",
    )
    fig_zh2.add_bar(
        name="Multilingual checkpoint",
        x=opts2,
        y=[ZH_RESULTS2["multilingual"]["probs"].get(o2, 0) for o2 in opts2],
        marker_color="#4a7ebb",
    )
    fig_zh2.update_layout(
        barmode="group",
        title=dict(
            text=f"P(department) on the Mandarin email - "
            f"english {ZH_RESULTS2['english']['dept']} @{ZH_RESULTS2['english']['conf']:.2f} vs "
            f"multilingual {ZH_RESULTS2['multilingual']['dept']} @{ZH_RESULTS2['multilingual']['conf']:.2f}",
            font=dict(size=13),
        ),
        height=340,
        margin=dict(l=10, r=20, t=60, b=10),
        template="plotly_white",
        yaxis=dict(range=[0, 1.1]),
    )
    fig_zh2
    return


@app.cell(hide_code=True)
def mandarin_reading(ZH_RESULTS2, mo):
    mo.md(f"""
    Grade the two against what you read.

    - The **English checkpoint** gets the department right but shakily:
      {ZH_RESULTS2["english"]["conf"]:.2f} confidence, with real mass leaking
      to other options. It read Mandarin, but it read it like a second
      language.
    - The **multilingual checkpoint** nails it:
      {ZH_RESULTS2["multilingual"]["dept"]} at
      {ZH_RESULTS2["multilingual"]["conf"]:.2f}.
    - On churn they disagree, and it is a genuine judgment call. The email
      says "we will cancel the account if this is not fixed this week": a
      conditional threat. The English checkpoint reads it as a strong churn
      signal ({ZH_RESULTS2["english"]["churn"]:.2f}); the multilingual
      checkpoint barely registers it ({ZH_RESULTS2["multilingual"]["churn"]:.2f}).
      Same sentence, same semantics. When two checkpoints disagree that
      widely on a borderline case, that is not noise; it is a calibration
      conversation, and it gets its own act.

    The visible lesson: on a language you can check, the English checkpoint
    degrades gracefully. Keep that next to what follows.
    """)
    return


@app.cell(hide_code=True)
def khmer_intro(mo):
    mo.md("""
    ## A Khmer email, machine-translated

    Same complaint, Khmer script. I asked AI to translate it and I cannot
    check it: I do not read Khmer. Assume the translation is correct. If it
    is, the right triage is `billing`, mid urgency, some churn signal, and
    the model has none of that context; it sees only Khmer characters.
    Languages this far outside the training data are where classification
    collapses, and where fine-tuning earns its keep.
    """)
    return


@app.cell(hide_code=True)
def khmer_email(MailViewer):
    KHMER_EMAIL2 = {
        "subject": "វិក្កយបត្រ 4411 ត្រូវបានគិតថ្លៃពីរដង",
        "body": (
            "យើងត្រូវបានគិតថ្លៃពីរដងសម្រាប់ខែមីនា។ សូមប្រគល់ប្រាក់វិញនូវចំនួនដែលគិតពីរដង។ "
            "ប្រសិនបើមិនបានដោះស្រាយក្នុងសប្តាហ៍នេះ យើងនឹងលុបចោលគណនីរបស់យើង។"
        ),
    }

    MailViewer(
        emails=[
            {
                "sender_name": "Maya Chen",
                "sender_email": "maya@northwind-clinic.com",
                "initials": "MC",
                "subject": KHMER_EMAIL2["subject"],
                "body": KHMER_EMAIL2["body"],
                "time": "9:12 AM",
                "note": "English translation (machine-translated; I cannot verify it): Duplicate charge on invoice 4411. We were billed twice for March; please refund the duplicate. If this is not fixed this week, we will cancel our account.",
            }
        ]
    )
    return (KHMER_EMAIL2,)


@app.cell(hide_code=True)
def khmer_test_intro(mo):
    mo.md("""
    Run the Khmer email through the **English** checkpoint, the one tuned on
    English text.
    """)
    return


@app.cell(hide_code=True)
def khmer_test(KHMER_EMAIL2, TRIAGE_Q2, english2):
    r_kh_en2 = english2.predict(
        {"subject": KHMER_EMAIL2["subject"], "body": KHMER_EMAIL2["body"]},
        TRIAGE_Q2,
    )["answers"]
    KH_EN2 = {
        "dept": r_kh_en2["department"]["choice"],
        "conf": float(r_kh_en2["department"]["confidence"]),
        "probs": {
            k: float(v)
            for k, v in r_kh_en2["department"]["probabilities"].items()
        },
        "urgency": float(r_kh_en2["urgency"]["score"]),
        "churn": float(r_kh_en2["churn_risk"]["noul"]),
    }
    KH_EN2
    return (KH_EN2,)


@app.cell(hide_code=True)
def khmer_plot(KH_EN2, go):
    order_kh2 = sorted(KH_EN2["probs"], key=KH_EN2["probs"].get)
    fig_kh2 = go.Figure(
        go.Bar(
            x=[KH_EN2["probs"][k2] for k2 in order_kh2],
            y=order_kh2,
            orientation="h",
            marker_color="#d9534f",
            text=[f"{KH_EN2['probs'][k2]:.3f}" for k2 in order_kh2],
            textposition="outside",
        )
    )
    fig_kh2.update_layout(
        title=dict(
            text=f"English checkpoint on Khmer: '{KH_EN2['dept']}' at {KH_EN2['conf']:.3f} confidence",
            font=dict(size=13),
        ),
        height=260,
        margin=dict(l=10, r=60, t=46, b=10),
        template="plotly_white",
        xaxis=dict(range=[0, 1.12]),
    )
    fig_kh2
    return


@app.cell(hide_code=True)
def khmer_reading(KH_EN2, mo):
    mo.md(f"""
    Assuming our translation is right, the correct answer is `billing` (the
    gloss says so, and the model never saw the gloss). The checkpoint puts its
    mass on `{KH_EN2["dept"]}` at **{KH_EN2["conf"]:.3f} confidence**, and the
    confidence score says nothing is wrong. The model card's number for
    exactly this setup: **0.000 accuracy at 0.952 mean confidence** on Khmer.

    Put the two failures side by side. On Mandarin the checkpoint degraded
    visibly: right answer, low confidence. On Khmer, a language that is
    really out of distribution for it, classification collapses: wrong answer,
    top confidence. Collapse like that is where fine-tuning on your own
    labeled traffic stops being optional; it is the whole subject of
    notebook 5.
    """)
    return


@app.cell(hide_code=True)
def router_intro(mo):
    mo.md("""
    ## The fix sits before the model

    The library ships a Router that reads the script of the input, pure
    Python, under a millisecond, and dispatches to the right checkpoint. Run
    it on the Khmer email, then run the email through the checkpoint it
    picked.
    """)
    return


@app.cell(hide_code=True)
def router_demo(KHMER_EMAIL2, MailViewer, TRIAGE_Q2, multi2, router2):
    decision2 = router2.route(
        {"subject": KHMER_EMAIL2["subject"], "body": KHMER_EMAIL2["body"]}
    )
    reason2 = str(getattr(decision2, "reason", decision2))
    model2 = str(getattr(decision2, "model", decision2))

    r_kh_ml2 = multi2.predict(
        {"subject": KHMER_EMAIL2["subject"], "body": KHMER_EMAIL2["body"]},
        TRIAGE_Q2,
    )["answers"]

    MailViewer(
        emails=[
            {
                "sender_name": "Maya Chen",
                "sender_email": "maya@northwind-clinic.com",
                "initials": "MC",
                "subject": KHMER_EMAIL2["subject"],
                "body": KHMER_EMAIL2["body"],
                "time": "routed by script",
                "answers": {
                    "department": {
                        "choice": r_kh_ml2["department"]["choice"],
                        "confidence": float(
                            r_kh_ml2["department"]["confidence"]
                        ),
                    },
                    "urgency": {"score": float(r_kh_ml2["urgency"]["score"])},
                    "churn_risk": {
                        "noul": float(r_kh_ml2["churn_risk"]["noul"])
                    },
                },
                "note": "Router dispatch: "
                + model2
                + " ("
                + reason2
                + "). English translation (machine-translated): Duplicate charge on invoice 4411. We were billed twice for March; please refund the duplicate.",
            }
        ]
    )
    return


@app.cell(hide_code=True)
def router_reading(mo):
    mo.md("""
    The design decision is the lesson: put the deterministic check **before**
    the model, and do not trust the model to self-report confusion. The
    confidence score cannot warn you; this act started with proof of that.
    """)
    return


@app.cell(hide_code=True)
def languages_intro(mo):
    mo.md("""
    ## The same email, eleven languages

    The multilingual checkpoint covers scripts from Latin to Cyrillic to CJK.
    If cross-lingual consistency is real, the same complaint should get the
    same triage in every language on the list.
    """)
    return


@app.cell(hide_code=True)
def translations():
    TRANSLATIONS2 = {
        "en": (
            "Duplicate charge on invoice 4411",
            "We were billed twice for March. Please refund the duplicate. If this isn't fixed this week we will cancel our account.",
        ),
        "es": (
            "Cobro duplicado en la factura 4411",
            "Nos cobraron dos veces por marzo. Por favor, reembolsen el cargo duplicado. Si no se resuelve esta semana, cancelaremos nuestra cuenta.",
        ),
        "fr": (
            "Frais en double sur la facture 4411",
            "Nous avons été facturés deux fois pour mars. Merci de rembourser le montant en double. Si ce n'est pas réglé cette semaine, nous résilierons notre compte.",
        ),
        "de": (
            "Doppelte Abbuchung auf Rechnung 4411",
            "Uns wurde im März doppelt abgebucht. Bitte erstatten Sie den doppelten Betrag. Wenn das nicht diese Woche gelöst wird, kündigen wir unser Konto.",
        ),
        "pt": (
            "Cobrança duplicada na fatura 4411",
            "Fomos cobrados duas vezes em março. Por favor, reembolsem a cobrança duplicada. Se não for resolvido esta semana, cancelaremos nossa conta.",
        ),
        "ru": (
            "Двойное списание по счёту 4411",
            "С нас дважды списали платёж за март. Пожалуйста, верните лишнюю сумму. Если проблема не будет решена на этой неделе, мы отменим наш аккаунт.",
        ),
        "ar": (
            "رسوم مكررة على الفاتورة 4411",
            "تم خصم رسوم شهر مارس مرتين. يرجى إعادة المبلغ المكرر. إذا لم يتم حل المشكلة هذا الأسبوع، فسنقوم بإلغاء حسابنا.",
        ),
        "hi": (
            "इनवॉइस 4411 पर दोहरा शुल्क",
            "हमसे मार्च का शुल्क दो बार लिया गया है। कृपया दोहरा शुल्क वापस करें। यदि यह सप्ताह भर में ठीक नहीं होता, तो हम अपना खाता रद्द कर देंगे।",
        ),
        "zh": (
            "发票 4411 重复扣费",
            "我们三月份的费用被重复扣取了。请退还重复扣款。如果本周内不解决，我们将取消账户。",
        ),
        "ja": (
            "請求書4411の重複請求",
            "3月分の料金が二重に請求されました。重複分を返金してください。今週中に解決されない場合、アカウントを解約します。",
        ),
        "ko": (
            "송장 4411 이중 청구",
            "3월 요금이 두 번 청구되었습니다. 중복 청구분을 환불해 주세요. 이번 주 안에 해결되지 않으면 계정을 해지하겠습니다.",
        ),
    }
    return (TRANSLATIONS2,)


@app.cell(hide_code=True)
def languages_grid(TRANSLATIONS2, TRIAGE_Q2, go, multi2):
    grid_rows2 = []
    for lang2, (subj2, body2) in TRANSLATIONS2.items():
        ans2 = multi2.predict({"subject": subj2, "body": body2}, TRIAGE_Q2)[
            "answers"
        ]
        grid_rows2.append(
            {
                "lang": lang2,
                "dept": ans2["department"]["choice"],
                "conf": float(ans2["department"]["confidence"]),
                "urgency": float(ans2["urgency"]["score"]),
                "churn": float(ans2["churn_risk"]["noul"]),
            }
        )

    fig_table2 = go.Figure(
        go.Table(
            header=dict(
                values=["lang", "department", "conf", "urgency", "P(churn)"],
                fill_color="#13253f",
                font=dict(color="#eef4fb", size=12),
                align="left",
            ),
            cells=dict(
                values=[
                    [r2["lang"] for r2 in grid_rows2],
                    [r2["dept"] for r2 in grid_rows2],
                    [f"{r2['conf']:.4f}" for r2 in grid_rows2],
                    [f"{r2['urgency']:.2f}" for r2 in grid_rows2],
                    [f"{r2['churn']:.4f}" for r2 in grid_rows2],
                ],
                fill_color=[["#fbfbfd", "#f2f4f7"] * 6],
                font=dict(size=12),
                align="left",
            ),
        )
    )
    fig_table2.update_layout(height=400, margin=dict(l=5, r=5, t=10, b=0))
    fig_table2
    return (grid_rows2,)


@app.cell(hide_code=True)
def languages_plot(go, grid_rows2):
    fig_conf2 = go.Figure(
        go.Bar(
            x=[r2["lang"] for r2 in grid_rows2],
            y=[r2["conf"] for r2 in grid_rows2],
            marker_color="#4a7ebb",
            text=[f"{r2['conf']:.4f}" for r2 in grid_rows2],
            textposition="outside",
        )
    )
    fig_conf2.update_layout(
        title=dict(
            text="department confidence on the same complaint, 11 languages (multilingual checkpoint)",
            font=dict(size=13),
        ),
        height=280,
        margin=dict(l=10, r=20, t=46, b=0),
        template="plotly_white",
        yaxis=dict(range=[0, 1.12]),
    )
    fig_conf2
    return


@app.cell(hide_code=True)
def languages_reading(mo):
    mo.md("""
    Eleven languages, eleven `billing` calls, urgency in the same band every
    time. Several confidence values display as 1.000; that is softmax
    saturation on a clean-cut case, not a copy-paste bug. The fourth decimal
    is where the languages actually differ, and no language flips the
    decision. The multilingual checkpoint is the one to route non-English
    traffic to, and the Router already knows it.
    """)
    return


@app.cell(hide_code=True)
def latency_intro(mo):
    mo.md("""
    ## Latency, measured live on this machine

    Consistency should not cost much. Run both checkpoints ten times each on
    the English email and compare.
    """)
    return


@app.cell(hide_code=True)
def latency_bench(
    TRANSLATIONS2,
    TRIAGE_Q2,
    english2,
    go,
    multi2,
    statistics,
    time,
):
    STATE_EN2 = {
        "subject": "Duplicate charge on invoice 4411",
        "body": TRANSLATIONS2["en"][1],
    }

    def bench_latency2(agent2, state2, n2=10):
        for _ in range(3):
            agent2.predict(state2, TRIAGE_Q2)
        samples2 = []
        for _ in range(n2):
            t2 = time.perf_counter()
            agent2.predict(state2, TRIAGE_Q2)
            samples2.append((time.perf_counter() - t2) * 1000)
        return samples2

    samples_en2 = bench_latency2(english2, STATE_EN2)
    samples_ml2 = bench_latency2(multi2, STATE_EN2)
    mean_en2 = statistics.mean(samples_en2)
    mean_ml2 = statistics.mean(samples_ml2)
    std_en2 = statistics.stdev(samples_en2)
    std_ml2 = statistics.stdev(samples_ml2)

    fig_lat2 = go.Figure()
    fig_lat2.add_bar(
        x=["English checkpoint", "Multilingual checkpoint"],
        y=[mean_en2, mean_ml2],
        error_y=dict(type="data", array=[std_en2, std_ml2], visible=True),
        marker_color=["#a8b2bd", "#4a7ebb"],
        opacity=0.75,
    )
    fig_lat2.add_trace(
        go.Scatter(
            x=[0.18] * len(samples_en2) + [1.18] * len(samples_ml2),
            y=samples_en2 + samples_ml2,
            mode="markers",
            marker=dict(color="#1d1d1f", size=7, opacity=0.7),
            showlegend=False,
        )
    )
    fig_lat2.update_layout(
        title=dict(
            text=f"ms per pass, 4 questions (n=10 dots + mean ± sd): "
            f"{mean_en2:.1f}±{std_en2:.1f} vs {mean_ml2:.1f}±{std_ml2:.1f}",
            font=dict(size=13),
        ),
        height=300,
        margin=dict(l=10, r=20, t=46, b=10),
        template="plotly_white",
        yaxis=dict(title="milliseconds"),
    )
    fig_lat2
    return


@app.cell(hide_code=True)
def latency_reading(mo):
    mo.md("""
    Your numbers above are the ones that count on your hardware. In the runs
    behind the blog post, the multilingual checkpoint routed about twice as
    fast as the English one: smaller model, and it was built for exactly this
    job.
    """)
    return


@app.cell(hide_code=True)
def takeaways(mo):
    mo.md("""
    ## What to carry out of Act 2

    1. Degrading is visible; collapse is not. On Mandarin the English
       checkpoint keeps the right answer and drops confidence. On Khmer,
       really out of distribution, it keeps the confidence and drops the
       answer.
    2. When your traffic includes languages like that, fine-tuning on labeled
       examples is not a nice-to-have; it is the fix, and notebook 5 measures
       how far 212 labels and 20 minutes get you.
    3. The confidence score stayed silent in every failure above. It is not a
       safety rail; it is another output.
    4. Deterministic checks belong before the model: the Router reads the
       script in under a millisecond and dispatches to the checkpoint that
       can actually read the text.
    5. For non-English traffic, route to the multilingual checkpoint:
       smaller, faster, and right.

    That is what notebook 3 is all about: the same silence, plus one
    badly-phrased sentence of mine that suppressed all 38 adverse-event
    detections in a surveillance set.
    """)
    return


if __name__ == "__main__":
    app.run()
