# /// script
# requires-python = ">=3.10"
# dependencies = ["laya==0.3.5", "huggingface_hub"]
# ///
"""Before/after fine-tuning experiment on the real pharma document corpus.

Setup: ~282 real docs (50-doc corpus + large script-fetched corpus), stratified
80/20 train/test split. Five questions per doc (primary type, secondary type, route,
action-required noul, urgency), labels from our authorial ground truth.

Fine-tuning: laptop-scale reimplementation of the RLCD loop using Laya's own
pipeline functions (rl_common.py from the HF repo, vendored in
laya_training_src/): one forward pass gives option logits; G Gaussian-noise
perturbations form a candidate group; each candidate is scored with
proper_reward (log + spherical + RPS - strictly proper, so honest
probabilities maximize reward); group-normalized advantages weight the
differentiable reward (GRPO-style), plus a small soft-CE anchor for stability
on tiny data. Full encoder fine-tune, AdamW lr=1e-5, 3 epochs, MPS.

Reported on TEST: argmax accuracy per question, mean log score, Brier, and
ECE for the noul - each BEFORE vs AFTER, raw and with post-hoc per-bucket
temperature fitting (fitted on TRAIN logits, exactly like the shipped
checkpoints do).

Honesty: n=50 means this demonstrates mechanics and direction, not
production-grade numbers. Expect overfitting pressure; watch whether
fine-tuning sharpens probabilities (good) while shipping them over-confident
(known RLFT behavior - which is exactly what the temperature refit then fixes).
"""

import json
import math
import os
import random
import sys

os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "laya_training_src"))
from rl_common import (QTYPES, build_model, collate_items, ece_score, encode_record,  # noqa: E402
                       proper_reward, temp_bucket)

from real_docs_corpus import CATEGORIES, DOCS  # noqa: E402
from real_docs_corpus_large import LARGE_DOCS  # noqa: E402

ALL_DOCS = DOCS + LARGE_DOCS  # original 50 (incl. 5 synthetic spam) + large script-fetched corpus

SEED = 0
EPOCHS = 3
LR = 1e-5
G = 4          # noise candidates per question (the "group" in GRPO-style)
SIGMA = 0.3    # Gaussian logit-noise scale (exploration)
BATCH = 12     # items per step
CE_ANCHOR = 0.3
MAX_LEN, HEAD_MAX_LEN = 512, 192
OUT = "finetune_results_run.json"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {device} | corpus: {len(DOCS)} docs")

# ------------------------------------------------------------------ schema (matches doc_triage_real.py)
CATS = CATEGORIES + ["none"]  # secondary includes none
ROUTE_KEYS = ["regulatory_affairs", "quality", "pharmacovigilance", "medical_affairs",
              "legal", "communications", "executive", "archive"]
QUESTIONS = {
    "doc_type": {"t": "choice", "ins": "What kind of document is this?", "crit": {
        "regulatory_guidance": "guidance, rules, or requirement notices from a health authority",
        "inspection_finding": "inspections, 483 observations, warning letters, GMP findings",
        "safety_communication": "drug safety alerts, label changes, recalls, adverse event warnings",
        "scientific_publication": "journal articles, preprints, conference abstracts and presentations",
        "legal": "petitions, lawsuits, settlements, enforcement, counsel correspondence",
        "press_coverage": "news articles, media requests for comment",
        "commercial": "vendor outreach, product marketing, sponsorships, cold sales",
        "public_comment": "comments or letters submitted to a public docket or agency",
        "corporate_report": "company filings, annual reports, risk factor disclosures",
        "spam_or_irrelevant": "mass mailings, awards, phishing, unrelated content"}},
    "secondary_type": {"t": "choice", "ins": "If this document also fits a second category, which one?",
                       "crit": {c: c.replace("_", " ") for c in CATEGORIES} | {"none": "the first category alone captures it"}},
    "route_to": {"t": "choice", "ins": "Which team should own this document?", "crit": {k: k.replace("_", " ") for k in ROUTE_KEYS}},
    "action_required": {"t": "noul", "ins": "Should someone on our team take an action or make a decision about this document?", "crit": {}},
    "urgency": {"t": "choice", "ins": "How quickly must someone act on this?",
                "crit": {"routine": "no deadline", "this_week": "deadline within two weeks", "same_day": "deadline within 24-48 hours"}},
}
QIDS = list(QUESTIONS)


def doc_records(docs):
    """Corpus docs -> rl_common record format (internal keys t/ins/crit/y)."""
    recs = []
    for d in docs:
        qs = []
        for qid in QIDS:
            q = dict(QUESTIONS[qid])
            keys = list(q["crit"].keys())
            if qid == "doc_type":
                q["y"] = keys.index(d["cat"])
            elif qid == "secondary_type":
                q["y"] = keys.index(d["cat2"] or "none")
            elif qid == "route_to":
                q["y"] = keys.index(d["team"])
            elif qid == "action_required":
                q["y"] = 1 if d["action"] else 0
            else:
                q["y"] = d["urg"]
            recs_q = {"state": {"source": d["source"], "title": d["title"], "body": d["text"]}, "qs": [q], "src": d["id"]}
            qs.append(recs_q)
        recs.extend(qs)  # one record per question keeps item counts balanced across questions
    return recs


# stratified 80/20 split: >=2 test docs per category, overall ~20% test
by_cat = {}
for d in ALL_DOCS:
    by_cat.setdefault(d["cat"], []).append(d)
test_ids, train_docs = set(), []
for cat in sorted(by_cat):
    pool = by_cat[cat][:]
    random.shuffle(pool)
    n_test = max(2, round(0.2 * len(pool)))
    test_ids.update(p["id"] for p in pool[:n_test])
    train_docs.extend(pool[n_test:])
test_docs = [d for d in ALL_DOCS if d["id"] in test_ids]
print(f"split: {len(train_docs)} train / {len(test_docs)} test docs -> {len(train_docs)*5} / {len(test_docs)*5} question items")

# ------------------------------------------------------------------ model + tokenizer (from the HF repo)
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

tok_dir = os.path.join(snapshot_download("convaiinnovations/laya", allow_patterns=["tokenizer/*"]), "tokenizer")
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(tok_dir)

cfg = json.load(open(os.path.join(os.path.dirname(__file__), "laya_training_src", "rl_agent_config.json")))
model = build_model(cfg)  # loads answerdotai/ModernBERT-large (HF cache)
sd_path = hf_hub_download("convaiinnovations/laya", "model.safetensors")
from safetensors.torch import load_file  # noqa: E402

model.load_state_dict(load_file(sd_path), strict=True)
model.to(device).eval()
model.encoder.config.reference_compile = False
print(f"model loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")


# ------------------------------------------------------------------ helpers
def encode(recs, train_mode):
    items = []
    for rec in recs:
        rng = random.Random(SEED) if not train_mode else random.Random(random.getrandbits(32))
        items.extend(encode_record(rec, tok, {"max_len": MAX_LEN, "head_max_len": HEAD_MAX_LEN, "max_prefixes": 1}, rng, train_mode))
    return items


def forward_logits(items):
    """Batched no-grad forward -> per-item logits [K] (padded groups by length)."""
    out = [None] * len(items)
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    for s in range(0, len(order), BATCH):
        sel = [items[i] for i in order[s:s + BATCH]]
        b = collate_items([sel], tok.pad_token_id)
        with torch.no_grad():
            logits, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                              b["marker_pos"].to(device), b["marker_mask"].to(device), b["qtype"].to(device))
        logits = logits.float().cpu()
        for r, it in enumerate(sel):
            k = len(it["markers"])
            out[order[s + r]] = logits[r, :k]
    return out


def train_epoch(items):
    random.shuffle(items)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_r, steps = 0.0, 0
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    # shuffle within length-sorted chunks to keep padding cheap
    chunks = [order[s:s + BATCH * 4] for s in range(0, len(order), BATCH * 4)]
    random.shuffle(chunks)
    batch_idx = [i for c in chunks for i in c]
    for s in range(0, len(batch_idx), BATCH):
        sel = [items[i] for i in batch_idx[s:s + BATCH]]
        b = collate_items([sel], tok.pad_token_id)
        logits, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                          b["marker_pos"].to(device), b["marker_mask"].to(device), b["qtype"].to(device))
        logits = logits.float()
        target = b["target"].to(device)
        mask = b["marker_mask"].to(device)
        qt = b["qtype"].to(device)
        # RLCD: G noisy logit candidates per item, group-relative advantages, proper-scoring reward
        z = logits.unsqueeze(0).repeat(G, 1, 1)                     # [G, N, K]
        z = z + SIGMA * torch.randn_like(z)
        p = torch.softmax(z, -1)
        tgt = target.unsqueeze(0).expand(G, -1, -1)
        msk = mask.unsqueeze(0).expand(G, -1, -1)
        qtg = qt.unsqueeze(0).expand(G, -1)
        r = proper_reward(p, tgt, qtg, msk)                        # [G, N]
        adv = (r - r.mean(0, keepdim=True)) / (r.std(0, keepdim=True) + 1e-4)
        rl_loss = -(adv.detach() * r).mean()
        # soft-CE anchor (log score) - the component the typed-decisions recipe also mixes in
        logp = torch.log_softmax(logits, -1)
        ce = -(target * logp).sum(-1).mean()  # target is 0 on padded option columns
        loss = rl_loss + CE_ANCHOR * ce
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_r += float(r.mean())
        steps += 1
    model.eval()
    return total_r / max(1, steps)


def metrics_for(items, logits_list, temps=None):
    """Per-question metrics. temps: optional {bucket: T} applied to logits before softmax."""
    per_q = {qid: {"p": [], "y": []} for qid in QIDS}
    for it, z in zip(items, logits_list):
        qid = it["_qid"]
        k = len(it["markers"])
        T = (temps or {}).get(temp_bucket(it["qtype"], k), 1.0)
        p = torch.softmax(z / T, -1).numpy()
        per_q[qid]["p"].append(p)
        per_q[qid]["y"].append(int(it["label"]))
    res = {}
    for qid, d in per_q.items():
        P = np.array(d["p"])
        Y = np.array(d["y"])
        k = P.shape[1]
        nll = -np.mean(np.log(np.clip(P[np.arange(len(Y)), Y], 1e-9, 1)))
        onehot = np.eye(k)[Y]
        brier = float(np.mean(np.sum((P - onehot) ** 2, -1)))
        acc = float((P.argmax(-1) == Y).mean())
        entry = {"n": len(Y), "log_score": float(-nll), "brier": brier, "acc": acc}
        if QUESTIONS[qid]["t"] == "noul":
            conf = np.where(P[:, 1] >= 0.5, P[:, 1], 1 - P[:, 1])
            correct = (P[:, 1] >= 0.5).astype(float) == Y
            entry["ece"] = ece_score(conf.astype(np.float64), correct.astype(np.float64), bins=5)
        res[qid] = entry
    return res


# attach question id to items (encode_record keeps meta; simplest: parallel walk)
def tag_qid(items, recs):
    # encode_record flattens one item per record here (each rec has exactly 1 question)
    assert len(items) == len(recs), (len(items), len(recs))
    for it, rec in zip(items, recs):
        it["_qid"] = rec["qs"][0]["t"] + ":" + rec["qs"][0]["ins"][:20]
    return items


# map to stable qid keys
def tag_by_records(items, recs):
    ins2qid = {(QUESTIONS[q]["t"], QUESTIONS[q]["ins"]): q for q in QIDS}
    for it, rec in zip(items, recs):
        q = rec["qs"][0]
        it["_qid"] = ins2qid[(q["t"], q["ins"])]
    return items


train_recs = doc_records(train_docs)
test_recs = doc_records(test_docs)
train_items = tag_by_records(encode(train_recs, train_mode=True), train_recs)
test_items = tag_by_records(encode(test_recs, train_mode=False), test_recs)
print(f"encoded: {len(train_items)} train / {len(test_items)} test items")

# ------------------------------------------------------------------ BEFORE
base_train_logits = forward_logits(train_items)
base_test_logits = forward_logits(test_items)
def fit_temperature(logits_list, labels):
    """Scalar T minimizing NLL of softmax(z/T)."""
    best_T, best_nll = 1.0, math.inf
    for T in np.geomspace(0.05, 100, 4000):
        z = np.stack([l / T for l in logits_list])
        p = np.exp(z - z.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        nll = -np.mean(np.log(np.clip(p[np.arange(len(labels)), labels], 1e-9, 1)))
        if nll < best_nll:
            best_T, best_nll = float(T), nll
    return best_T


temps_before = {}
for qid in QIDS:
    idxs = [i for i, it in enumerate(train_items) if it["_qid"] == qid]
    buckets = {}
    for i in idxs:
        k = len(train_items[i]["markers"])
        buckets.setdefault(temp_bucket(train_items[i]["qtype"], k), ([], []))
        buckets[temp_bucket(train_items[i]["qtype"], k)][0].append(base_train_logits[i].numpy())
        buckets[temp_bucket(train_items[i]["qtype"], k)][1].append(train_items[i]["label"])
    for b, (ls, ys) in buckets.items():
        if len(set(ys)) > 1:
            temps_before[b] = fit_temperature(ls, ys)
print(f"\nfitted temperatures BEFORE (train split): { {k: round(v,3) for k,v in temps_before.items()} }")

before_raw = metrics_for(test_items, base_test_logits)
before_T = metrics_for(test_items, base_test_logits, temps_before)

# ------------------------------------------------------------------ FINE-TUNE
print("\nfine-tuning (RLCD-style, full encoder, MPS)...")
CURVE = []
for epoch in range(EPOCHS):
    r = train_epoch(train_items)
    print(f"  epoch {epoch+1}/{EPOCHS}: train mean reward {r:.4f}")
    probe = metrics_for(test_items, forward_logits(test_items))
    point = {
        "epoch": epoch + 1,
        "train_mean_reward": round(r, 4),
        "test_acc_mean": round(sum(probe[q]["acc"] for q in QIDS) / len(QIDS), 4),
        "test_brier_mean": round(sum(probe[q]["brier"] for q in QIDS) / len(QIDS), 4),
    }
    CURVE.append(point)
    print(f"    curve: {point}")

# ------------------------------------------------------------------ AFTER
ft_train_logits = forward_logits(train_items)
ft_test_logits = forward_logits(test_items)
temps_after = {}
for qid in QIDS:
    idxs = [i for i, it in enumerate(train_items) if it["_qid"] == qid]
    buckets = {}
    for i in idxs:
        k = len(train_items[i]["markers"])
        buckets.setdefault(temp_bucket(train_items[i]["qtype"], k), ([], []))
        buckets[temp_bucket(train_items[i]["qtype"], k)][0].append(ft_train_logits[i].numpy())
        buckets[temp_bucket(train_items[i]["qtype"], k)][1].append(train_items[i]["label"])
    for b, (ls, ys) in buckets.items():
        if len(set(ys)) > 1:
            temps_after[b] = fit_temperature(ls, ys)
print(f"fitted temperatures AFTER (train split): { {k: round(v,3) for k,v in temps_after.items()} }\n")

after_raw = metrics_for(test_items, ft_test_logits)
after_T = metrics_for(test_items, ft_test_logits, temps_after)

# ------------------------------------------------------------------ curve out
from pathlib import Path as _Path

_curve_out = _Path(__file__).resolve().parent.parent / "data" / "finetune_training_curve.json"
_curve_out.write_text(json.dumps({"epochs": CURVE}, indent=1))
print(f"training curve written: {_curve_out}")

# ------------------------------------------------------------------ report
print("=" * 100)
print(f"TEST RESULTS ({len(test_items)} items, {len(test_docs)} docs) - before vs after fine-tuning")
print("=" * 100)
print(f"{'question':<18}{'metric':<12}{'before':>9}{'after':>9}   {'before+T':>9}{'after+T':>9}")
for qid in QIDS:
    b, a = before_raw[qid], after_raw[qid]
    bT, aT = before_T[qid], after_T[qid]
    print(f"{qid:<18}{'acc':<12}{b['acc']:>9.2f}{a['acc']:>9.2f}")
    print(f"{'':<18}{'log_score':<12}{b['log_score']:>9.3f}{a['log_score']:>9.3f}   (T-refit shifts probs, not acc)")
    print(f"{'':<18}{'brier':<12}{b['brier']:>9.3f}{a['brier']:>9.3f}")
    if "ece" in b:
        print(f"{'':<18}{'ECE(noul)':<12}{b['ece']:>9.3f}{a['ece']:>9.3f}   {bT['ece']:>9.3f}{aT['ece']:>9.3f}")
    print()

# probability shift on two test docs
print("probability shift example (first two test docs, doc_type distribution):")
for d in test_docs[:2]:
    pairs = [(it, zb, za) for it, zb, za in zip(test_items, base_test_logits, ft_test_logits)
             if it["_qid"] == "doc_type" and it.get("src") == d["id"]]
    for it, zb, za in pairs:
        keys = list(QUESTIONS["doc_type"]["crit"].keys())
        pb = torch.softmax(zb, -1).numpy()
        pa = torch.softmax(za, -1).numpy()
        print(f"  {d['id']} (gt={d['cat']}):")
        top = sorted(range(len(keys)), key=lambda i: -max(pb[i], pa[i]))[:4]
        for i in top:
            print(f"    {keys[i]:<24} before={pb[i]:.3f}  after={pa[i]:.3f}")

with open(OUT, "w") as f:
    json.dump({"before_raw": before_raw, "after_raw": after_raw, "before_T": before_T,
               "after_T": after_T, "temps_before": temps_before, "temps_after": temps_after,
               "train_docs": [d["id"] for d in train_docs], "test_docs": [d["id"] for d in test_docs]}, f, indent=1)
print(f"\nsaved: {OUT}")
