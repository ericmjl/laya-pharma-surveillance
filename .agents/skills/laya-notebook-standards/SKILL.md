---
name: laya-notebook-standards
description: Authoring standards for every notebook in the laya-pharma-surveillance repo — structure, naming, prose, language honesty, lens workflow, and self-verification. Use alongside laya-pharma-theme (colors) whenever creating or reviewing a notebook here.
license: MIT
---

# Notebook standards (learned from Acts 1-2)

## Structure

1. **Banner first**: full-width SVG hero (960x210), ink gradient, series kicker,
   act badge, ghost act numeral, one-metric motif. Copy an existing act's
   banner cell and swap number/title/tagline.
2. **Goal cell second**, exactly this pattern:

   > In this notebook, we will show ...
   >
   > By the end of the notebook, you will:
   >
   > - see ...
   > - see ...

   No "how to read this notebook" meta-paragraphs.

3. **Pedagogy prose around every code cell**: before it, say what is coming;
   after its output, interpret what we see. Written for a first-time reader.
4. **One moment per notebook**: build toward a single output that carries the
   act's point, and end on it.

## Naming and code hygiene

- Every cell has a unique snake_case name (set at creation; used for
  code_mode references).
- No `_underscore_private` names. Suffix counters instead (`lang2`, `fig_kh2`).
- Imports happen exactly once, in the `imports` cell. Never re-import per
  cell; downstream cells reference the imported name directly.
- All cells hidden (`hide_code=True`). Where readers should see code, add
  `mo.accordion` with an **action-oriented label** ("Click here to see the
  code that runs Laya"), showing the real code, not a paraphrase.

## Language

- Write prose in Eric's voice, direct, no AI tells. Every draft passes the
  unslop triple pass + `eric_slop_scan.py` before handoff.
- First person where Eric can verify ("I read Mandarin, so I can verify every
  word myself"); not second-person "you can check".
- Honest caveats are mandatory where truth is borrowed: machine-translations
  ("I asked AI to translate it and I cannot check it; we proceed assuming the
  translation is correct"), synthetic data ("labels are authorial ground
  truth"), and saturation artifacts ("confidence pins at 1.0 because the
  softmax saturates; the fourth decimal is where languages differ").
- Prefer examples the reader can verify (Mandarin over Khmer for Eric).
- When two results disagree for a real reason, say so and name where the
  deeper conversation happens (e.g. "a calibration conversation, notebook 6").
- Numbers in prose come from computed variables (f-strings), never hardcoded,
  so prose always matches the live output.

## Visuals

- Colors: Pharmacopeia theme only — see `laya-pharma-theme/SKILL.md`.
  CRIMSON is reserved for adverse/failure. TEAL is brand. No Moderna red.
- Plotly, never matplotlib. `template="plotly_white"`, values labeled
  outside bars, axes that start at zero, real distributions (box/swarm with
  all n points) instead of single mean bars.
- Fixed figure geometry when placement math depends on pixels (`width=760`).
- UI copy is action-oriented ("Click here to see...").
- A cell cannot read the `.value` of a UI element it creates: split controls
  and consumers into separate cells.

## Data honesty

- Every numeric claim in prose must be reproducible from `data/*.json` or a
  live cell. Verify before writing.
- Show failure cases, not just headline accuracy (the failure gallery is the
  lesson).
- Cache expensive live runs to `data/*.json` (delete to recompute), and
  commit the cache once verified.

## Lens workflow (marimo-lens)

- Read all comments first (`lens.agent.connect()` -> `context().text`), fix
  every one, then `ML.resolve(ids, expected_revision, summary=<240 chars>)` —
  **inside the same kernel invocation as the edits**. Editing the imports
  cell (or any heavy rerun) re-mounts the Lens widget and wipes the store;
  a deferred resolve finds nothing. Real selection ids are UUIDs from
  `references['selections']`, not "S1" labels. Summaries max 240 UTF-16 units.

## Self-verification before handoff

1. `ast.parse` the saved file; marimo graph clean (no MultipleDefinition /
   CycleError); zero cell errors, zero stale.
2. Run every cell (code_mode edits do not execute).
3. Screenshot via agent-browser and **read the image**: banner, key figures,
   widgets render correctly at 1440px.
4. Prose scans: unslop `banned_phrase_scan.py` + `eric_slop_scan.py` both 0.
5. `git status` clean; header PEP 723 pins match the running sandbox
   (marimo==0.25.1, laya==0.3.5, plotly==7.1.0, marimo-lens==0.2.3,
   marimo-studio==0.3.0; anywidget only if a widget is used).
