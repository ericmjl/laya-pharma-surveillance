---
name: laya-pharma-theme
description: Color theme and visual conventions for every notebook in the laya-pharma-surveillance repo. Use whenever creating or editing a notebook here — banners, plotly figures, tables, mermaid diagrams, or mail_viewer widget colors. Also the reference when porting the series to new notebooks.
license: MIT
---

# Pharmacopeia theme

The visual world of this repo is a **clinical chart**: calm, precise, cool,
with one warm alarm color reserved exclusively for adverse results. It is
built for pharmacovigilance content — drug safety, adverse-event
surveillance, decision models grading medical text.

**Hard rule: do not use Moderna's palette.** Moderna's brand red (#D2232A on
black/white) appears nowhere here. Our red is a darker clinical crimson
(#B3282D) with a narrow semantic job, never a brand color. The brand primary
is assay teal.

## The palette

Define these once per notebook, in the imports cell (see any existing
notebook's `imports` cell for the exact block):

| constant | hex | role | where it appears |
|---|---|---|---|
| `INK` | `#0F2233` | clinical navy | banner gradients, dark cards, table headers, marker outlines |
| `INK2` | `#16303F` | elevated ink | gradient end, chip fills on dark |
| `TEAL` | `#1F7A8C` | **assay teal — brand primary** | main data series, avatars, selection accents, links |
| `GREEN` | `#2F9E68` | recovery green | safe/selected/positive (selected-language highlight, license line) |
| `AMBER` | `#C9821F` | caution amber | urgency metrics, warning gates |
| `CRIMSON` | `#B3282D` | adverse crimson | adverse/danger ONLY (confident-collapse plots, mean diamonds, danger chips) |
| `SLATE` | `#8A97A5` | secondary series | comparison series (e.g. English checkpoint next to teal) |
| `MIST` | `#A7B3BD` | muted text on ink | banner taglines |
| `KICK` | `#8FB0B8` | kicker text on ink | banner kickers, dark-card labels |
| `LINE` | `#D9D4CA` | warm hairline | light-card borders |
| `PAPER` | `#F7F5F1` | lab paper | note cards, light backgrounds |

Text on ink uses `#EEF4FB` (near-white). Teal-tinted hairlines on ink:
`#2E5566` (borders/connectors), ghost text `#26404F`, muted captions on ink
`#6E93A0`.

## Semantic rules

1. **CRIMSON is reserved.** It means "the model failed / adverse event".
   Never use it for decoration, means, or highlights. This restraint is what
   makes the collapse plots land.
2. **TEAL is the brand.** Default data series, primary accents, selection
   chrome. When in doubt, teal.
3. **AMBER means urgency/warning** (the score head, the Router gate). GREEN
   means safe/selected. Don't swap them.
4. **SLATE is the de-emphasis color** for the comparison series (the thing
   that is not the point of the chart).
5. Light cards sit on `PAPER` with `LINE` borders; dark cards sit on `INK`
   with `#24404F` borders and `KICK` labels. Don't mix.
6. Window chrome in `mail_viewer.py` stays **macOS-native grey** (platform
   convention beats theme); only semantic elements (chips, avatars,
   selection, unread dot) take the theme.
7. Data must not rely on color alone: label values outside bars, use the
   selected-language highlight together with a title that names it, keep
   hover text naming the language.

## Where things live

- Banner SVGs: gradient stops (`INK`→`INK2` background, `TEAL`→`GREEN`→`CRIMSON`
  accent strip), chip fills `INK2`/stroke `#2E5566`, kicker `KICK`, ghost act
  numeral `#26404F`.
- Plotly: `template="plotly_white"`, values labeled `textposition="outside"`,
  series colors from the constants. Comparison charts: SLATE vs TEAL.
  Adverse-only charts: CRIMSON bars. Highlighted selection: GREEN among TEAL.
- Mermaid diagrams: `classDef` fills INK/INK2 with role strokes — router gate
  AMBER, multilingual TEAL, English SLATE, answers GREEN.
- `mail_viewer.py`: chips (dept TEAL, urgency AMBER, danger CRIMSON, safe
  GREEN), avatar gradient TEAL→INK, selection/unread TEAL, note cards PAPER.
- Dark-card text labels: `KICK`, values `#EEF4FB`.

## Adding a new notebook

Copy the imports-cell theme block from an existing notebook verbatim, reuse
the banner cell as the template (swap act number, title, tagline), and keep
every color a named constant — no inline hex in new code. If a new semantic
role is genuinely needed, add it to this skill first, then to the notebooks.
