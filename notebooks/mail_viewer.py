"""Shared macOS Mail-style viewer widget for the Laya pharma surveillance notebooks.

Colors follow the Pharmacopeia theme (see .agents/skills/laya-pharma-theme/SKILL.md):
teal #1F7A8C primary, amber #C9821F caution, crimson #B3282D adverse, green #2F9E68 safe.
Window chrome stays macOS-native by design.

Import from the notebooks directory:

    from mail_viewer import MailViewer
"""

import anywidget
import traitlets


class MailViewer(anywidget.AnyWidget):
    """macOS Mail-style viewer for Laya-triaged email."""

    _esm = """
function render({ model, el }) {
  const esc = (s) => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  let emails = model.get("emails") || [];
  let sel = Math.min(model.get("selected") || 0, Math.max(emails.length - 1, 0));

  const chip = (label, value, color) =>
    '<span style="display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:999px;background:' + color + '16;border:1px solid ' + color + '55;font-size:12px;">' +
    '<span style="width:7px;height:7px;border-radius:50%;background:' + color + '"></span><b>' + esc(label) + '</b>&nbsp;' + esc(value) + '</span>';

  const answerChips = (a) => {
    if (!a) return "";
    const parts = [];
    if (a.department) parts.push(chip("dept", a.department.choice + " · " + a.department.confidence.toFixed(2), "#1F7A8C"));
    if (a.urgency) parts.push(chip("urgency", a.urgency.score.toFixed(2), "#C9821F"));
    if (a.is_phishing) parts.push(chip("phishing", a.is_phishing.noul.toFixed(3), a.is_phishing.noul >= 0.5 ? "#B3282D" : "#2F9E68"));
    if (a.churn_risk) parts.push(chip("churn", a.churn_risk.noul.toFixed(3), a.churn_risk.noul >= 0.5 ? "#B3282D" : "#2F9E68"));
    return parts.join(" ");
  };

  const reader = (m) => `
    <div style="padding:14px 18px 10px;">
      <div style="font-size:17px;font-weight:700;letter-spacing:-0.2px;">${esc(m.subject)}</div>
      <div style="display:flex;align-items:center;gap:10px;margin-top:10px;">
        <div style="width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,#1F7A8C,#0F2233);color:#fff;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;flex-shrink:0;">${esc(m.initials)}</div>
        <div style="flex:1;min-width:0;">
          <div style="font-size:13px;font-weight:600;">${esc(m.sender_name)} <span style="font-weight:400;color:#86868b;">&lt;${esc(m.sender_email)}&gt;</span></div>
          <div style="font-size:12px;color:#86868b;">to you@acme-health.com</div>
        </div>
        <div style="font-size:12px;color:#86868b;">${esc(m.time || "")}</div>
      </div>
      ${m.answers ? '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;">' + answerChips(m.answers) + '</div>' : ""}
      <div style="font-size:13.5px;line-height:1.5;white-space:pre-wrap;padding:14px 0 4px;">${esc(m.body)}</div>
      ${m.note ? '<div style="font-size:12px;color:#51515b;background:#F7F5F1;border-radius:8px;padding:8px 12px;margin-top:8px;">' + esc(m.note) + '</div>' : ""}
      ${m.ms ? '<div style="font-size:11px;color:#86868b;border-top:1px solid #ececf0;padding-top:8px;margin-top:6px;">triaged by convaiinnovations/laya in ' + esc(m.ms) + ' ms · one forward pass, zero generated text</div>' : ""}
    </div>`;

  const row = (m, i) => `
    <div data-i="${i}" style="cursor:pointer;display:flex;gap:10px;padding:10px 12px;border-bottom:1px solid #ececf0;background:${i === sel ? "#1F7A8C" : "transparent"};color:${i === sel ? "#fff" : "inherit"};">
      <div style="width:34px;height:34px;border-radius:50%;background:${i === sel ? "rgba(255,255,255,.28)" : "linear-gradient(135deg,#1F7A8C,#0F2233)"};color:#fff;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;flex-shrink:0;">${esc(m.initials)}</div>
      <div style="flex:1;min-width:0;">
        <div style="display:flex;justify-content:space-between;gap:8px;">
          <span style="font-size:13px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${esc(m.sender_name)}</span>
          <span style="font-size:11px;opacity:.75;flex-shrink:0;">${esc(m.time || "")}</span>
        </div>
        <div style="font-size:12.5px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${esc(m.subject)}</div>
        <div style="font-size:12px;opacity:.7;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${esc(m.preview)}</div>
      </div>
    </div>`;

  const wrap = document.createElement("div");
  wrap.style.cssText = "border-radius:12px;overflow:hidden;border:1px solid #d9d9de;box-shadow:0 1px 3px rgba(0,0,0,.08);background:#fff;";
  el.style.fontFamily = "-apple-system,BlinkMacSystemFont,'SF Pro Text',Helvetica,Arial,sans-serif";
  el.appendChild(wrap);

  const draw = () => {
    const m = emails[sel] || {};
    const listHtml = emails.length > 1
      ? '<div style="width:270px;flex-shrink:0;border-right:1px solid #e5e5ea;background:#fbfbfd;max-height:540px;overflow-y:auto;">' +
        '<div style="padding:10px 12px;font-size:12px;font-weight:700;color:#86868b;text-transform:uppercase;letter-spacing:.5px;">Inbox</div>' +
        emails.map(row).join("") + '</div>'
      : "";
    wrap.innerHTML =
      '<div style="display:flex;align-items:center;gap:8px;padding:9px 14px;background:linear-gradient(#f6f6f8,#ececef);border-bottom:1px solid #d9d9de;">' +
      '<span style="width:12px;height:12px;border-radius:50%;background:#ff5f57;display:inline-block;"></span>' +
      '<span style="width:12px;height:12px;border-radius:50%;background:#febc2e;display:inline-block;"></span>' +
      '<span style="width:12px;height:12px;border-radius:50%;background:#28c840;display:inline-block;"></span>' +
      '<span style="margin:0 auto;font-size:13px;font-weight:600;color:#6e6e73;">Inbox — Laya Triage</span>' +
      '<span style="width:64px;"></span></div>' +
      '<div style="display:flex;min-height:280px;">' + listHtml +
      '<div style="flex:1;min-width:0;">' + reader(m) + '</div></div>';
    wrap.querySelectorAll("[data-i]").forEach((n) =>
      n.addEventListener("click", () => {
        sel = parseInt(n.dataset.i, 10);
        model.set("selected", sel);
        model.save_changes();
        draw();
      })
    );
  };

  draw();
  model.on("change:emails", () => { emails = model.get("emails") || []; sel = 0; draw(); });
}

export default { render };
"""

    emails = traitlets.List().tag(sync=True)
    selected = traitlets.Int(0).tag(sync=True)


class SampleDocs(anywidget.AnyWidget):
    """Random per-category document sampler: reshuffles 3 docs per category, live."""

    _esm = """
function render({ model, el }) {
  const esc = (s) => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const docs = model.get("docs") || [];
  const perCat = {};
  docs.forEach((d) => { (perCat[d.category] = perCat[d.category] || []).push(d); });
  const cats = Object.keys(perCat).sort();
  el.style.fontFamily = "-apple-system,BlinkMacSystemFont,'SF Pro Text',Helvetica,Arial,sans-serif";

  const wrap = document.createElement("div");
  el.appendChild(wrap);

  const sample3 = (arr) => {
    const c = arr.slice();
    for (let i = c.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [c[i], c[j]] = [c[j], c[i]];
    }
    return c.slice(0, 3);
  };

  const draw = () => {
    wrap.innerHTML =
      '<button style="cursor:pointer;border:1px solid #D9D4CA;background:#fff;border-radius:8px;padding:7px 14px;font-size:12.5px;font-weight:600;color:#1D2530;">Reshuffle sample</button>' +
      '<div style="margin-top:6px;font-size:11.5px;color:#86868b;">three random documents per category, drawn live in your browser</div>' +
      cats.map((cat) =>
        '<div style="margin-top:14px;">' +
        '<div style="font-size:12px;font-weight:700;color:#5C6B7A;text-transform:uppercase;letter-spacing:.5px;">' + esc(cat) + '</div>' +
        sample3(perCat[cat]).map((d) =>
          '<div style="padding:8px 0;border-bottom:1px solid #EFEDE7;">' +
          '<div style="font-size:13px;font-weight:600;color:#1D2530;">' + esc(d.title) + '</div>' +
          '<div style="font-size:11.5px;color:#86868b;">' + esc(d.source) + '</div>' +
          '<div style="font-size:12.5px;color:#3D4450;margin-top:2px;">' + esc(d.snippet) + '</div>' +
          '</div>'
        ).join("") +
        '</div>'
      ).join("");
    wrap.querySelector("button").addEventListener("click", draw);
  };

  draw();
}

export default { render };
"""

    docs = traitlets.List().tag(sync=True)


class DocViewer(anywidget.AnyWidget):
    """Mail-style two-pane viewer for misclassified documents: list + reader."""

    _esm = """
function render({ model, el }) {
  const esc = (s) => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  let docs = model.get("docs") || [];
  let sel = Math.min(model.get("selected") || 0, Math.max(docs.length - 1, 0));

  const reader = (m) => `
    <div style="padding:14px 18px 10px;">
      <div style="font-size:16px;font-weight:700;letter-spacing:-0.2px;">${esc(m.title)}</div>
      <div style="display:flex;align-items:center;gap:10px;margin-top:10px;">
        <div style="width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#1F7A8C,#0F2233);color:#fff;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;flex-shrink:0;">${esc(m.id)}</div>
        <div style="flex:1;min-width:0;">
          <div style="font-size:12.5px;color:#5C6B7A;">${esc(m.source)}</div>
        </div>
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;font-family:ui-monospace,Menlo,monospace;font-size:11.5px;">
        <span style="padding:3px 10px;border-radius:999px;background:#1F7A8C16;border:1px solid #1F7A8C55;color:#1D2530;">label: <b>${esc(m.label)}</b></span>
        <span style="padding:3px 10px;border-radius:999px;background:#B3282D16;border:1px solid #B3282D55;color:#1D2530;">predicted: <b>${esc(m.predicted)}</b> (P=${esc(m.prob)})</span>
      </div>
      <div style="font-size:13px;line-height:1.55;white-space:pre-wrap;padding:12px 0 4px;color:#1D2530;">${esc(m.body)}</div>
    </div>`;

  const row = (m, i) => `
    <div data-i="${i}" style="cursor:pointer;display:flex;gap:8px;padding:9px 12px;border-bottom:1px solid #EFEDE7;background:${i === sel ? "#1F7A8C" : "transparent"};color:${i === sel ? "#fff" : "inherit"};">
      <div style="flex:1;min-width:0;">
        <div style="font-size:12.5px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${esc(m.title)}</div>
        <div style="font-size:11.5px;opacity:.75;">${esc(m.label)} -> ${esc(m.predicted)}</div>
      </div>
    </div>`;

  const wrap = document.createElement("div");
  wrap.style.cssText = "border-radius:12px;overflow:hidden;border:1px solid #D9D4CA;box-shadow:0 1px 3px rgba(0,0,0,.08);background:#fff;";
  el.style.fontFamily = "-apple-system,BlinkMacSystemFont,'SF Pro Text',Helvetica,Arial,sans-serif";
  el.appendChild(wrap);

  const draw = () => {
    const m = docs[sel] || {};
    wrap.innerHTML =
      '<div style="padding:9px 14px;background:linear-gradient(#f6f6f8,#ececef);border-bottom:1px solid #d9d9de;font-size:13px;font-weight:600;color:#6e6e73;">Failing samples - ' + docs.length + ' total</div>' +
      '<div style="display:flex;min-height:300px;">' +
      '<div style="width:280px;flex-shrink:0;border-right:1px solid #EFEDE7;background:#fbfbfd;max-height:520px;overflow-y:auto;">' + docs.map(row).join("") + '</div>' +
      '<div style="flex:1;min-width:0;">' + reader(m) + '</div></div>';
    wrap.querySelectorAll("[data-i]").forEach((n) =>
      n.addEventListener("click", () => {
        sel = parseInt(n.dataset.i, 10);
        model.set("selected", sel);
        model.save_changes();
        draw();
      })
    );
  };

  draw();
  model.on("change:docs", () => { docs = model.get("docs") || []; sel = 0; draw(); });
}

export default { render };
"""

    docs = traitlets.List().tag(sync=True)
    selected = traitlets.Int(0).tag(sync=True)
