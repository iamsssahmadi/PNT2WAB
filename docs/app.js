// User interface for the PNT -> WAB-R estimator. All computation is local.
import { predict, validateInput } from "./predict.js";

const $ = (id) => document.getElementById(id);
const fmt = (x, d = 1) => (Math.round(x * 10 ** d) / 10 ** d).toFixed(d);
const pct = (x) => `${Math.round(x * 100)}%`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const DOMAINS = [
  ["Naming & word finding", ["object_naming", "word_fluency", "sentence_completion", "responsive_speech"]],
  ["Spontaneous speech", ["information_content", "fluency"]],
  ["Repetition", ["repetition"]],
  ["Auditory comprehension", ["yes_no", "auditory_words", "sequential_commands"]],
];

let MODEL, VAL, valByKey = {};

function severity(aq) {
  if (aq >= 93.8) return "Above the aphasia cut-off";
  if (aq >= 76) return "Mild";
  if (aq > 50) return "Moderate";
  if (aq > 25) return "Severe";
  return "Very severe";
}

function accuracyTag(key) {
  const v = valByKey[key];
  if (!v) return "";
  const r2 = Math.round(v.r2 * 100) / 100;  // classify the value that is displayed
  const cls = r2 >= 0.75 ? "good" : r2 >= 0.6 ? "mid" : "low";
  const word = r2 >= 0.75 ? "good" : r2 >= 0.6 ? "moderate" : "limited";
  return `<span class="tag ${cls}" title="Cross-validated R² = ${v.r2.toFixed(2)}">${word} · R² ${v.r2.toFixed(2)}</span>`;
}

function bar(s) {
  const w = (v) => `${Math.max(0, Math.min(100, (v / s.max) * 100))}%`;
  return `<div class="bar" role="img" aria-label="${esc(s.label)}: ${fmt(s.pred)} (interval ${fmt(s.lo)} to ${fmt(s.hi)}) of ${s.max}">
    <div class="rng" style="left:${w(s.lo)};width:calc(${w(s.hi)} - ${w(s.lo)})"></div>
    <div class="pt" style="left:${w(s.pred)}"></div></div>`;
}

// ---------------------------------------------------------------- single
function renderSingle() {
  const correct = Number($("correct").value), total = Number($("total").value), level = $("level").value;
  const err = validateInput(correct, total);
  $("single-error").hidden = !err;
  $("single-error").textContent = err || "";
  $("single-out").hidden = Boolean(err);
  if (err) return;
  $("total-warn").hidden = total === 175;
  document.querySelectorAll(".lvl").forEach((e) => { e.textContent = pct(Number(level)); });

  const r = predict(MODEL, correct, total, level);
  const byKey = Object.fromEntries([...r.subtests, ...r.derived].map((s) => [s.key, s]));
  const aq = byKey.aq;
  $("hero").innerHTML = `<div class="k">Estimated Aphasia Quotient (${pct(Number(level))} interval)</div>
    <div class="v">${fmt(aq.pred)}</div><div class="r">${fmt(aq.lo)} – ${fmt(aq.hi)} of 100</div>
    <div class="sev">${severity(aq.pred)}</div>`;
  const sw = $("severity-warn");
  sw.hidden = aq.pred > 50;
  sw.textContent = aq.pred > 50 ? "" : "Predicted severity is severe or very severe. In validation, comprehension intervals in this range "
    + "contained the true score in as few as 60–79% of cases (Auditory Word Recognition, Yes/No) instead of 90%. Assess comprehension directly.";

  $("composites").innerHTML = r.derived.filter((d) => d.key !== "aq").map((d) => `<div class="card">
    <div class="k">${esc(d.label)} (0–${d.max})</div><div class="v">${fmt(d.pred)}</div>
    <div class="r">${fmt(d.lo)} – ${fmt(d.hi)} · ${accuracyTag(d.key)}</div></div>`).join("");

  const rows = DOMAINS.map(([name, keys]) => `<tr class="group"><td colspan="5">${name}</td></tr>` + keys.map((k) => {
    const s = byKey[k];
    return `<tr><th scope="row">${esc(s.label)} <span class="hint">/${s.max}</span><span class="tag-inline">${accuracyTag(k)}</span></th>
      <td class="num">${fmt(s.pred)}</td><td class="num">${fmt(s.lo)} – ${fmt(s.hi)}</td>
      <td class="barcol">${bar(s)}</td><td class="acc">${accuracyTag(k)}</td></tr>`;
  }).join("")).join("");
  $("subtests").querySelector("tbody").innerHTML = rows;
}

function summaryText() {
  const correct = Number($("correct").value), total = Number($("total").value), level = $("level").value;
  const r = predict(MODEL, correct, total, level);
  const lines = [`PNT ${correct}/${total} correct. Estimated WAB-R (${pct(Number(level))} prediction intervals; PNT-to-WAB-R estimator v${MODEL.version}):`];
  for (const s of [...r.derived.slice(3), ...r.derived.slice(0, 3), ...r.subtests]) {
    lines.push(`  ${s.label}: ${fmt(s.pred)} (${fmt(s.lo)}–${fmt(s.hi)}) of ${s.max}`);
  }
  lines.push("Estimates only; comprehension is poorly predicted from naming and should be assessed directly.");
  return lines.join("\n");
}

// ---------------------------------------------------------------- batch
function parseCSV(text) {
  const rows = [];
  let row = [], field = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') q = false;
      else field += c;
    } else if (c === '"') q = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.some((v) => v.trim() !== "")) rows.push(row);
      row = [];
    } else field += c;
  }
  row.push(field);
  if (row.some((v) => v.trim() !== "")) rows.push(row);
  return rows;
}

const csvCell = (v) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
let batchCSV = "";

function runBatch(text) {
  const out = $("batch-out"), errEl = $("batch-error");
  errEl.hidden = true;
  out.hidden = true;
  const rows = parseCSV(text.replace(/^﻿/, ""));
  if (rows.length < 2) { errEl.textContent = "The file has no data rows."; errEl.hidden = false; return; }
  const head = rows[0].map((h) => h.trim().toLowerCase());
  const iId = head.indexOf("id"), iC = head.indexOf("pnt_correct"), iT = head.indexOf("pnt_total");
  if (iId < 0 || iC < 0) { errEl.textContent = "Required columns: id, pnt_correct (optional: pnt_total)."; errEl.hidden = false; return; }
  const level = $("batch-level").value;
  const keys = [...MODEL.derived.slice(3), ...MODEL.derived.slice(0, 3), ...MODEL.subtests].map((s) => s.key);
  const labels = Object.fromEntries([...MODEL.subtests, ...MODEL.derived].map((s) => [s.key, s.label]));
  const header = ["id", "pnt_correct", "pnt_total", "status"];
  keys.forEach((k) => header.push(k, `${k}_lo`, `${k}_hi`));
  const outRows = [header];
  let ok = 0, bad = 0;
  for (const r of rows.slice(1)) {
    const id = (r[iId] ?? "").trim();
    const c = Number((r[iC] ?? "").trim());
    const t = iT >= 0 && (r[iT] ?? "").trim() !== "" ? Number(r[iT].trim()) : 175;
    const e = validateInput(c, t);
    if (e) { bad++; outRows.push([id, r[iC] ?? "", t, e]); continue; }
    ok++;
    const p = predict(MODEL, c, t, level);
    const byKey = Object.fromEntries([...p.subtests, ...p.derived].map((s) => [s.key, s]));
    const line = [id, c, t, t === 175 ? "ok" : "ok (non-standard length)"];
    keys.forEach((k) => line.push(fmt(byKey[k].pred, 2), fmt(byKey[k].lo, 2), fmt(byKey[k].hi, 2)));
    outRows.push(line);
  }
  batchCSV = outRows.map((r) => r.map(csvCell).join(",")).join("\n");
  $("batch-summary").textContent = `${ok} row${ok === 1 ? "" : "s"} estimated` + (bad ? `, ${bad} skipped (see status column).` : ".")
    + ` Intervals: ${pct(Number(level))}.`;
  const show = ["aq", "naming", "object_naming", "information_content", "comprehension"];
  const prev = outRows.slice(1, 21);
  $("batch-preview").innerHTML = `<thead><tr><th>id</th><th class="num">PNT</th>${show.map((k) => `<th class="num">${esc(labels[k])}</th>`).join("")}<th>status</th></tr></thead><tbody>`
    + prev.map((r) => {
      if (r.length <= 4) return `<tr><td>${esc(r[0])}</td><td class="num">${esc(r[1])}</td>${show.map(() => "<td></td>").join("")}<td>${esc(r[3])}</td></tr>`;
      const cell = (k) => { const j = header.indexOf(k); return `<td class="num">${r[j]} <span class="hint">(${r[j + 1]}–${r[j + 2]})</span></td>`; };
      return `<tr><td>${esc(r[0])}</td><td class="num">${r[1]}/${r[2]}</td>${show.map(cell).join("")}<td>${esc(r[3])}</td></tr>`;
    }).join("") + "</tbody>";
  if (outRows.length > 21) $("batch-summary").textContent += " Preview shows the first 20 rows; the download contains all.";
  out.hidden = false;
}

function download(name, text) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ---------------------------------------------------------------- about
function renderAbout() {
  $("val-source").textContent = VAL.source;
  const labels = Object.fromEntries([...MODEL.subtests, ...MODEL.derived].map((s) => [s.key, s.label]));
  const order = ["aq", "naming", "spontaneous_speech", "comprehension", ...DOMAINS.flatMap(([, k]) => k)];
  $("val-table").innerHTML = `<thead><tr><th>Outcome</th><th class="num">R² (this model)</th><th class="num">R² (linear)</th>
    <th class="num">Mean abs. error</th><th class="num">90% coverage</th><th class="num">80% coverage</th></tr></thead><tbody>`
    + order.map((k) => { const v = valByKey[k]; return `<tr><td>${esc(labels[k])}</td><td class="num">${v.r2.toFixed(2)}</td>
      <td class="num">${v.r2_linear.toFixed(2)}</td><td class="num">${v.mae.toFixed(2)}</td>
      <td class="num">${pct(v.coverage90)}</td><td class="num">${pct(v.coverage80)}</td></tr>`; }).join("") + "</tbody>";
  const shortBand = (b) => b.replace(" (", "<br><span class=\"hint\">").replace(")", "</span>");
  $("band-table").innerHTML = `<thead><tr><th>Outcome</th>${VAL.bands.map((b) => `<th class="num">${shortBand(b)}<br><span class="hint">n = ${VAL.band_n[b]}</span></th>`).join("")}</tr></thead><tbody>`
    + order.map((k) => `<tr><td>${esc(labels[k])}</td>` + VAL.bands.map((b) => {
      const c = valByKey[k].coverage90_by_band[b];
      return `<td class="num ${c < 0.85 ? "lo" : c > 0.95 ? "hi" : "ok"}">${pct(c)}</td>`;
    }).join("") + "</tr>").join("") + "</tbody>";
}

// ---------------------------------------------------------------- tabs
function setupTabs() {
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  const select = (t) => {
    tabs.forEach((x) => {
      const on = x === t;
      x.setAttribute("aria-selected", String(on));
      x.tabIndex = on ? 0 : -1;
      $(x.getAttribute("aria-controls")).hidden = !on;
    });
  };
  tabs.forEach((t, i) => {
    t.addEventListener("click", () => select(t));
    t.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      const n = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
      select(n);
      n.focus();
    });
  });
}

// ---------------------------------------------------------------- boot
async function main() {
  setupTabs();
  try {
    [MODEL, VAL] = await Promise.all(["model.json", "validation.json"].map((f) => fetch(f).then((r) => {
      if (!r.ok) throw new Error(`${f}: HTTP ${r.status}`);
      return r.json();
    })));
  } catch (e) {
    $("single-error").textContent = `Could not load the model (${e.message}). If you opened this file directly, serve the folder over HTTP instead.`;
    $("single-error").hidden = false;
    return;
  }
  valByKey = Object.fromEntries(VAL.outcomes.map((o) => [o.key, o]));
  $("ver").textContent = MODEL.version;
  const repo = document.querySelector('meta[name="repository"]');
  if (repo) $("repo-link").href = repo.content;

  ["correct", "total", "level"].forEach((id) => $(id).addEventListener("input", renderSingle));
  $("single-form").addEventListener("submit", (e) => e.preventDefault());
  $("copy").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(summaryText());
      $("copied").hidden = false;
      setTimeout(() => { $("copied").hidden = true; }, 2000);
    } catch { /* clipboard unavailable */ }
  });
  $("csv").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (f) runBatch(await f.text());
  });
  $("batch-level").addEventListener("change", () => { const f = $("csv").files[0]; if (f) f.text().then(runBatch); });
  $("download").addEventListener("click", () => download("pnt_wab_estimates.csv", batchCSV));
  $("template").addEventListener("click", () => download("pnt_wab_template.csv", "id,pnt_correct,pnt_total\nP001,120,175\nP002,45,175\n"));
  renderSingle();
  renderAbout();
}

main();
