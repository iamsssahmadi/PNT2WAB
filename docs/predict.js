// Pure prediction logic shared by the web page and tests/parity.mjs.
// Mirrors pnt2wab/core.py (FracLogit "acc" features) and scripts/fit_final.py
// (derived composites and AQ; normalised conformal-quantile prediction intervals).

const sig = (z) => 1 / (1 + Math.exp(-z));
const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

function spread(p, max, floor) {
  const r = clip(p / max, 0, 1);
  return max * (Math.sqrt(r * (1 - r)) + floor);
}

export function validateInput(correct, total) {
  if (!Number.isFinite(correct) || !Number.isFinite(total)) return "Enter numbers only.";
  if (total < 1) return "Items administered must be at least 1.";
  if (correct < 0 || correct > total) return `Correct must be between 0 and ${total}.`;
  return null;
}

// Subtest predictions of one linear-logit model ({scaler_mean, scaler_sd, coef}).
function subtestPred(m, maxes, correct, total) {
  const a = Math.log((correct + 0.5) / (total - correct + 0.5));
  const f = [a, a * a, a * a * a].map((v, i) => (v - m.scaler_mean[i]) / m.scaler_sd[i]);
  return m.coef.map((b, j) => sig(b[0] + b[1] * f[0] + b[2] * f[1] + b[3] * f[2]) * maxes[j]);
}

// Composites and AQ from the 10 subtests (WAB-R formulas), in model.derived order.
function derivedFrom(sub, keys) {
  const v = Object.fromEntries(keys.map((k, j) => [k, sub[j]]));
  const ss = v.information_content + v.fluency;
  const comp = (v.yes_no + v.auditory_words + v.sequential_commands) / 20;
  const nam = (v.object_naming + v.word_fluency + v.sentence_completion + v.responsive_speech) / 10;
  return [ss, comp, nam, 2 * (ss + comp + v.repetition + nam)];
}

// Returns { subtests: [{key,label,max,pred,lo,hi}], derived: [...] }.
// level is "0.9" or "0.8" (central prediction-interval coverage).
// Intervals: prediction +/- q * spread(prediction), where q is the stored conformal quantile of
// normalised out-of-fold residuals (one aggregate number per outcome and level).
export function predict(model, correct, total = 175, level = "0.9") {
  const maxes = model.subtests.map((s) => s.max);
  const keys = model.subtests.map((s) => s.key);
  const sub = subtestPred({ ...model.features, coef: model.coef }, maxes, correct, total);
  const der = derivedFrom(sub, keys);
  const floor = model.conformal.spread_floor;
  const q = model.conformal.q[level];
  const band = (pred, max, qq) => {
    const w = qq * spread(pred, max, floor);
    return { pred, lo: clip(pred - w, 0, max), hi: clip(pred + w, 0, max) };
  };
  return {
    subtests: model.subtests.map((s, j) => ({ ...s, ...band(sub[j], s.max, q.subtests[j]) })),
    derived: model.derived.map((d, j) => ({ ...d, ...band(der[j], d.max, q.derived[j]) })),
  };
}
