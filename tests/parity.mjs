// Checks that docs/predict.js reproduces the Python model exactly.
// usage: node tests/parity.mjs
import { readFileSync } from "node:fs";
import { predict } from "../docs/predict.js";

const model = JSON.parse(readFileSync(new URL("../docs/model.json", import.meta.url)));
const cases = JSON.parse(readFileSync(new URL("./parity_cases.json", import.meta.url)));

let worst = 0;
for (const c of cases) {
  const r = predict(model, c.correct, c.total, "0.9");
  const pairs = [
    [r.subtests.map((s) => s.pred), c.pred], [r.subtests.map((s) => s.lo), c.lo],
    [r.subtests.map((s) => s.hi), c.hi], [r.derived.map((s) => s.pred), c.dpred],
    [r.derived.map((s) => s.lo), c.dlo], [r.derived.map((s) => s.hi), c.dhi],
  ];
  for (const [js, py] of pairs)
    js.forEach((v, i) => { worst = Math.max(worst, Math.abs(v - py[i])); });
}
console.log(`${cases.length} cases, max |JS - Python| = ${worst.toExponential(2)}`);
if (worst > 1e-6) { console.error("PARITY FAILED"); process.exit(1); }
console.log("parity OK");
