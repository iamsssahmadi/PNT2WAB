"""Write docs/validation.json: the cross-validated performance shown on the web tool.

All numbers come from the saved analysis results (no participant-level data):
fold-averaged R2 and MAE (revision_summary.csv), and the coverage of the interval method the
web tool uses (normalised conformal quantile, "cv-approx-norm-0.05"), marginally and by AQ
severity band. Only aggregate summaries are written.

usage: python scripts/export_site_data.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
BANDS = ["Very severe (0-25)", "Severe (26-50)", "Moderate (51-75)", "Mild (76-93.7)", "Above cut-off (>=93.8)"]


def main():
    s = pd.read_csv(RES / "revision_summary.csv")
    cov = pd.read_csv(RES / "revision2_coverage.csv")
    cov = cov[cov.method == "cv-approx-norm-0.05"]
    long = pd.read_csv(RES / "revision2_coverage_long.csv.gz")
    long = long[(long.alpha == 0.1) & (long.method == "cv-approx-norm-0.05")]
    band = long.groupby(["band", "outcome"]).agg(coverage=("covered", "mean"), n=("covered", "size")).reset_index()
    band["n"] = band.n / 10  # 10 cross-validation repeats
    fl, b1 = s[s.model == "FL-acc"].set_index("outcome"), s[s.model == "B1"].set_index("outcome")
    model = json.load(open(ROOT / "docs" / "model.json"))
    keys = [x["key"] for x in model["subtests"]] + [x["key"] for x in model["derived"]]
    out = {
        "source": "10 x repeated 5-fold nested cross-validation, N = 213 (fold-averaged R2; coverage of the intervals this tool uses).",
        "bands": BANDS,
        "band_n": {b: int(band[band.band == b].n.iloc[0]) for b in BANDS},
        "outcomes": [],
    }
    for k in keys:
        bb = band[band.outcome == k].set_index("band")
        out["outcomes"].append({
            "key": k,
            "r2": round(float(fl.loc[k, "r2_fold"]), 3),
            "r2_linear": round(float(b1.loc[k, "r2_fold"]), 3),
            "mae": round(float(fl.loc[k, "mae"]), 2),
            "coverage90": round(float(cov[(cov.alpha == 0.1) & (cov.outcome == k)].coverage.iloc[0]), 3),
            "coverage80": round(float(cov[(cov.alpha == 0.2) & (cov.outcome == k)].coverage.iloc[0]), 3),
            "coverage90_by_band": {b: round(float(bb.loc[b, "coverage"]), 3) for b in BANDS},
        })
    json.dump(out, open(ROOT / "docs" / "validation.json", "w"), indent=1)
    print("wrote docs/validation.json")


if __name__ == "__main__":
    main()
