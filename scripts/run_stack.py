"""Stacking TabPFN v3.5 (PNT proportions) with the bounded model (FL-acc).

Inputs are the PNT profile only. Per outer fold and per subtest, the blend weight
w in [0, 1] (prediction = w * TabPFN + (1 - w) * FL) is learned from inner 5-fold
out-of-fold predictions on the training partition, so the test fold is never used.
A fixed 50/50 average (no tuning) is reported alongside.

usage: TABPFN_TOKEN=... python scripts/run_stack.py ../pNT_TO_WAB.xlsx
(reuses results/cv_folds.csv and results/tabpfn_v3.5_folds.csv for comparisons)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from pnt2wab import core  # noqa: E402
import run_tabpfn  # noqa: E402

run_tabpfn.VERSION = "v3.5"
REPEATS, K, INNER, LAM = 10, 5, 5, 0.1
OUT = ROOT / "results"
GRID = np.linspace(0, 1, 21)


def base_preds(Xtr, Ytr, Xte):
    props = lambda X: X / X.sum(1, keepdims=True)  # noqa: E731
    pt = run_tabpfn.fit_predict(props(Xtr), Ytr, props(Xte))
    pf = core.FracLogit("acc", LAM).fit(Xtr, Ytr).predict(Xte)
    return pt, pf


def main(path):
    d, X, Y, C = core.load(path)
    labels = core.KEYS + [k for k, _, _ in core.COMPOSITES]
    rows, weights = [], []
    for r in range(REPEATS):
        for f, (tr, te) in enumerate(KFold(K, shuffle=True, random_state=42 + r).split(X)):
            ot, of = np.zeros_like(Y[tr]), np.zeros_like(Y[tr])
            for a, b in KFold(INNER, shuffle=True, random_state=1000 + r).split(tr):
                ot[b], of[b] = base_preds(X[tr][a], Y[tr][a], X[tr][b])
            mse = np.array([((Y[tr] - (w * ot + (1 - w) * of)) ** 2).mean(0) for w in GRID])
            w = GRID[mse.argmin(0)]
            weights.append(w)
            pt, pf = base_preds(X[tr], Y[tr], X[te])
            Yt = np.hstack([Y[te], C[te]])
            for name, P in (("Stack-learned", w * pt + (1 - w) * pf), ("Stack-50/50", (pt + pf) / 2)):
                P = np.hstack([P, core.composites(P)])
                for j, lab in enumerate(labels):
                    rows.append(dict(model=name, rep=r, fold=f, outcome=lab,
                                     r2=1 - ((Yt[:, j] - P[:, j]) ** 2).sum()
                                     / ((Yt[:, j] - Yt[:, j].mean()) ** 2).sum(),
                                     mae=np.abs(Yt[:, j] - P[:, j]).mean()))
        print(f"repeat {r + 1}/{REPEATS} done", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "stack_folds.csv", index=False)
    W = pd.DataFrame(weights, columns=core.KEYS)
    W.describe().loc[["mean", "min", "max"]].round(2).to_csv(OUT / "stack_weights.csv")

    ref = pd.concat([pd.read_csv(OUT / "cv_folds.csv"), pd.read_csv(OUT / "tabpfn_v3.5_folds.csv")])
    allf = pd.concat([ref[["model", "rep", "fold", "outcome", "r2", "mae"]],
                      T[["model", "rep", "fold", "outcome", "r2", "mae"]]])
    n = len(X)
    corr = 1 / (REPEATS * K) + (n / K) / (n * (K - 1) / K)
    get = lambda m, lab: allf[(allf.model == m) & (allf.outcome == lab)].sort_values(["rep", "fold"])  # noqa: E731
    out = []
    for m in ("Stack-learned", "Stack-50/50"):
        for lab in labels:
            a = get(m, lab)
            row = dict(model=m, outcome=lab, r2=a.r2.mean(), mae=a.mae.mean())
            for base in ("B1", "FL-acc", "TabPFNv3.5-props"):
                diff = a.r2.to_numpy() - get(base, lab).r2.to_numpy()
                t = diff.mean() / np.sqrt(corr * diff.var(ddof=1))
                row[f"d_{base}"] = diff.mean()
                row[f"p_{base}"] = 2 * stats.t.sf(abs(t), len(diff) - 1)
            out.append(row)
    S = pd.DataFrame(out)
    S.to_csv(OUT / "stack_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(S.round(3).to_string(index=False))
    print("\nlearned TabPFN weight per subtest (mean over folds):\n", W.mean().round(2).to_string())
    sub = allf[allf.outcome.isin(core.KEYS)].groupby("model").r2.mean().sort_values(ascending=False)
    print("\nmean R2 over 10 independent subtests:\n", sub.round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
