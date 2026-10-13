"""Learning curve: cross-validated R2 as a function of training-set size (PNT profile only).

For every outer test fold (5-fold, repeated), the model is trained on random subsets
of the training partition of increasing size and scored on the full held-out fold.
A saturating power law R2(n) = a - b * n^(-c) is fitted to each model's mean curve
and extrapolated to larger samples.

usage: TABPFN_TOKEN=... python scripts/learning_curve.py ../pNT_TO_WAB.xlsx
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from pnt2wab import core  # noqa: E402
import run_tabpfn  # noqa: E402

run_tabpfn.VERSION = "v3.5"
OUT = ROOT / "results"
SIZES = [30, 45, 60, 80, 100, 125, 150, 170]
REPEATS = {"B1": 20, "FL-acc": 20, "TabPFN v3.5": 4}
LABELS = core.KEYS + [k for k, _, _ in core.COMPOSITES]


def fit_predict(name, Xtr, Ytr, Xte):
    if name == "B1":
        return core.OLS("pct").fit(Xtr, Ytr).predict(Xte)
    if name == "FL-acc":
        return core.FracLogit("acc", 0.1).fit(Xtr, Ytr).predict(Xte)
    props = lambda X: X / X.sum(1, keepdims=True)  # noqa: E731
    return run_tabpfn.fit_predict(props(Xtr), Ytr, props(Xte))


def power_law(n, a, b, c):
    return a - b * n ** (-c)


def main(path):
    d, X, Y, C = core.load(path)
    rows = []
    for name, reps in REPEATS.items():
        for r in range(reps):
            rng = np.random.default_rng(r)
            for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=42 + r).split(X)):
                Yt = np.hstack([Y[te], C[te]])
                for n in SIZES:
                    sub = tr if n >= len(tr) else rng.choice(tr, n, replace=False)
                    P = fit_predict(name, X[sub], Y[sub], X[te])
                    P = np.hstack([P, core.composites(P)])
                    r2 = 1 - ((Yt - P) ** 2).sum(0) / ((Yt - Yt.mean(0)) ** 2).sum(0)
                    for j, lab in enumerate(LABELS):
                        rows.append(dict(model=name, rep=r, fold=f, n=min(n, len(tr)),
                                         outcome=lab, r2=r2[j]))
            print(f"{name}: repeat {r + 1}/{reps}", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "learning_curve_folds.csv", index=False)

    # mean over the 10 independent subtests, per fold
    M = (D[D.outcome.isin(core.KEYS)].groupby(["model", "rep", "fold", "n"]).r2.mean()
         .reset_index())
    curve = M.groupby(["model", "n"]).r2.agg(["mean", "std", "count"]).reset_index()
    curve["ci"] = 1.96 * curve["std"] / np.sqrt(curve["count"])
    curve.to_csv(OUT / "learning_curve_mean.csv", index=False)

    extrap = []
    for name, g in curve.groupby("model"):
        p, _ = curve_fit(power_law, g.n, g["mean"], p0=(0.7, 1, 0.5),
                         bounds=([0, 0, 0.05], [1, 50, 3]), maxfev=20000)
        row = dict(model=name, asymptote=p[0], b=p[1], c=p[2])
        for n in (170, 250, 300, 400, 500, 1000):
            row[f"R2@{n}"] = power_law(n, *p)
        extrap.append(row)
    E = pd.DataFrame(extrap)
    E.to_csv(OUT / "learning_curve_extrapolation.csv", index=False)

    per = D[D.n == 170].groupby(["model", "outcome"]).r2.mean().unstack(0)
    small = D[D.n == 60].groupby(["model", "outcome"]).r2.mean().unstack(0)
    print(curve.pivot(index="n", columns="model", values="mean").round(3).to_string())
    print("\nextrapolation (power law fitted on n = 30..170):\n", E.round(3).to_string(index=False))
    print("\nper outcome, gain from n = 60 to n = 170:\n",
          (per - small).loc[LABELS].round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
