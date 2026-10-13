"""TabPFN on the same 10 x 5 folds as scripts/run_cv.py, compared with the published
baseline (B1) and the deployed model (FL-acc) by corrected repeated k-fold t-tests.

TabPFN is used with default settings (no tuning): it is a pretrained in-context learner.
Two inputs are tested: the `acc+comp` features used by the other models, and the
seven raw count proportions.

usage: TABPFN_TOKEN=... python scripts/run_tabpfn.py ../pNT_TO_WAB.xlsx
(run scripts/run_cv.py first; its results/cv_folds.csv is reused)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pnt2wab import core  # noqa: E402

REPEATS, K = 10, 5
OUT = ROOT / "results"


VERSION = "v2"  # Hollmann et al. 2025 (Nature); v2.5+ weights need a Prior Labs license


def fit_predict(Ftr, Ytr, Fte):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    P = np.zeros((len(Fte), Ytr.shape[1]))
    for j in range(Ytr.shape[1]):
        m = TabPFNRegressor.create_default_for_version(ModelVersion(VERSION), device="cuda",
                                                       random_state=0)
        m.fit(Ftr, Ytr[:, j] / core.MAXES[j])
        P[:, j] = np.clip(m.predict(Fte), 0, 1) * core.MAXES[j]
    return P


def main(path):
    d, X, Y, C = core.load(path)
    inputs = {f"TabPFN{VERSION}-acc+comp": core.features(X, "acc+comp"),
              f"TabPFN{VERSION}-props": X / X.sum(1, keepdims=True),
              f"TabPFN{VERSION}-acc": X[:, :1] / X.sum(1, keepdims=True)}
    labels = core.KEYS + [k for k, _, _ in core.COMPOSITES]
    rows = []
    for r in range(REPEATS):
        for f, (tr, te) in enumerate(KFold(K, shuffle=True, random_state=42 + r).split(X)):
            Yt = np.hstack([Y[te], C[te]])
            for name, F in inputs.items():
                P = fit_predict(F[tr], Y[tr], F[te])
                P = np.hstack([P, core.composites(P)])
                for j, lab in enumerate(labels):
                    rows.append(dict(model=name, rep=r, fold=f, outcome=lab,
                                     r2=1 - ((Yt[:, j] - P[:, j]) ** 2).sum()
                                     / ((Yt[:, j] - Yt[:, j].mean()) ** 2).sum(),
                                     mae=np.abs(Yt[:, j] - P[:, j]).mean()))
        print(f"repeat {r + 1}/{REPEATS} done", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / f"tabpfn_{VERSION}_folds.csv", index=False)

    ref = pd.read_csv(OUT / "cv_folds.csv")[["model", "rep", "fold", "outcome", "r2"]]
    allf = pd.concat([ref, T[["model", "rep", "fold", "outcome", "r2"]]])
    n = len(X)
    corr = 1 / (REPEATS * K) + (n / K) / (n * (K - 1) / K)
    out = []
    for m in inputs:
        for lab in labels:
            a = allf[(allf.model == m) & (allf.outcome == lab)].sort_values(["rep", "fold"]).r2.to_numpy()
            row = dict(model=m, outcome=lab, r2=a.mean())
            for base in ("B1", "FL-acc"):
                b = allf[(allf.model == base) & (allf.outcome == lab)].sort_values(["rep", "fold"]).r2.to_numpy()
                diff = a - b
                t = diff.mean() / np.sqrt(corr * diff.var(ddof=1))
                row[f"d_vs_{base}"] = diff.mean()
                row[f"p_vs_{base}"] = 2 * stats.t.sf(abs(t), len(diff) - 1)
            out.append(row)
    S = pd.DataFrame(out)
    S.to_csv(OUT / f"tabpfn_{VERSION}_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(S.round(3).to_string(index=False))
    sub = allf[allf.outcome.isin(core.KEYS)].groupby("model").r2.mean().sort_values(ascending=False)
    print("\nmean R2 over 10 independent subtests:\n", sub.round(4).to_string())


if __name__ == "__main__":
    if len(sys.argv) > 2:
        VERSION = sys.argv[2]
    main(sys.argv[1])
