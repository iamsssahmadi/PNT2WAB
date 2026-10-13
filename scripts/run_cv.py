"""Nested, repeated cross-validation of every model in ANALYSIS_PLAN.md.

usage: python scripts/run_cv.py path/to/pNT_TO_WAB.xlsx
"""
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pnt2wab import core  # noqa: E402

REPEATS, K, INNER = 10, 5, 5
OUT = Path(__file__).resolve().parents[1] / "results"
COMP_MAX = np.array([m for _, _, m in core.COMPOSITES], float)

GRIDS = {
    "B0": [lambda: core.Mean()],
    "B1": [lambda: core.OLS("pct")],
    "B3": [lambda: core.OLS("pct+prop")],
    "FL-acc": [lambda l=l: core.FracLogit("acc", l) for l in (0.1, 1, 10, 100)],
    "FL-comp": [lambda l=l: core.FracLogit("acc+comp", l) for l in (0.1, 1, 10, 100)],
    "MTL": [lambda h=h, w=w: core.MTL("acc+comp", h, w)
            for h, w in itertools.product((4, 8), (1e-3, 1e-2, 1e-1))],
    "CB": [lambda d=d: core.CB("acc+comp", d) for d in (3, 5)],
}


def r2(Y, P):
    return 1 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def inner_oof(make, X, Y, seed):
    P = np.zeros_like(Y)
    for tr, te in KFold(INNER, shuffle=True, random_state=seed).split(X):
        P[te] = make().fit(X[tr], Y[tr]).predict(X[te])
    return P


def outer_fold(rep, fold, tr, te, X, Y):
    import torch
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    res = {}
    for name, grid in GRIDS.items():
        best, best_score, best_oof = grid[0], -np.inf, None
        if len(grid) > 1 or name != "B0":
            for make in grid:
                oof = inner_oof(make, X[tr], Y[tr], seed=1000 * rep + fold)
                s = np.mean(r2(Y[tr], oof))
                if s > best_score:
                    best, best_score, best_oof = make, s, oof
        m = best().fit(X[tr], Y[tr])
        P = m.predict(X[te])
        q = {}
        if best_oof is not None:
            Cin, Cin_p = core.composites(Y[tr]), core.composites(best_oof)
            for a in (0.1, 0.2):
                q[a] = (core.conformal_q(Y[tr], best_oof, core.MAXES, a),
                        core.conformal_q(Cin, Cin_p, COMP_MAX, a))
        res[name] = dict(P=P, q=q, hp=getattr(m, "lam", None) or getattr(m, "wd", None)
                         or getattr(m, "depth", None))
    return rep, fold, te, res


def main(path):
    OUT.mkdir(exist_ok=True)
    d, X, Y, C = core.load(path)
    n = len(X)
    jobs = [(r, f, tr, te) for r in range(REPEATS)
            for f, (tr, te) in enumerate(KFold(K, shuffle=True, random_state=42 + r).split(X))]
    out = Parallel(n_jobs=min(24, len(jobs)), verbose=5)(
        delayed(outer_fold)(r, f, tr, te, X, Y) for r, f, tr, te in jobs)

    names = list(GRIDS)
    labels = core.KEYS + [k for k, _, _ in core.COMPOSITES]
    rows, cover = [], []
    oof = {m: np.zeros((REPEATS, n, len(labels))) for m in names}
    for rep, fold, te, res in out:
        Yt = np.hstack([Y[te], C[te]])
        for m in names:
            P = np.hstack([res[m]["P"], core.composites(res[m]["P"])])
            oof[m][rep, te] = P
            for j, lab in enumerate(labels):
                rows.append(dict(model=m, rep=rep, fold=fold, outcome=lab,
                                 se=((Yt[:, j] - P[:, j]) ** 2).sum(),
                                 ae=np.abs(Yt[:, j] - P[:, j]).sum(),
                                 ss=((Yt[:, j] - Yt[:, j].mean()) ** 2).sum(), n=len(te),
                                 hp=res[m]["hp"]))
            for a, (qs, qc) in res[m]["q"].items():
                lo, hi = core.interval(res[m]["P"], qs, core.MAXES)
                Pc = core.composites(res[m]["P"])
                clo, chi = core.interval(Pc, qc, COMP_MAX)
                lo, hi = np.hstack([lo, clo]), np.hstack([hi, chi])
                for j, lab in enumerate(labels):
                    cover.append(dict(model=m, alpha=a, outcome=lab,
                                      covered=((Yt[:, j] >= lo[:, j] - 1e-9)
                                               & (Yt[:, j] <= hi[:, j] + 1e-9)).sum(),
                                      width=(hi[:, j] - lo[:, j]).sum(), n=len(te)))
    R = pd.DataFrame(rows)
    R["r2"] = 1 - R.se / R.ss
    R.to_csv(OUT / "cv_folds.csv", index=False)
    pd.DataFrame(cover).to_csv(OUT / "cv_coverage.csv", index=False)
    np.savez_compressed(OUT / "cv_oof.npz", **oof, Y=np.hstack([Y, C]))

    # summary table + corrected repeated k-fold t-test vs B1
    n_tr, n_te = n * (K - 1) / K, n / K
    summ = []
    for (m, lab), g in R.groupby(["model", "outcome"], sort=False):
        b = R[(R.model == "B1") & (R.outcome == lab)].sort_values(["rep", "fold"])
        g = g.sort_values(["rep", "fold"])
        diff = g.r2.to_numpy() - b.r2.to_numpy()
        kr = len(diff)
        sd = diff.std(ddof=1)
        if sd > 0:
            t = diff.mean() / np.sqrt((1 / kr + n_te / n_tr) * sd ** 2)
            p = 2 * stats.t.sf(abs(t), kr - 1)
            half = stats.t.ppf(.975, kr - 1) * np.sqrt((1 / kr + n_te / n_tr) * sd ** 2)
        else:
            p, half = np.nan, np.nan
        Yall = np.hstack([Y, C])[:, labels.index(lab)]
        slopes = [np.polyfit(oof[m][r, :, labels.index(lab)], Yall, 1)[0]
                  if oof[m][r, :, labels.index(lab)].std() > 0 else np.nan
                  for r in range(REPEATS)]
        summ.append(dict(model=m, outcome=lab, r2=g.r2.mean(),
                         mae=g.ae.sum() / g.n.sum(), rmse=np.sqrt(g.se.sum() / g.n.sum()),
                         d_r2_vs_B1=diff.mean(), ci_lo=diff.mean() - half,
                         ci_hi=diff.mean() + half, p_vs_B1=p,
                         calib_slope=np.nanmean(slopes)))
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "cv_summary.csv", index=False)
    Cv = pd.DataFrame(cover).groupby(["model", "alpha", "outcome"], sort=False).sum()
    Cv["coverage"] = Cv.covered / Cv.n
    Cv["mean_width"] = Cv.width / Cv.n
    Cv[["coverage", "mean_width"]].to_csv(OUT / "cv_coverage_summary.csv")

    sub = S[S.outcome.isin(core.KEYS)].groupby("model", sort=False).r2.mean()
    print("\nmean R2 over 10 independent subtests:\n", sub.round(4).to_string())
    json.dump(sub.to_dict(), open(OUT / "cv_mean_r2.json", "w"), indent=1)
    print(S.pivot(index="outcome", columns="model", values="r2").round(3).loc[labels].to_string())


if __name__ == "__main__":
    main(sys.argv[1])
