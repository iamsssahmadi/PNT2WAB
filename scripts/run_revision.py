"""Revision analyses (same outer folds and inner tuning as scripts/run_cv.py).

1. NL-Ridge: ridge regression on the same three accuracy terms as FL-acc, identity link,
   lambda tuned by inner 5-fold CV. Unclipped (primary) and clipped to the subtest range.
2. Fold-averaged vs pooled out-of-fold R2 for B1, NL-Ridge and FL-acc, including the AQ.
3. Prediction intervals: the cross-validated normalised calibration used in the paper,
   the same with standard (unnormalised) scores, and a strict split-conformal variant
   (70/30 split of each outer training partition) that carries a finite-sample guarantee.

usage: python scripts/run_revision.py ../pNT_TO_WAB.xlsx
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
from fit_final import derived, DERIVED  # noqa: E402

REPEATS, K, INNER = 10, 5, 5
OUT = ROOT / "results"
DMAX = np.array([m for *_, m in DERIVED], float)
LABELS = core.KEYS + [k for k, *_ in DERIVED]  # 10 subtests + SS, COMP, NAM, AQ


class NLRidge:
    """Ridge on standardised [a, a^2, a^3] (logit accuracy), identity link."""

    def __init__(self, lam=1.0, clip=False):
        self.lam, self.clip = lam, clip

    def fit(self, X, Y):
        F = core.features(X, "acc")
        self.sc = core.Scaler().fit(F)
        Z = self.sc(F)
        self.mu = Y.mean(0)
        A = Z.T @ Z + self.lam * np.eye(Z.shape[1])
        self.B = np.linalg.solve(A, Z.T @ (Y - self.mu))
        return self

    def predict(self, X):
        P = self.sc(core.features(X, "acc")) @ self.B + self.mu
        return np.clip(P, 0, core.MAXES) if self.clip else P


GRIDS = {
    "B1": [lambda: core.OLS("pct")],
    "NL-Ridge": [lambda l=l: NLRidge(l) for l in (0.01, 0.1, 1, 10, 100)],
    "NL-Ridge-clip": [lambda l=l: NLRidge(l, clip=True) for l in (0.01, 0.1, 1, 10, 100)],
    "FL-acc": [lambda l=l: core.FracLogit("acc", l) for l in (0.1, 1, 10, 100)],
}


def r2cols(Y, P):
    return 1 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def inner_oof(make, X, Y, seed):
    P = np.zeros_like(Y)
    for tr, te in KFold(INNER, shuffle=True, random_state=seed).split(X):
        P[te] = make().fit(X[tr], Y[tr]).predict(X[te])
    return P


def q_level(n, alpha):
    return min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)


def std_q(Yc, Pc, alpha):
    s = np.abs(Yc - Pc)
    return np.quantile(s, q_level(len(s), alpha), axis=0, method="higher")


def main(path):
    d, X, Y, _ = core.load(path)
    n = len(X)
    Yall = np.hstack([Y, derived(Y)])
    rows, cover = [], []
    oof = {m: np.zeros((REPEATS, n, len(LABELS))) for m in GRIDS}
    for r in range(REPEATS):
        for f, (tr, te) in enumerate(KFold(K, shuffle=True, random_state=42 + r).split(X)):
            seed = 1000 * r + f
            for name, grid in GRIDS.items():
                best, bs, boof = None, -np.inf, None
                for make in grid:
                    o = inner_oof(make, X[tr], Y[tr], seed)
                    s = np.mean(r2cols(Y[tr], o))
                    if s > bs:
                        best, bs, boof = make, s, o
                m = best().fit(X[tr], Y[tr])
                P = m.predict(X[te])
                Pfull = np.hstack([P, derived(P)])
                oof[name][r, te] = Pfull
                Yt = Yall[te]
                r2 = r2cols(Yt, Pfull)
                for j, lab in enumerate(LABELS):
                    rows.append(dict(model=name, rep=r, fold=f, outcome=lab, r2=r2[j],
                                     ae=np.abs(Yt[:, j] - Pfull[:, j]).mean(),
                                     hp=getattr(m, "lam", None)))
                # ---- intervals (subtests + derived), alpha = 0.1 and 0.2
                Yc, Pc = np.hstack([Y[tr], derived(Y[tr])]), np.hstack([boof, derived(boof)])
                mx = np.r_[core.MAXES, DMAX]
                for a in (0.1, 0.2):
                    variants = {}
                    qn = core.conformal_q(Yc, Pc, mx, a)  # normalised, cross-validated calibration
                    variants["cv-normalised"] = core.interval(Pfull, qn, mx)
                    qs = std_q(Yc, Pc, a)  # unnormalised, cross-validated calibration
                    variants["cv-standard"] = (np.clip(Pfull - qs, 0, mx), np.clip(Pfull + qs, 0, mx))
                    if name in ("FL-acc", "NL-Ridge"):  # strict split conformal: 70% fit / 30% calibrate
                        rng = np.random.default_rng(seed)
                        perm = rng.permutation(tr)
                        cut = int(round(0.7 * len(tr)))
                        fi, ci = perm[:cut], perm[cut:]
                        lam = getattr(m, "lam", None)
                        ms = (core.FracLogit("acc", lam) if name == "FL-acc" else NLRidge(lam)).fit(X[fi], Y[fi])
                        Pcal = ms.predict(X[ci])
                        Pcal = np.hstack([Pcal, derived(Pcal)])
                        Ycal = np.hstack([Y[ci], derived(Y[ci])])
                        Ps = ms.predict(X[te])
                        Ps = np.hstack([Ps, derived(Ps)])
                        variants["split-normalised"] = core.interval(Ps, core.conformal_q(Ycal, Pcal, mx, a), mx)
                    for v, (lo, hi) in variants.items():
                        inside = (Yt >= lo - 1e-9) & (Yt <= hi + 1e-9)
                        for j, lab in enumerate(LABELS):
                            cover.append(dict(model=name, method=v, alpha=a, outcome=lab, n=len(te),
                                              covered=inside[:, j].sum(), width=(hi - lo)[:, j].sum()))
        print(f"repeat {r + 1}/{REPEATS}", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "revision_folds.csv", index=False)
    Cv = pd.DataFrame(cover).groupby(["model", "method", "alpha", "outcome"], sort=False)[
        ["covered", "width", "n"]].sum()
    Cv["coverage"], Cv["mean_width"] = Cv.covered / Cv.n, Cv.width / Cv.n
    Cv[["coverage", "mean_width"]].to_csv(OUT / "revision_intervals.csv")

    # sanity: FL-acc and B1 fold R2 must match the original run
    orig = pd.read_csv(OUT / "cv_folds.csv")
    for m in ("B1", "FL-acc"):
        a = R[(R.model == m) & R.outcome.isin(core.KEYS)].sort_values(["rep", "fold", "outcome"]).r2.to_numpy()
        b = orig[(orig.model == m) & orig.outcome.isin(core.KEYS)].sort_values(["rep", "fold", "outcome"]).r2.to_numpy()
        print(f"{m}: max |R2 diff| vs original run = {np.abs(a - b).max():.2e}")

    corr = 1 / (REPEATS * K) + (n / K) / (n * (K - 1) / K)
    get = lambda m, lab: R[(R.model == m) & (R.outcome == lab)].sort_values(["rep", "fold"])  # noqa: E731
    summ = []
    for m in GRIDS:
        for j, lab in enumerate(LABELS):
            g = get(m, lab)
            pooled = [1 - ((Yall[:, j] - oof[m][r, :, j]) ** 2).sum() / ((Yall[:, j] - Yall[:, j].mean()) ** 2).sum()
                      for r in range(REPEATS)]
            slope = np.mean([np.polyfit(oof[m][r, :, j], Yall[:, j], 1)[0] for r in range(REPEATS)])
            row = dict(model=m, outcome=lab, r2_fold=g.r2.mean(), r2_fold_sd=g.r2.std(ddof=1),
                       r2_pooled=np.mean(pooled), mae=g.ae.mean(), calib_slope=slope)
            for base in ("B1", "NL-Ridge"):
                if base == m:
                    continue
                diff = g.r2.to_numpy() - get(base, lab).r2.to_numpy()
                se = np.sqrt(corr * diff.var(ddof=1))
                half = stats.t.ppf(.975, len(diff) - 1) * se
                row[f"d_vs_{base}"] = diff.mean()
                row[f"lo_vs_{base}"], row[f"hi_vs_{base}"] = diff.mean() - half, diff.mean() + half
                row[f"p_vs_{base}"] = 2 * stats.t.sf(abs(diff.mean() / se), len(diff) - 1)
            summ.append(row)
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "revision_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.round(3).to_string(index=False))
    print("\nmean over 10 subtests (fold-averaged R2):\n",
          S[S.outcome.isin(core.KEYS)].groupby("model", sort=False)[["r2_fold", "r2_pooled"]].mean().round(4))
    print("\nselected lambda:", R.drop_duplicates(["model", "rep", "fold"]).groupby("model").hp.value_counts().to_dict())
    c = Cv.reset_index()
    print("\nintervals, alpha=0.1, 10 subtests:\n",
          c[(c.alpha == 0.1) & c.outcome.isin(core.KEYS)].groupby(["model", "method"])
          .agg(cov_min=("coverage", "min"), cov_max=("coverage", "max"), cov_mean=("coverage", "mean")).round(3))


if __name__ == "__main__":
    main(sys.argv[1])
