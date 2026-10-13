"""Second-round revision analyses (same outer folds as scripts/run_cv.py).

1. FL-spline: fractional-logit on a restricted cubic spline of proportion correct
   (knots at the 5th/35th/65th/95th percentiles of the training fold) — a lower-leverage
   alternative to the cubic in logit accuracy; nested tuning, fold-level comparisons,
   monotonicity of the full-data fit.
2. Intervals for FL-acc: normalised CV+ (primary; Barber et al., 2021; Gupta et al., 2022),
   unnormalised CV+, sensitivity of the spread floor (0.01-0.20); coverage and width
   overall, by observed AQ severity band and by recorded aphasia type.
3. Calibration-in-the-large and calibration slope (pooled out-of-fold, per repeat).
4. TOST equivalence of FL-acc vs NL-Ridge (margin +/-0.02 R2), from revision_folds.csv.
5. Learning-curve endpoint check (first 10 repeats, n = 170).

usage: python scripts/run_revision2.py ../pNT_TO_WAB.xlsx
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
from run_revision import NLRidge  # noqa: E402

REPEATS, K, INNER = 10, 5, 5
OUT = ROOT / "results"
DMAX = np.array([m for *_, m in DERIVED], float)
MX = np.r_[core.MAXES, DMAX]
LABELS = core.KEYS + [k for k, *_ in DERIVED]
FLOORS = (0.01, 0.05, 0.10, 0.20)


def rcs(x, knots):
    """Restricted (natural) cubic spline basis, Harrell parameterisation."""
    k = np.asarray(knots)
    t = len(k)
    pos = lambda u: np.clip(u, 0, None) ** 3  # noqa: E731
    cols = [x]
    for j in range(t - 2):
        cols.append(pos(x - k[j]) - pos(x - k[t - 2]) * (k[t - 1] - k[j]) / (k[t - 1] - k[t - 2])
                    + pos(x - k[t - 1]) * (k[t - 2] - k[j]) / (k[t - 1] - k[t - 2]))
    return np.column_stack(cols) / 1.0


class FLSpline(core.FracLogit):
    """Fractional-logit GLM on an RCS of proportion correct (4 knots from the training data)."""

    def __init__(self, lam=1.0):
        super().__init__("acc", lam)

    def _feat(self, X):
        p = X[:, 0] / X.sum(1)
        return rcs(p, self.knots)

    def fit(self, X, Y):
        p = X[:, 0] / X.sum(1)
        self.knots = np.quantile(p, [0.05, 0.35, 0.65, 0.95])
        F = self._feat(X)
        self.sc = core.Scaler().fit(F)
        Z = np.hstack([np.ones((len(X), 1)), self.sc(F)])
        T = np.clip(Y / core.MAXES, 0, 1)
        from scipy.optimize import minimize
        pen = np.r_[0, np.ones(Z.shape[1] - 1)] * self.lam / len(X)
        self.B = np.zeros((Z.shape[1], Y.shape[1]))
        for j in range(Y.shape[1]):
            t = T[:, j]

            def f(b):
                pp = np.clip(1 / (1 + np.exp(-Z @ b)), 1e-9, 1 - 1e-9)
                return (-np.mean(t * np.log(pp) + (1 - t) * np.log(1 - pp)) + np.sum(pen * b * b),
                        Z.T @ (pp - t) / len(t) + 2 * pen * b)

            self.B[:, j] = minimize(f, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x
        return self

    def predict(self, X):
        Z = np.hstack([np.ones((len(X), 1)), self.sc(self._feat(X))])
        return 1 / (1 + np.exp(-Z @ self.B)) * core.MAXES


def spread(P, floor):
    p = np.clip(P / MX, 0, 1)
    return MX * (np.sqrt(p * (1 - p)) + floor)


def cvplus(mu_te, s, sig_te, alpha):
    """CV+ interval. mu_te, sig_te: (m, n, J) fold-model predictions/spreads for each training i;
    s: (n, J) out-of-fold scores. Returns lo, hi of shape (m, J)."""
    n = s.shape[0]
    lo_all = mu_te - s[None] * sig_te
    hi_all = mu_te + s[None] * sig_te
    kl = int(np.floor(alpha * (n + 1)))
    kh = int(np.ceil((1 - alpha) * (n + 1)))
    lo = np.sort(lo_all, axis=1)[:, max(kl, 1) - 1]
    hi = np.sort(hi_all, axis=1)[:, min(kh, n) - 1]
    return np.clip(lo, 0, MX), np.clip(hi, 0, MX)


def r2cols(Y, P):
    return 1 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def main(path):
    d, X, Y, _ = core.load(path)
    n = len(X)
    Yall = np.hstack([Y, derived(Y)])
    aq = Yall[:, -1]
    band = pd.cut(aq, [-1, 25, 50, 75, 93.8, 101], labels=["Very severe (0-25)", "Severe (26-50)",
                                                          "Moderate (51-75)", "Mild (76-93.7)",
                                                          "Above cut-off (>=93.8)"]).astype(str)
    atype = d["Aphasia Type from WAB"].to_numpy()

    rows, cov = [], []
    oof = {m: np.zeros((REPEATS, n, len(LABELS))) for m in ("B1", "NL-Ridge", "FL-acc", "FL-spline")}
    for r in range(REPEATS):
        for f, (tr, te) in enumerate(KFold(K, shuffle=True, random_state=42 + r).split(X)):
            seed = 1000 * r + f
            inner = list(KFold(INNER, shuffle=True, random_state=seed).split(tr))
            # ---- point models with nested tuning
            fits = {}
            for name, grid in (("B1", [lambda: core.OLS("pct")]),
                               ("NL-Ridge", [lambda l=l: NLRidge(l) for l in (0.01, 0.1, 1, 10, 100)]),
                               ("FL-acc", [lambda l=l: core.FracLogit("acc", l) for l in (0.1, 1, 10, 100)]),
                               ("FL-spline", [lambda l=l: FLSpline(l) for l in (0.1, 1, 10, 100)])):
                best, bs = None, -np.inf
                for make in grid:
                    o = np.zeros_like(Y[tr])
                    for a, b in inner:
                        o[b] = make().fit(X[tr][a], Y[tr][a]).predict(X[tr][b])
                    s_ = np.mean(r2cols(Y[tr], o))
                    if s_ > bs:
                        best, bs = make, s_
                m = best().fit(X[tr], Y[tr])
                fits[name] = (best, m)
                P = m.predict(X[te])
                Pf = np.hstack([P, derived(P)])
                oof[name][r, te] = Pf
                rr = r2cols(Yall[te], Pf)
                for j, lab in enumerate(LABELS):
                    rows.append(dict(model=name, rep=r, fold=f, outcome=lab, r2=rr[j]))
            # ---- CV+ for FL-acc (fold models = inner-fold models with the tuned penalty)
            make = fits["FL-acc"][0]
            mu_tr = np.zeros((len(tr), len(LABELS)))
            mu_te = np.zeros((len(te), len(tr), len(LABELS)))
            for a, b in inner:
                mk = make().fit(X[tr][a], Y[tr][a])
                pb = mk.predict(X[tr][b])
                mu_tr[b] = np.hstack([pb, derived(pb)])
                pt = mk.predict(X[te])
                mu_te[:, b] = np.hstack([pt, derived(pt)])[:, None, :]
            Ytr = Yall[tr]
            Pfull = oof["FL-acc"][r, te]
            for a_ in (0.1, 0.2):
                variants = {}
                for fl in FLOORS:
                    s = np.abs(Ytr - mu_tr) / spread(mu_tr, fl)
                    sig = spread(mu_te.reshape(-1, len(LABELS)), fl).reshape(mu_te.shape)
                    variants[f"cvplus-norm-{fl}"] = cvplus(mu_te, s, sig, a_)
                    q = core.conformal_q(Ytr, mu_tr, MX, a_) if fl == 0.05 else None
                    if fl == 0.05:
                        variants["cv-approx-norm-0.05"] = core.interval(Pfull, q, MX)
                s = np.abs(Ytr - mu_tr)
                variants["cvplus-standard"] = cvplus(mu_te, s, np.ones_like(mu_te), a_)
                for v, (lo, hi) in variants.items():
                    inside = (Yall[te] >= lo - 1e-9) & (Yall[te] <= hi + 1e-9)
                    for ii, idx in enumerate(te):
                        for j, lab in enumerate(LABELS):
                            cov.append((v, a_, lab, band[idx], atype[idx], inside[ii, j], hi[ii, j] - lo[ii, j]))
        print(f"repeat {r + 1}/{REPEATS}", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "revision2_folds.csv", index=False)
    C = pd.DataFrame(cov, columns=["method", "alpha", "outcome", "band", "type", "covered", "width"])
    C.to_csv(OUT / "revision2_coverage_long.csv.gz", index=False)

    # ---------- point-model summaries
    corr = 1 / (REPEATS * K) + (n / K) / (n * (K - 1) / K)
    get = lambda m, lab: R[(R.model == m) & (R.outcome == lab)].sort_values(["rep", "fold"]).r2.to_numpy()  # noqa
    out = []
    for lab in LABELS:
        for m in ("FL-spline",):
            for base in ("FL-acc", "B1"):
                dlt = get(m, lab) - get(base, lab)
                se = np.sqrt(corr * dlt.var(ddof=1))
                out.append(dict(contrast=f"{m}-{base}", outcome=lab, r2=get(m, lab).mean(), d=dlt.mean(),
                                lo=dlt.mean() - 2.0096 * se, hi=dlt.mean() + 2.0096 * se,
                                p=2 * stats.t.sf(abs(dlt.mean() / se), 49)))
    S = pd.DataFrame(out)
    S.to_csv(OUT / "revision2_spline.csv", index=False)
    print(S.round(3).to_string(index=False))
    print("mean R2 10 subtests:", R[R.outcome.isin(core.KEYS)].groupby("model").r2.mean().round(4).to_dict())

    # monotonicity of full-data fits
    g = np.zeros((176, 7))
    g[:, 0] = np.arange(176)
    g[:, 3] = 175 - g[:, 0]
    for name, m in (("FL-acc", core.FracLogit("acc", 0.1).fit(X, Y)), ("FL-spline", FLSpline(1.0).fit(X, Y)),
                    ("NL-Ridge", NLRidge(1.0).fit(X, Y))):
        P = m.predict(g)
        dec = [(core.KEYS[j], round(float(np.min(np.diff(P[:, j]))), 3)) for j in range(10) if np.min(np.diff(P[:, j])) < -1e-6]
        print(f"{name}: non-monotone subtests (largest single-step drop): {dec}")

    # ---------- calibration
    cal = []
    for m in oof:
        for j, lab in enumerate(LABELS):
            citl = np.mean([Yall[:, j].mean() - oof[m][r, :, j].mean() for r in range(REPEATS)])
            slope = np.mean([np.polyfit(oof[m][r, :, j], Yall[:, j], 1)[0] for r in range(REPEATS)])
            cal.append(dict(model=m, outcome=lab, citl=citl, slope=slope))
    Cal = pd.DataFrame(cal)
    Cal.to_csv(OUT / "revision2_calibration.csv", index=False)
    print(Cal.pivot(index="outcome", columns="model", values="citl").loc[LABELS].round(3).to_string())

    # ---------- coverage summaries
    agg = lambda df: df.groupby(["method", "alpha", "outcome"]).agg(coverage=("covered", "mean"),  # noqa
                                                                     width=("width", "mean")).reset_index()
    A = agg(C)
    A.to_csv(OUT / "revision2_coverage.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nmarginal coverage, alpha=0.1, 10 subtests (min-max) and mean width ratio to floor 0.05:")
    a10 = A[(A.alpha == 0.1) & A.outcome.isin(core.KEYS)]
    print(a10.groupby("method").agg(cmin=("coverage", "min"), cmax=("coverage", "max")).round(3))
    print(a10.pivot(index="outcome", columns="method", values="width").round(2).to_string())
    print(A[(A.alpha == 0.2) & A.outcome.isin(core.KEYS)].groupby("method").coverage.agg(["min", "max"]).round(3))
    B = C[(C.alpha == 0.1) & (C.method == "cvplus-norm-0.05")].groupby(["band", "outcome"]).agg(
        coverage=("covered", "mean"), width=("width", "mean"), n=("covered", "size")).reset_index()
    B["n"] = B.n / REPEATS
    B.to_csv(OUT / "revision2_coverage_by_band.csv", index=False)
    print("\ncoverage by AQ band (CV+ normalised, 90%):")
    print(B.pivot(index="outcome", columns="band", values="coverage").loc[LABELS].round(3).to_string())
    print(B.pivot(index="outcome", columns="band", values="width").loc[LABELS].round(2).to_string())
    print(B.groupby("band").n.first())
    T = C[(C.alpha == 0.1) & (C.method == "cvplus-norm-0.05")].groupby(["type"]).agg(
        coverage=("covered", "mean"), n=("covered", "size")).reset_index()
    T["n"] = T.n / REPEATS / len(LABELS)
    T.to_csv(OUT / "revision2_coverage_by_type.csv", index=False)
    print(T.round(3))
    Ct = C[(C.alpha == 0.1) & (C.method == "cvplus-norm-0.05")].groupby(["type", "outcome"]).covered.mean().unstack()
    print("by type, min over outcomes:", Ct.min(1).round(3).to_dict())

    # ---------- TOST FL-acc vs NL-Ridge
    Rv = pd.read_csv(OUT / "revision_folds.csv")
    tost = []
    for lab in LABELS:
        a = Rv[(Rv.model == "FL-acc") & (Rv.outcome == lab)].sort_values(["rep", "fold"]).r2.to_numpy()
        b = Rv[(Rv.model == "NL-Ridge") & (Rv.outcome == lab)].sort_values(["rep", "fold"]).r2.to_numpy()
        dlt = a - b
        se = np.sqrt(corr * dlt.var(ddof=1))
        t90 = stats.t.ppf(0.95, 49)
        lo, hi = dlt.mean() - t90 * se, dlt.mean() + t90 * se
        p_tost = max(stats.t.sf((dlt.mean() + 0.02) / se, 49), stats.t.sf((0.02 - dlt.mean()) / se, 49))
        tost.append(dict(outcome=lab, d=dlt.mean(), ci90_lo=lo, ci90_hi=hi, p_tost=p_tost,
                         equivalent=(lo > -0.02) and (hi < 0.02)))
    Tt = pd.DataFrame(tost)
    Tt.to_csv(OUT / "revision2_tost.csv", index=False)
    print(Tt.round(3).to_string(index=False))

    # ---------- learning-curve endpoint check
    L = pd.read_csv(OUT / "learning_curve_folds.csv")
    e = L[(L.n == 170) & L.outcome.isin(core.KEYS)]
    print("LC n=170, all repeats:", e.groupby("model").r2.mean().round(4).to_dict())
    print("LC n=170, repeats 0-9:", e[e.rep < 10].groupby("model").r2.mean().round(4).to_dict())
    print("LC n=170, TabPFN repeats 0-3 vs main CV repeats 0-3:",
          e[(e.rep < 4)].groupby("model").r2.mean().round(4).to_dict())
    print("n with 0 correct:", int((X[:, 0] == 0).sum()), "| n with 175 correct:", int((X[:, 0] == 175).sum()),
          "| n <180 days:", int((pd.to_numeric(d["Days Post-Stroke"], errors="coerce") < 180).sum()),
          "| n AQ>=93.8:", int((aq >= 93.8).sum()))


if __name__ == "__main__":
    main(sys.argv[1])
