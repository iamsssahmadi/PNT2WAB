"""Fit the deployed model (FL-acc, chosen by the rule in ANALYSIS_PLAN.md) on all
participants and export it, with conformal quantiles and validation metrics,
to docs/model.json for the web tool.

usage: python scripts/fit_final.py path/to/pNT_TO_WAB.xlsx
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pnt2wab import core  # noqa: E402

LAM = 0.1  # selected in every outer fold of scripts/run_cv.py
REPEATS = 10
DERIVED = [("spontaneous_speech", "Spontaneous Speech", 20),
           ("comprehension", "Auditory Verbal Comprehension", 10),
           ("naming", "Naming & Word Finding", 10),
           ("aq", "Aphasia Quotient (AQ)", 100)]
LABELS = {
    "information_content": "Information Content", "fluency": "Fluency",
    "yes_no": "Yes/No Questions", "auditory_words": "Auditory Word Recognition",
    "sequential_commands": "Sequential Commands", "object_naming": "Object Naming",
    "word_fluency": "Word Fluency", "sentence_completion": "Sentence Completion",
    "responsive_speech": "Responsive Speech", "repetition": "Repetition",
}


def derived(Y):
    c = core.composites(Y)
    rep = Y[:, core.KEYS.index("repetition")]
    aq = 2 * (c[:, 0] + c[:, 1] + rep + c[:, 2])
    return np.column_stack([c, aq])


def main(path):
    d, X, Y, _ = core.load(path)
    dmax = np.array([m for *_, m in DERIVED], float)
    Yd = derived(Y)

    # repeated out-of-fold predictions -> conformal scores and validation metrics
    oof = np.zeros((REPEATS,) + Y.shape)
    for r in range(REPEATS):
        for tr, te in KFold(5, shuffle=True, random_state=42 + r).split(X):
            oof[r, te] = core.FracLogit("acc", LAM).fit(X[tr], Y[tr]).predict(X[te])
    P_all = oof.reshape(-1, Y.shape[1])
    Y_all = np.tile(Y, (REPEATS, 1))
    Pd_all, Yd_all = derived(P_all), np.tile(Yd, (REPEATS, 1))

    q = {}
    for a in (0.1, 0.2):
        q[str(1 - a)] = dict(
            subtests=core.conformal_q(Y_all, P_all, core.MAXES, a).tolist(),
            derived=core.conformal_q(Yd_all, Pd_all, dmax, a).tolist())

    def metrics(Yt, P):
        out = []
        for j in range(Yt.shape[1]):
            r2 = [1 - ((Yt[:len(Y), j] - P[r * len(Y):(r + 1) * len(Y), j]) ** 2).sum()
                  / ((Yt[:len(Y), j] - Yt[:len(Y), j].mean()) ** 2).sum() for r in range(REPEATS)]
            out.append(dict(r2=float(np.mean(r2)), mae=float(np.abs(Yt[:, j] - P[:, j]).mean())))
        return out

    m = core.FracLogit("acc", LAM).fit(X, Y)

    # CV+ (Barber et al., 2021) with normalised scores: 5 fold models fitted on all participants
    mx_all = np.r_[core.MAXES, dmax]
    fold_of = np.zeros(len(X), int)
    folds, mu_oof = [], np.zeros((len(X), len(mx_all)))
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=42).split(X)):
        fm = core.FracLogit("acc", LAM).fit(X[tr], Y[tr])
        folds.append(dict(scaler_mean=fm.sc.m.tolist(), scaler_sd=fm.sc.s.tolist(), coef=fm.B.T.tolist()))
        pk = fm.predict(X[te])
        mu_oof[te] = np.hstack([pk, derived(pk)])
        fold_of[te] = k
    floor = 0.05
    Yfull = np.hstack([Y, Yd])
    scores = np.abs(Yfull - mu_oof) / (mx_all * (np.sqrt(np.clip(mu_oof / mx_all, 0, 1)
                                                           * (1 - np.clip(mu_oof / mx_all, 0, 1))) + floor))

    def cvplus(Xq, alpha):
        mus = []
        for f in folds:
            mm = core.FracLogit("acc", LAM)
            mm.sc = core.Scaler()
            mm.sc.m, mm.sc.s, mm.B = np.array(f["scaler_mean"]), np.array(f["scaler_sd"]), np.array(f["coef"]).T
            pq = mm.predict(Xq)
            mus.append(np.hstack([pq, derived(pq)]))
        mu = np.stack(mus)[fold_of]                       # (n, m, J): fold model of each calibration point
        sig = mx_all * (np.sqrt(np.clip(mu / mx_all, 0, 1) * (1 - np.clip(mu / mx_all, 0, 1))) + floor)
        lo_all, hi_all = mu - scores[:, None, :] * sig, mu + scores[:, None, :] * sig
        n = len(scores)
        kl, kh = int(np.floor(alpha * (n + 1))), int(np.ceil((1 - alpha) * (n + 1)))
        lo = np.sort(lo_all, axis=0)[max(kl, 1) - 1]
        hi = np.sort(hi_all, axis=0)[min(kh, n) - 1]
        return np.clip(lo, 0, mx_all), np.clip(hi, 0, mx_all)

    model = dict(
        name="PNT-to-WAB bounded model", version="2.3.0", trained=str(date.today()),
        n_train=int(len(X)),
        description="Ridge fractional-logit GLM on logit naming accuracy (linear, quadratic, cubic). "
                    "Composites and AQ are derived from predicted subtests with WAB-R formulas. "
                    "Intervals: normalised conformal quantile from 10x5-fold out-of-fold scores "
                    "(aggregate quantiles only; no participant-level values are stored).",
        features=dict(kind="acc", scaler_mean=m.sc.m.tolist(), scaler_sd=m.sc.s.tolist()),
        coef=m.B.T.tolist(),  # one row per subtest: [intercept, a, a^2, a^3]
        subtests=[dict(key=k, label=LABELS[k], max=mx) for k, _, mx in core.SUBTESTS],
        derived=[dict(key=k, label=l, max=mx) for k, l, mx in DERIVED],
        conformal=dict(spread_floor=floor, q=q, method="cv-normalised-quantile"),
        validation=dict(subtests=metrics(Y_all, P_all), derived=metrics(Yd_all, Pd_all)),
    )
    # reference predictions for the JavaScript parity test
    rng = np.random.default_rng(0)
    cases = [0, 1, 50, 87, 140, 174, 175] + rng.integers(0, 176, 13).tolist()
    Xc = np.zeros((len(cases), 7))
    Xc[:, 0] = cases
    Xc[:, 3] = 175 - Xc[:, 0]
    Pc = m.predict(Xc)
    Pdc = derived(Pc)
    lo, hi = core.interval(Pc, np.array(q["0.9"]["subtests"]), core.MAXES)
    dlo, dhi = core.interval(Pdc, np.array(q["0.9"]["derived"]), dmax)
    # participant-level CV+ material: kept locally for the manuscript figures, never published
    json.dump(dict(spread_floor=floor, fold_of=fold_of.tolist(), folds=folds, scores=scores.tolist()),
              open(ROOT / "results" / "cvplus_private.json", "w"))
    (ROOT / "docs").mkdir(exist_ok=True)
    json.dump(model, open(ROOT / "docs" / "model.json", "w"))
    (ROOT / "tests").mkdir(exist_ok=True)
    json.dump([dict(correct=int(c), total=175, pred=Pc[i].tolist(), lo=lo[i].tolist(),
                    hi=hi[i].tolist(), dpred=Pdc[i].tolist(), dlo=dlo[i].tolist(),
                    dhi=dhi[i].tolist()) for i, c in enumerate(cases)],
              open(ROOT / "tests" / "parity_cases.json", "w"))

    v = model["validation"]
    print(pd.DataFrame(v["subtests"] + v["derived"],
                       index=core.KEYS + [k for k, *_ in DERIVED]).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
