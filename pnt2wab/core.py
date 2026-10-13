"""Data, features, models and conformal intervals for PNT -> WAB-R prediction.

Everything a deployed model needs is kept simple enough to be reproduced
exactly in the browser (see docs/app.js): fixed feature formulas, linear or
small-MLP predictors with sigmoid outputs, and per-outcome conformal quantiles.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

PNT = ["Correct", "Formal", "Mixed", "NR", "Neologism", "Semantic", "Unrelated"]
PNT_SRC = ["pntCorrect", "pntFormal", "pntMixed", "pntNR", "pntNeologism",
           "pntSemantic", "pntUnrelated"]

# 10 independent subtests: (key, source column, maximum)
SUBTESTS = [
    ("information_content", "WABInformation Content", 10),
    ("fluency", "WABFluency Rating", 10),
    ("yes_no", "WABComprehension Yes/No", 60),
    ("auditory_words", "WABComprehension Auditory Words", 60),
    ("sequential_commands", "WABComprehension Sequential Commands", 80),
    ("object_naming", "WABObject Naming", 60),
    ("word_fluency", "WABWord Fluency", 20),
    ("sentence_completion", "WABSentence Completion", 10),
    ("responsive_speech", "WABResponsive Speech", 10),
    ("repetition", "WABRepetition Subscore", 10),
]
KEYS = [k for k, _, _ in SUBTESTS]
MAXES = np.array([m for _, _, m in SUBTESTS], float)
COMPOSITES = [
    ("spontaneous_speech", "WABSpontaneous Speech Total", 20),
    ("comprehension", "WABComprehension Subscore", 10),
    ("naming", "WABNaming subscore", 10),
]


def composites(Y):
    """WAB-R composite formulas applied to an (n, 10) subtest matrix."""
    i = {k: j for j, k in enumerate(KEYS)}
    return np.column_stack([
        Y[:, i["information_content"]] + Y[:, i["fluency"]],
        (Y[:, i["yes_no"]] + Y[:, i["auditory_words"]] + Y[:, i["sequential_commands"]]) / 20,
        (Y[:, i["object_naming"]] + Y[:, i["word_fluency"]]
         + Y[:, i["sentence_completion"]] + Y[:, i["responsive_speech"]]) / 10,
    ])


def load(path):
    d = pd.read_excel(path)
    X = d[PNT_SRC].to_numpy(float)
    Y = d[[c for _, c, _ in SUBTESTS]].to_numpy(float)
    C = d[[c for _, c, _ in COMPOSITES]].to_numpy(float)
    return d, X, Y, C


# ---------------------------------------------------------------- features
def feat_acc(counts):
    n = counts.sum(1)
    a = np.log((counts[:, 0] + .5) / (n - counts[:, 0] + .5))
    return np.column_stack([a, a ** 2, a ** 3])


def feat_comp(counts):
    e = np.log(counts[:, 1:] + .5)
    return e - e.mean(1, keepdims=True)  # centred log-ratio of the 6 error types


def features(counts, kind):
    if kind == "acc":
        return feat_acc(counts)
    if kind == "acc+comp":
        return np.hstack([feat_acc(counts), feat_comp(counts)])
    if kind == "pct":
        return (counts[:, :1] / counts.sum(1, keepdims=True))
    if kind == "pct+prop":  # published B3: % correct + 5 error proportions (NR omitted)
        p = counts / counts.sum(1, keepdims=True)
        err = counts[:, 1:]
        tot = err.sum(1, keepdims=True)
        comp = np.divide(err, tot, out=np.zeros_like(err), where=tot > 0)
        return np.hstack([p[:, :1], np.delete(comp, 2, axis=1)])  # drop NR
    raise ValueError(kind)


class Scaler:
    def fit(self, X):
        self.m, self.s = X.mean(0), X.std(0) + 1e-9
        return self

    def __call__(self, X):
        return (X - self.m) / self.s


# ---------------------------------------------------------------- models
def _sig(z):
    return 1 / (1 + np.exp(-z))


class Mean:
    name = "B0"

    def fit(self, X, Y):
        self.mu = Y.mean(0)
        return self

    def predict(self, X):
        return np.tile(self.mu, (len(X), 1))


class OLS:
    def __init__(self, kind):
        self.kind = kind

    def fit(self, X, Y):
        F = np.hstack([np.ones((len(X), 1)), features(X, self.kind)])
        self.B = np.linalg.lstsq(F, Y, rcond=None)[0]
        return self

    def predict(self, X):
        F = np.hstack([np.ones((len(X), 1)), features(X, self.kind)])
        return F @ self.B


class FracLogit:
    """Ridge-penalised fractional-logit GLM, one per subtest.

    E[y / max] = sigmoid(b0 + F b). Quasi-binomial loss handles scores at 0 and at
    ceiling without transformation, so predictions can never leave [0, max].
    """

    def __init__(self, kind, lam=1.0):
        self.kind, self.lam = kind, lam

    def fit(self, X, Y):
        self.sc = Scaler().fit(features(X, self.kind))
        F = np.hstack([np.ones((len(X), 1)), self.sc(features(X, self.kind))])
        T = np.clip(Y / MAXES, 0, 1)
        self.B = np.zeros((F.shape[1], Y.shape[1]))
        pen = np.r_[0, np.ones(F.shape[1] - 1)] * self.lam / len(X)
        for j in range(Y.shape[1]):
            t = T[:, j]

            def f(b):
                p = np.clip(_sig(F @ b), 1e-9, 1 - 1e-9)
                loss = -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)) + np.sum(pen * b * b)
                g = F.T @ (p - t) / len(t) + 2 * pen * b
                return loss, g

            self.B[:, j] = minimize(f, np.zeros(F.shape[1]), jac=True, method="L-BFGS-B").x
        return self

    def predict(self, X):
        F = np.hstack([np.ones((len(X), 1)), self.sc(features(X, self.kind))])
        return _sig(F @ self.B) * MAXES


class MTL:
    """Multi-task MLP: one shared tanh hidden layer, 10 sigmoid outputs.

    A seed ensemble is trained as one batched tensor program; predictions are
    averaged on the score scale.
    """

    def __init__(self, kind="acc+comp", hidden=8, wd=1e-2, seeds=8, epochs=1500, lr=0.02):
        self.kind, self.hidden, self.wd = kind, hidden, wd
        self.seeds, self.epochs, self.lr = seeds, epochs, lr

    def fit(self, X, Y):
        import torch
        torch.manual_seed(0)
        self.sc = Scaler().fit(features(X, self.kind))
        F = torch.tensor(self.sc(features(X, self.kind)), dtype=torch.float32)
        T = torch.tensor(np.clip(Y / MAXES, 0, 1), dtype=torch.float32)
        S, d, h, k = self.seeds, F.shape[1], self.hidden, T.shape[1]
        W1 = (torch.randn(S, d, h) / d ** .5).requires_grad_()
        b1 = torch.zeros(S, 1, h, requires_grad=True)
        W2 = (torch.randn(S, h, k) / h ** .5).requires_grad_()
        b2 = torch.zeros(S, 1, k, requires_grad=True)
        opt = torch.optim.Adam([W1, b1, W2, b2], lr=self.lr)
        Fb = F.expand(S, -1, -1)
        for _ in range(self.epochs):
            opt.zero_grad()
            z = torch.tanh(Fb @ W1 + b1) @ W2 + b2
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, T.expand(S, -1, -1))
            loss = loss + self.wd * (W1.pow(2).mean() + W2.pow(2).mean())
            loss.backward()
            opt.step()
        self.W1, self.b1 = W1.detach().numpy(), b1.detach().numpy()[:, 0]
        self.W2, self.b2 = W2.detach().numpy(), b2.detach().numpy()[:, 0]
        return self

    def predict(self, X):
        F = self.sc(features(X, self.kind))
        out = [_sig(np.tanh(F @ self.W1[s] + self.b1[s]) @ self.W2[s] + self.b2[s])
               for s in range(len(self.W1))]
        return np.mean(out, 0) * MAXES


class CB:
    """CatBoost MultiRMSE on scaled targets (reference model, not deployable)."""

    def __init__(self, kind="acc+comp", depth=4, lr=0.03, iters=600):
        self.kind, self.depth, self.lr, self.iters = kind, depth, lr, iters

    def fit(self, X, Y):
        from catboost import CatBoostRegressor
        self.m = CatBoostRegressor(loss_function="MultiRMSE", depth=self.depth,
                                   learning_rate=self.lr, iterations=self.iters,
                                   verbose=0, random_seed=0, thread_count=1)
        self.m.fit(features(X, self.kind), Y / MAXES)
        return self

    def predict(self, X):
        return np.clip(self.m.predict(features(X, self.kind)), 0, 1) * MAXES


# ---------------------------------------------------------------- conformal
def spread(P, maxes):
    """Binomial-type spread used to normalise residuals: wider mid-scale, narrow at ceiling/floor."""
    p = np.clip(P / maxes, 0, 1)
    return maxes * (np.sqrt(p * (1 - p)) + 0.05)


def conformal_q(Yt, Pt, maxes, alpha):
    """Normalised split-conformal quantile per column from out-of-fold predictions."""
    s = np.abs(Yt - Pt) / spread(Pt, maxes)
    n = len(s)
    lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(s, lvl, axis=0, method="higher")


def interval(P, q, maxes):
    w = q * spread(P, maxes)
    return np.clip(P - w, 0, maxes), np.clip(P + w, 0, maxes)
