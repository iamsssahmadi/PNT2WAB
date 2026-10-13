"""Publication figures for the manuscript (600 dpi PNG + TIFF, vector PDF).

usage: python scripts/make_figures.py ../pNT_TO_WAB.xlsx
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from pnt2wab import core  # noqa: E402
from fit_final import LAM, DERIVED, derived  # noqa: E402
from run_revision import NLRidge  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"

# ---- identity colours: the same model keeps the same colour in every figure
INK, INK2, GRID, SURF = "#1d1d1b", "#55554f", "#e6e5e1", "#ffffff"
C_LIN, C_FL, C_TAB, C_NL = "#8f8e88", "#2a78d6", "#eb6834", "#6250d6"
C_PTS = "#b9b8b2"
NAME = {
    "information_content": "Information Content", "fluency": "Fluency",
    "yes_no": "Yes/No Questions", "auditory_words": "Auditory Word Recognition",
    "sequential_commands": "Sequential Commands", "object_naming": "Object Naming",
    "word_fluency": "Word Fluency", "sentence_completion": "Sentence Completion",
    "responsive_speech": "Responsive Speech", "repetition": "Repetition",
    "spontaneous_speech": "Spontaneous Speech†", "comprehension": "Comprehension†",
    "naming": "Naming & Word Finding†", "aq": "Aphasia Quotient†",
}
DOMAINS = [
    ("Naming & word finding", ["naming", "object_naming", "responsive_speech",
                               "word_fluency", "sentence_completion"]),
    ("Spontaneous speech", ["spontaneous_speech", "information_content", "fluency"]),
    ("Repetition", ["repetition"]),
    ("Auditory comprehension", ["comprehension", "auditory_words", "sequential_commands", "yes_no"]),
]

plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "pdf.fonttype": 42, "legend.frameon": False, "legend.fontsize": 7.5,
})


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=600)
    Image.open(FIG / f"{name}.png").convert("RGB").save(FIG / f"{name}.tiff", compression="tiff_lzw",
                                                        dpi=(600, 600))
    plt.close(fig)


def panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=10, fontweight="bold",
            color=INK, ha="right", va="bottom")


# =============================================================== Figure 1: design
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")
    W, H = 22, 22
    xs = [1, 26.3, 51.6, 76.9]

    def box(x, y, title, lines, edge=INK2, fill="#f6f6f4", tc=INK, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0,rounding_size=1.4",
                                    fc=fill, ec=edge, lw=0.8, ls=ls))
        ax.text(x + 1.4, y + H - 2.0, title, fontsize=7.3, fontweight="bold", color=tc, va="top")
        step = 3.05 if len(lines) <= 5 else 2.72
        for i, ln in enumerate(lines):
            ax.text(x + 1.4, y + H - 6.4 - i * step, ln, fontsize=6.2, color=INK2, va="top")

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, shrinkA=0, shrinkB=0,
                                    mutation_scale=8))

    top, bot = 31, 3
    box(xs[0], top, "Data", ["n = 213 adults with chronic", "post-stroke aphasia", "PNT: 7 response counts",
                              "WAB-R: 10 subtests,", "3 composites and the AQ"])
    box(xs[1], top, "PNT-only predictors", ["Logit accuracy (a, a\u00b2, a\u00b3)", "Error mix (centred",
                                            "log-ratio of 6 errors)", "Raw proportions",
                                            "No other data used"])
    box(xs[2], top, "Candidate models", ["Linear (published baseline)", "Nonlinear ridge (identity)*", "Bounded fractional-logit",
                                         "Multi-task neural network", "CatBoost (multi-output)",
                                         "TabPFN v2/v3.5, stacking*"])
    box(xs[3], top, "Nested validation", ["10 \u00d7 repeated 5-fold CV", "Inner 5-fold for tuning,",
                                          "weights and calibration", "Corrected resampled",
                                          "t-tests vs baseline"])
    box(xs[0], bot, "Learning curves*", ["Training n = 30\u2013170", "Power-law fit,",
                                         "extrapolated to n = 500"], ls=(0, (3, 2)))
    box(xs[1], bot, "Outputs", ["10 subtests predicted", "Composites and AQ via", "WAB-R formulas",
                                "Conformal 80%/90%", "prediction intervals"])
    box(xs[2], bot, "Selection rule", ["Fixed before any results", "Browser-deployable only", "Simplest model within",
                                                 "0.005 of best mean R\u00b2"])
    box(xs[3], bot, "Deployed model", ["Bounded GLM on accuracy", "Refit on all 213", "Frozen; runs in the browser",
                                       "No data leave the device"], edge=C_FL, fill="#eaf2fc", tc=C_FL)
    for i in range(3):
        arrow(xs[i] + W, top + H / 2, xs[i + 1], top + H / 2)
    ym = (top + bot + H) / 2
    cx3, cx1 = xs[3] + W / 2, xs[1] + W / 2
    ax.plot([cx3, cx3, cx1], [top, ym, ym], color=INK2, lw=0.9)
    arrow(cx1, ym, cx1, bot + H)
    arrow(xs[1] + W, bot + H / 2, xs[2], bot + H / 2)
    arrow(xs[2] + W, bot + H / 2, xs[3], bot + H / 2)
    ax.text(xs[0], 0.2, "* exploratory analyses added after the pre-specified plan", fontsize=6.4,
            color=INK2, style="italic", va="bottom")
    save(fig, "Figure1_design")


# =============================================================== Figure 2: mapping
def fig2(path):
    d, X, Y, _ = core.load(path)
    q = json.load(open(ROOT / "docs" / "model.json"))["conformal"]["q"]["0.9"]
    fl = core.FracLogit("acc", LAM).fit(X, Y)
    lin = core.OLS("pct").fit(X, Y)
    nl = NLRidge(1.0).fit(X, Y)
    g = np.zeros((176, 7))
    g[:, 0] = np.arange(176)
    g[:, 3] = 175 - g[:, 0]
    acc = g[:, 0] / 175 * 100
    Pf, Pl, Pn = fl.predict(g), lin.predict(g), nl.predict(g)
    Pnd = derived(Pn)
    dmax = np.array([m for *_, m in DERIVED], float)
    Pd = derived(Pf)
    # CV+ band from the deployed model's stored fold models and calibration scores
    cf = json.load(open(RES / "cvplus_private.json"))
    mx_all = np.r_[core.MAXES, dmax]
    mus = []
    for f in cf["folds"]:
        mm = core.FracLogit("acc", LAM)
        mm.sc = core.Scaler()
        mm.sc.m, mm.sc.s, mm.B = np.array(f["scaler_mean"]), np.array(f["scaler_sd"]), np.array(f["coef"]).T
        pq = mm.predict(g)
        mus.append(np.hstack([pq, derived(pq)]))
    mu = np.stack(mus)[np.array(cf["fold_of"])]
    sc = np.array(cf["scores"])
    pp = np.clip(mu / mx_all, 0, 1)
    sg = mx_all * (np.sqrt(pp * (1 - pp)) + cf["spread_floor"])
    n = len(sc)
    lo14 = np.sort(mu - sc[:, None, :] * sg, axis=0)[int(np.floor(0.1 * (n + 1))) - 1]
    hi14 = np.sort(mu + sc[:, None, :] * sg, axis=0)[int(np.ceil(0.9 * (n + 1))) - 1]
    lo14, hi14 = np.clip(lo14, 0, mx_all), np.clip(hi14, 0, mx_all)
    lo, hi, dlo, dhi = lo14[:, :10], hi14[:, :10], lo14[:, 10:], hi14[:, 10:]
    Pld = derived(Pl)
    obs_acc = X[:, 0] / X.sum(1) * 100
    Yd = derived(Y)

    order = ["object_naming", "responsive_speech", "sentence_completion", "word_fluency",
             "information_content", "fluency", "repetition", "auditory_words",
             "sequential_commands", "yes_no", "aq"]
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 6.1), sharex=True)
    for ax, key in zip(axes.flat, order):
        if key == "aq":
            y, p, l, a, b, mx = Yd[:, 3], Pd[:, 3], Pld[:, 3], dlo[:, 3], dhi[:, 3], 100
            nlc = Pnd[:, 3]
        else:
            j = core.KEYS.index(key)
            y, p, l, a, b, mx = Y[:, j], Pf[:, j], Pl[:, j], lo[:, j], hi[:, j], core.MAXES[j]
            nlc = Pn[:, j]
        ax.fill_between(acc, a, b, color=C_FL, alpha=0.13, lw=0)
        ax.scatter(obs_acc, y, s=5, color=C_PTS, lw=0, zorder=2)
        ax.plot(acc, l, color=C_LIN, lw=1.1, ls=(0, (4, 2)), zorder=3)
        ax.plot(acc, nlc, color=C_NL, lw=1.0, ls=(0, (1, 1.5)), zorder=3)
        ax.plot(acc, p, color=C_FL, lw=1.6, zorder=4)
        ax.axhline(mx, color=GRID, lw=0.8, zorder=1)
        ax.set_ylim(-mx * 0.06, mx * 1.12)
        ax.set_yticks(np.linspace(0, mx, 3))
        ax.set_title(NAME[key], loc="left", color=INK, pad=3)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
    axes[1, 3].tick_params(labelbottom=True)
    leg = axes.flat[-1]
    leg.axis("off")
    leg.legend(handles=[
        Line2D([], [], color=C_FL, lw=1.6, label="Bounded model (deployed)"),
        Line2D([], [], color=C_FL, lw=6, alpha=0.18, label="90% prediction interval"),
        Line2D([], [], color=C_NL, lw=1.0, ls=(0, (1, 1.5)), label="Nonlinear ridge (identity link)"),
        Line2D([], [], color=C_LIN, lw=1.1, ls=(0, (4, 2)), label="Linear baseline"),
        Line2D([], [], marker="o", ls="", color=C_PTS, ms=3.5, label="Participants (n = 213)"),
        Line2D([], [], color=GRID, lw=0.8, label="Subtest maximum"),
    ], loc="center left", handlelength=2.2)
    fig.supxlabel("PNT accuracy (% correct)", fontsize=8, color=INK)
    fig.supylabel("WAB-R score", fontsize=8, color=INK)
    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    save(fig, "Figure2_mapping")


# =============================================================== Figure 3: per-outcome R2
def fig3():
    rev = pd.read_csv(RES / "revision_summary.csv")
    tab = pd.read_csv(RES / "tabpfn_v3.5_summary.csv")
    tab = tab[tab.model == "TabPFNv3.5-props"].set_index("outcome")
    b1 = rev[rev.model == "B1"].set_index("outcome")
    nl = rev[rev.model == "NL-Ridge"].set_index("outcome")
    flr = rev[rev.model == "FL-acc"].set_index("outcome")
    rows, ypos, y = [], [], 0
    seps = []
    for dom, keys in DOMAINS:
        seps.append((dom, y))
        for k in keys:
            rows.append(k)
            ypos.append(y)
            y += 1
        y += 0.9
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for k, yy in zip(rows, ypos):
        vals = [b1.loc[k, "r2_fold"], nl.loc[k, "r2_fold"], flr.loc[k, "r2_fold"], tab.loc[k, "r2"]]
        ax.plot([min(vals), max(vals)], [yy, yy], color=GRID, lw=2.2, zorder=1, solid_capstyle="round")
        ax.scatter(b1.loc[k, "r2_fold"], yy, s=30, color=C_LIN, zorder=3, lw=0)
        for col, r2, pv, mk, sz in ((C_NL, nl.loc[k, "r2_fold"], nl.loc[k, "p_vs_B1"], "^", 30),
                                    (C_FL, flr.loc[k, "r2_fold"], flr.loc[k, "p_vs_B1"], "o", 30),
                                    (C_TAB, tab.loc[k, "r2"], tab.loc[k, "p_vs_B1"], "D", 24)):
            ax.scatter(r2, yy, s=sz, marker=mk, zorder=4, facecolor=col if pv < 0.05 else SURF,
                       edgecolor=col, lw=1.1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([NAME[k] for k in rows])
    ax.invert_yaxis()
    for dom, yy in seps:
        ax.text(-0.012, yy - 0.78, dom.upper(), transform=ax.get_yaxis_transform(), fontsize=6.6,
                color=INK, fontweight="bold", ha="right")
    ax.set_xlim(0.2, 0.96)
    ax.set_xticks(np.arange(0.2, 1.0, 0.1))
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Cross-validated R\u00b2 (mean of 50 held-out folds)")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=C_LIN, ms=5, label="Linear baseline (published)"),
        Line2D([], [], marker="^", ls="", color=C_NL, ms=5, label="Nonlinear ridge, identity link*"),
        Line2D([], [], marker="o", ls="", color=C_FL, ms=5, label="Bounded model (deployed)"),
        Line2D([], [], marker="D", ls="", color=C_TAB, ms=4.5, label="TabPFN v3.5*"),
        Line2D([], [], marker="o", ls="", mfc=SURF, mec=INK2, ms=5, label="open = not significant vs baseline"),
    ], loc="lower right", borderaxespad=0.2)
    fig.subplots_adjust(left=0.27, right=0.98, top=0.97, bottom=0.10)
    save(fig, "Figure3_performance")


# =============================================================== Figure 4: intervals
def fig4(path):
    c = pd.read_csv(RES / "revision_intervals.csv")
    c = c[c.alpha == 0.1].set_index(["model", "method", "outcome"])
    get = lambda m, meth, k, col: c.loc[(m, meth, k), col]  # noqa: E731
    keys = sorted(core.KEYS, key=lambda k: get("FL-acc", "cv-normalised", k, "mean_width")
                  / get("B1", "cv-standard", k, "mean_width"))
    yy = np.arange(len(keys))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True,
                                 gridspec_kw=dict(width_ratios=[1, 1.2], wspace=0.08))
    a1.axvline(90, color=INK2, lw=0.8, ls=(0, (3, 2)))
    a1.text(90.12, -0.95, "nominal 90%", fontsize=6.8, color=INK2)
    cvp = pd.read_csv(RES / "revision2_coverage.csv")
    cvp = cvp[(cvp.alpha == 0.1) & (cvp.method == "cvplus-norm-0.05")].set_index("outcome")
    a1.scatter([get("B1", "cv-standard", k, "coverage") * 100 for k in keys], yy, s=26, color=C_LIN, zorder=4)
    a1.scatter([cvp.loc[k, "coverage"] * 100 for k in keys], yy, s=30, marker="s", color=C_FL, zorder=5)
    a1.scatter([get("FL-acc", "split-normalised", k, "coverage") * 100 for k in keys], yy, s=26, zorder=4,
               facecolor=SURF, edgecolor=C_FL, lw=1.1)
    a1.set_xlim(88.5, 95)
    a1.text(88.6, len(keys) - 0.35, "", fontsize=6)
    a1.set_xlabel("Coverage of 90% intervals (%)")
    a1.set_yticks(yy)
    a1.set_yticklabels([NAME[k] for k in keys])
    a1.invert_yaxis()
    a1.grid(axis="x", color=GRID, lw=0.6)
    a1.set_axisbelow(True)
    a1.tick_params(axis="y", length=0)
    a1.spines["left"].set_visible(False)
    panel_label(a1, "A")

    a2.axvline(0, color=INK2, lw=0.8)
    for m, meth, col, mk, filled in (("NL-Ridge", "cv-normalised", C_NL, "^", True),
                                     ("FL-acc", "cv-normalised", C_FL, "o", True),
                                     ("FL-acc", "split-normalised", C_FL, "o", False)):
        rel = [(get(m, meth, k, "mean_width") / get("B1", "cv-standard", k, "mean_width") - 1) * 100 for k in keys]
        a2.scatter(rel, yy, s=26, marker=mk, zorder=4, facecolor=col if filled else SURF, edgecolor=col, lw=1.1)
    for y_, k in zip(yy, keys):
        a2.text(1.02, y_, f"{get('B1', 'cv-standard', k, 'mean_width'):.1f}", transform=a2.get_yaxis_transform(),
                va="center", fontsize=6.6, color=INK2)
    a2.text(1.02, -0.75, "linear width", transform=a2.get_yaxis_transform(), fontsize=6.4, color=INK2,
            va="bottom")
    a2.set_xlim(-30, 15)
    a2.set_xlabel("Mean interval width vs linear baseline (%)")
    a2.grid(axis="x", color=GRID, lw=0.6)
    a2.set_axisbelow(True)
    a2.tick_params(axis="y", length=0)
    a2.spines["left"].set_visible(False)
    panel_label(a2, "B")
    handles = [Line2D([], [], marker="s", ls="", color=C_FL, ms=5, label="Bounded, normalized CV+ (primary; A only)"),
               Line2D([], [], marker="o", ls="", color=C_LIN, ms=5, label="Linear, standard scores"),
               Line2D([], [], marker="^", ls="", color=C_NL, ms=5, label="Nonlinear ridge, normalized*"),
               Line2D([], [], marker="o", ls="", color=C_FL, ms=5, label="Bounded, normalized (cross-validated calibration)"),
               Line2D([], [], marker="o", ls="", mfc=SURF, mec=C_FL, ms=5, label="Bounded, normalized (split conformal)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.55, 0.0), fontsize=7)
    fig.subplots_adjust(left=0.22, right=0.92, top=0.90, bottom=0.30)
    save(fig, "Figure4_intervals")


# =============================================================== Figure 5: validation
def fig5(train_path):
    """Out-of-fold predictions (first CV repeat) with nested conformal 90% intervals."""
    d, X, Y, _ = core.load(train_path)
    dmax = np.array([m for *_, m in DERIVED], float)
    P, lo, hi = np.zeros_like(Y), np.zeros_like(Y), np.zeros_like(Y)
    Pd, dlo, dhi = (np.zeros((len(Y), 4)) for _ in range(3))
    from run_revision2 import cvplus as cvp, spread as spr
    mx_all = np.r_[core.MAXES, dmax]
    for tr, te in KFold(5, shuffle=True, random_state=42).split(X):
        inner = list(KFold(5, shuffle=True, random_state=0).split(tr))
        mu_tr = np.zeros((len(tr), 14))
        mu_te = np.zeros((len(te), len(tr), 14))
        for a, b in inner:
            mk = core.FracLogit("acc", LAM).fit(X[tr][a], Y[tr][a])
            pb, pt = mk.predict(X[tr][b]), mk.predict(X[te])
            mu_tr[b] = np.hstack([pb, derived(pb)])
            mu_te[:, b] = np.hstack([pt, derived(pt)])[:, None, :]
        Ytr = np.hstack([Y[tr], derived(Y[tr])])
        sc = np.abs(Ytr - mu_tr) / spr(mu_tr, 0.05)
        sg = spr(mu_te.reshape(-1, 14), 0.05).reshape(mu_te.shape)
        l14, h14 = cvp(mu_te, sc, sg, 0.1)
        P[te] = core.FracLogit("acc", LAM).fit(X[tr], Y[tr]).predict(X[te])
        Pd[te] = derived(P[te])
        lo[te], hi[te], dlo[te], dhi[te] = l14[:, :10], h14[:, :10], l14[:, 10:], h14[:, 10:]
    Yd = derived(Y)
    panels = [("aq", Yd[:, 3], Pd[:, 3], dlo[:, 3], dhi[:, 3], 100)]
    for k in ("object_naming", "information_content", "auditory_words"):
        j = core.KEYS.index(k)
        panels.append((k, Y[:, j], P[:, j], lo[:, j], hi[:, j], core.MAXES[j]))
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.35))
    for ax, (k, y, p, a, b, mx), lab in zip(axes, panels, "ABCD"):
        inside = (y >= a - 1e-9) & (y <= b + 1e-9)
        ax.plot([0, mx], [0, mx], color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.errorbar(p, y, xerr=[p - a, b - p], fmt="none", ecolor=C_FL, alpha=0.12, lw=0.7, zorder=2)
        ax.scatter(p[inside], y[inside], s=7, color=C_FL, lw=0, alpha=0.85, zorder=3)
        ax.scatter(p[~inside], y[~inside], s=10, facecolor=SURF, edgecolor="#c0392b", lw=0.8, zorder=4)
        r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        ax.text(0.04, 0.97, f"R\u00b2 = {r2:.2f}\n{inside.mean() * 100:.0f}% within 90% PI",
                transform=ax.transAxes, va="top", fontsize=6.8, color=INK, zorder=6,
                bbox=dict(boxstyle="square,pad=0.25", fc=SURF, ec="none", alpha=0.9))
        ax.set_xlim(-mx * 0.04, mx * 1.04)
        ax.set_ylim(-mx * 0.04, mx * 1.04)
        ax.set_xticks(np.linspace(0, mx, 3))
        ax.set_yticks(np.linspace(0, mx, 3))
        ax.set_aspect("equal")
        ax.set_title(NAME[k], loc="left", color=INK, pad=3)
        panel_label(ax, lab)
    axes[0].set_ylabel("Observed WAB-R score")
    fig.supxlabel("Cross-validated predicted score (n = 213; bars = 90% prediction interval; "
                  "open red = outside interval)", fontsize=7.5, color=INK, y=0.02)
    fig.tight_layout(w_pad=1.0)
    save(fig, "Figure6_crossvalidated")



# =============================================================== Figure 5: conditional coverage
def fig_conditional():
    B = pd.read_csv(RES / "revision2_coverage_by_band.csv")
    bands = ["Very severe (0-25)", "Severe (26-50)", "Moderate (51-75)", "Mild (76-93.7)", "Above cut-off (>=93.8)"]
    blabel = ["Very severe\n0\u201325", "Severe\n26\u201350", "Moderate\n51\u201375", "Mild\n76\u201393.7",
              "Above cut-off\n\u226593.8"]
    order = [k for _, ks in DOMAINS for k in ks] + ["aq"]
    cov = B.pivot(index="outcome", columns="band", values="coverage").loc[order, bands] * 100
    wid = B.pivot(index="outcome", columns="band", values="width").loc[order, bands]
    nb = B.groupby("band").n.first().loc[bands]
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("cov", ["#c0392b", "#f3d6cf", "#f2f1ee", "#d5e4f6", "#2a78d6"])
    norm = TwoSlopeNorm(vmin=55, vcenter=90, vmax=100)
    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    im = ax.imshow(cov.to_numpy(), cmap=cmap, norm=norm, aspect="auto")
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            v = cov.iat[i, j]
            ax.text(j, i - 0.12, f"{v:.0f}%", ha="center", va="center", fontsize=7.2,
                    color=SURF if v < 70 else INK, fontweight="bold" if v < 85 else "normal")
            ax.text(j, i + 0.27, f"\u00b1{wid.iat[i, j] / 2:.1f}", ha="center", va="center", fontsize=5.8,
                    color=SURF if v < 70 else INK2)
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([f"{b}\n(n = {int(n)})" for b, n in zip(blabel, nb)], fontsize=7)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([NAME[k] for k in order], fontsize=7.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(bands)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order)), minor=True)
    ax.grid(which="minor", color=SURF, lw=1.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[60, 70, 80, 90, 100])
    cb.set_label("Coverage of 90% intervals (%)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    ax.set_xlabel("Observed WAB-R AQ severity band", fontsize=8, labelpad=6)
    ax.xaxis.set_label_position("top")
    fig.subplots_adjust(left=0.26, right=0.93, top=0.86, bottom=0.03)
    save(fig, "Figure5_conditional_coverage")


# =============================================================== Appendix Figure A1: spline
def fig_spline(path):
    from run_revision2 import FLSpline
    d, X, Y, _ = core.load(path)
    g = np.zeros((176, 7))
    g[:, 0] = np.arange(176)
    g[:, 3] = 175 - g[:, 0]
    acc = g[:, 0] / 175 * 100
    Pa = core.FracLogit("acc", LAM).fit(X, Y).predict(g)
    Ps = FLSpline(1.0).fit(X, Y).predict(g)
    obs = X[:, 0] / X.sum(1) * 100
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 4.0), sharex=True)
    for ax, j in zip(axes.flat, range(10)):
        mx = core.MAXES[j]
        ax.scatter(obs, Y[:, j], s=3, color=C_PTS, lw=0)
        ax.plot(acc, Pa[:, j], color=C_FL, lw=1.4, label="Cubic in logit accuracy (deployed)")
        ax.plot(acc, Ps[:, j], color=INK, lw=1.0, ls=(0, (4, 1.5)), label="Restricted cubic spline")
        ax.set_title(NAME[core.KEYS[j]].replace("Auditory Word ", "Auditory Word\n").replace("Sequential ", "Sequential\n")
                     .replace("Sentence ", "Sentence\n").replace("Responsive ", "Responsive\n")
                     .replace("Information ", "Information\n").replace("Yes/No ", "Yes/No\n"),
                     loc="left", fontsize=6.8, pad=2)
        ax.set_ylim(-mx * 0.06, mx * 1.08)
        ax.set_yticks([0, mx])
        ax.set_xticks([0, 50, 100])
        ax.tick_params(labelsize=6.5)
    h, lb = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lb, loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.0))
    fig.supxlabel("PNT accuracy (% correct)", fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=0.8, w_pad=0.4)
    save(fig, "FigureA1_spline")


# =============================================================== Figure 7: learning curve
def fig6():
    cur = pd.read_csv(RES / "learning_curve_mean.csv")
    ex = pd.read_csv(RES / "learning_curve_extrapolation.csv").set_index("model")
    law = lambda n, a, b, c: a - b * n ** (-c)  # noqa: E731
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    spec = [("TabPFN v3.5", C_TAB, "D", "TabPFN v3.5"), ("FL-acc", C_FL, "o", "Bounded model (deployed)"),
            ("B1", C_LIN, "s", "Linear baseline")]
    for m, col, mk, lab in spec:
        g = cur[cur.model == m].sort_values("n")
        ax.fill_between(g.n, g["mean"] - g.ci, g["mean"] + g.ci, color=col, alpha=0.14, lw=0)
        ax.plot(g.n, g["mean"], color=col, lw=1.5, marker=mk, ms=3.6, mec=SURF, mew=0.6, zorder=3)
        p = ex.loc[m, ["asymptote", "b", "c"]].to_numpy()
        nn = np.linspace(170, 500, 80)
        ax.plot(nn, law(nn, *p), color=col, lw=1.3, ls=(0, (3, 2)))
        ax.text(505, law(500, *p), f"{lab}\n{law(500, *p):.3f}", color=col if m != "B1" else INK2,
                fontsize=6.8, va="center")
    ax.axvline(170, color=INK2, lw=0.7, ls=(0, (1, 2)))
    ax.text(174, 0.566, "all available data", fontsize=6.6, color=INK2)
    ax.text(335, 0.566, "extrapolated", fontsize=6.6, color=INK2, ha="center", style="italic")
    ax.set_xlim(20, 500)
    ax.set_ylim(0.56, 0.68)
    ax.set_xticks([30, 100, 170, 300, 400, 500])
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Training participants")
    ax.set_ylabel("Mean cross-validated R²\n(10 WAB-R subtests)")
    fig.subplots_adjust(left=0.15, right=0.74, top=0.96, bottom=0.14)
    save(fig, "Figure7_learning_curve")


if __name__ == "__main__":
    train = sys.argv[1]
    fig1()
    fig2(train)
    fig3()
    fig4(train)
    fig5(train)
    fig6()
    fig_conditional()
    fig_spline(train)
    print("saved to", FIG)
