# PNT → WAB-R: calibrated prediction of the WAB-R profile from Philadelphia Naming Test performance

Code, validation results and a browser-based estimator for the study
*Calibrated prediction of the Western Aphasia Battery–Revised profile from Philadelphia Naming Test performance: nonlinear, bounded, and foundation models, and the limits of naming-based inference* (Ahmadi et al., preprint 2026).

**Web tool:** https://saeed-ahmadi.github.io/PNT2WAB/ — enter the PNT response counts (or upload a CSV) and get estimates of all 10 WAB-R subtests, the three composites and the Aphasia Quotient, each with a calibrated prediction interval. Everything runs in your browser; no data leave your computer.

> **An adjunct, not a replacement.** Production subtests are estimated with useful precision; auditory comprehension is not, and comprehension intervals under-cover in severe aphasia. The tool is not suitable for monitoring an individual's change over time and is not a diagnostic device.

## What the model does

| | |
|---|---|
| Input | PNT response counts: correct plus, optionally, the six error types (checked to sum to the items administered and shown as an error profile). The estimate uses naming accuracy, since error types did not improve prediction in the validation study |
| Model | Ridge fractional-logit GLM per WAB-R subtest on logit accuracy (linear, quadratic, cubic terms); predictions stay on each subtest's 0-to-maximum scale |
| Composites | Spontaneous Speech, Auditory Verbal Comprehension, Naming & Word Finding and the AQ are computed from predicted subtests with the WAB-R formulas |
| Intervals | Normalised conformal quantile from nested cross-validation: 90.1–91.2% observed coverage for 90% intervals (lower in severe aphasia) |
| Development data | 213 adults with post-stroke aphasia, single centre, PNT and WAB-R within 7 days |
| Validation | 10 × repeated 5-fold nested cross-validation (no external validation yet) |

Cross-validated performance (fold-averaged R²) of the deployed model, as reported in the paper:

| Outcome | R² | Linear baseline R² | MAE |
|---|:-:|:-:|:-:|
| Naming & Word Finding | .913 | .885 | 0.68 |
| Object Naming | .901 | .854 | 4.30 |
| Aphasia Quotient | .865 | .858 | 6.89 |
| Information Content | .803 | .752 | 0.89 |
| Responsive Speech | .766 | .720 | 1.25 |
| Auditory Word Recognition | .435 | .429 | 5.66 |
| Yes/No Questions | .275 | .285 | 4.22 |

The full table, interval coverage overall and by severity band, and the comparison with neural networks, gradient boosting and TabPFN are on the web tool's *Validation & method* tab and in the paper.

## Repository layout

```
docs/                 the web tool (GitHub Pages serves this folder)
  index.html, app.js, style.css
  predict.js          prediction + intervals (shared with the parity test)
  model.json          the frozen model: coefficients and aggregate interval quantiles only
  validation.json     cross-validated performance shown on the site
pnt2wab/core.py       features, models, conformal helpers
scripts/              analysis pipeline (see "Reproducing the analysis")
tests/                JavaScript–Python parity test and headless-browser test
ANALYSIS_PLAN.md      time-stamped analysis plan with dated revision notes
DATA.md               required input-data format
```

## Reproducing the analysis

The participant data are not public (see *Data availability*). With a file in the format described in [DATA.md](DATA.md):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash reproduce.sh path/to/pNT_TO_WAB.xlsx
```

`reproduce.sh` writes results to `results/` and figures to `figures/` (both git-ignored) and runs, in order: the pre-specified nested cross-validation (`run_cv.py`), the final model export (`fit_final.py`), the revision analyses (`run_revision.py`, `run_revision2.py`), the learning curves (`learning_curve.py`), the figures (`make_figures.py`) and the site data (`export_site_data.py`). The TabPFN analyses (`run_tabpfn.py`, `run_stack.py`) need a GPU for reasonable run time and a free Prior Labs licence token (`TABPFN_TOKEN`) for the v2.5+ weights.

## Tests

```bash
node tests/parity.mjs                      # browser code reproduces Python predictions and intervals
python tests/browser_check.py              # headless-browser test of the web tool (needs: playwright install chromium)
```

`tests/parity.mjs` also runs on every push (GitHub Actions).

## Running the web tool locally

The page loads `model.json` with `fetch`, so serve the folder rather than opening the file:

```bash
python -m http.server --directory docs 8000   # then open http://localhost:8000
```
## Data availability

This repository contains no participant data and nothing derived at the level of individual participants: `docs/model.json` holds only model coefficients, scaling constants and one interval quantile per outcome, and `docs/validation.json` holds summary performance statistics. Analysis results and figures are generated locally by `reproduce.sh` and are not part of the repository. Data are available from the corresponding author on reasonable request, subject to institutional approval.

## Citation

See [CITATION.cff](CITATION.cff). Please cite the paper when using the tool or code.

## Licence

Code: MIT (see [LICENSE](LICENSE)). The estimator is provided for research and decision support without warranty.

Contact: Saeed Ahmadi — ahmadis@email.sc.edu
