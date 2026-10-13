#!/usr/bin/env bash
# Reproduce the analyses, figures and web-tool data from the participant workbook.
# usage: bash reproduce.sh path/to/pNT_TO_WAB.xlsx
set -euo pipefail
DATA="${1:?usage: bash reproduce.sh path/to/data.xlsx}"
python scripts/run_cv.py "$DATA"            # pre-specified nested CV (all primary models)
python scripts/fit_final.py "$DATA"         # deployed model + CV+ export -> docs/model.json
python scripts/run_revision.py "$DATA"      # NL-Ridge, pooled vs fold R2, interval variants
python scripts/run_revision2.py "$DATA"     # spline, CV+, severity coverage, TOST
python scripts/learning_curve.py "$DATA"    # learning curves (uses TabPFN; GPU recommended)
python scripts/make_figures.py "$DATA"      # manuscript figures
python scripts/export_site_data.py          # docs/validation.json
node tests/parity.mjs                       # JavaScript reproduces Python
