# PNT → WAB-R: analysis plan (fixed before results were seen)

Written 2026-10-12, before any model in this folder was evaluated.

## Data
- Primary: `pNT_TO_WAB.xlsx`, n = 213, seven PNT response counts (sum = 175) and
  13 WAB-R subscores from the same assessment window (≤ 1 week).
- The 241-participant file is **not** used yet: it carries only total correct and pairs
  different sessions than the 213 file for most overlapping participants.

## Outcomes
- 10 independent subtests are modelled: Information Content (0–10), Fluency (0–10),
  Yes/No (0–60), Auditory Word Recognition (0–60), Sequential Commands (0–80),
  Object Naming (0–60), Word Fluency (0–20), Sentence Completion (0–10),
  Responsive Speech (0–10), Repetition subscore (0–10).
- The 3 composites are **derived** from predicted subtests with the WAB-R formulas,
  so predicted profiles are always internally consistent:
  Spontaneous Speech = IC + Fluency; Comprehension = (Y/N + AWR + SC) / 20;
  Naming = (ON + WF + SC + RS) / 10.

## Features (two sets)
- `acc`: naming accuracy on the logit scale, log((C + .5) / (175 − C + .5)),
  with quadratic and cubic terms.
- `acc+comp`: `acc` plus the centred log-ratio of the six error categories
  (pseudocount 0.5), i.e. the error *composition* independent of severity.

## Models
| id | model | browser-deployable |
|----|-------|--------------------|
| B0 | training mean | yes |
| B1 | OLS on % correct (the published baseline) | yes |
| B3 | OLS on % correct + 5 error proportions (published) | yes |
| FL-acc | ridge fractional-logit (bounded, ceiling-aware) GLM, `acc` | yes |
| FL-comp | ridge fractional-logit GLM, `acc+comp` | yes |
| MTL | multi-task neural net: shared hidden layer, 10 sigmoid outputs, seed ensemble, `acc+comp` | yes |
| CB | CatBoost MultiRMSE, `acc+comp` | no (reference only) |

All hyperparameters are tuned by inner 5-fold CV inside each outer training fold.

## Validation
- 10 × repeated 5-fold CV, identical folds for every model.
- Metrics per outcome: R², MAE, RMSE, calibration slope (observed on predicted).
- Each model vs B1: mean ΔR² with the corrected repeated k-fold t-test
  (Nadeau & Bengio 2003; Bouckaert & Frank 2004).
- Prediction intervals: normalised split-conformal using inner out-of-fold residuals,
  scaled by a binomial-type spread; reported coverage and width at 80 % and 90 %
  on the outer folds.

## Pre-specified decision rule for the deployed model
1. Only browser-deployable models are eligible.
2. Primary criterion: mean R² across the 10 independent subtests.
3. Choose the **simplest** eligible model (order: B1 < FL-acc < FL-comp < MTL) whose
   mean R² is within 0.005 of the best eligible model.
4. Claims of improvement over B1 require a corrected p < .05 for that outcome;
   otherwise the result is reported as "no detectable improvement".

## Secondary: item-level analysis (added 2026-10-12, before stage 2 was run)

Data: POLAR item-level PNT (78 participants, 686 sessions), linked to master IDs with
the C-STAR enrollment table; WAB-R from `PNT_WAB13_participants.csv`.

Scoring: `Code 1` (first complete attempt). `X` = unscorable → missing. Seven
categories: C, SEM (S + M), FORM (F), PHON (N), ABSTRUSE (AN), UNREL (U), NR.
Item 49 is two items (umbrella / eskimo).

Key constraint: under Rasch-family models the sum score (and category counts) are
sufficient for person parameters, so item-level data can add information only through
item-specific weights or multiple dimensions.

Stage 1 (no WAB data; all 686 sessions): multidimensional nominal-response factor
model, logit P(category k | session u, item i) = β_ik + Σ_d a_ikd θ_ud, ridge-penalised,
D chosen by held-out log-likelihood on 10 % masked responses. Reliability of every
person score is measured across parallel A/B administrations.

Stage 2 (linked participants): the PNT session whose total matches the WAB file's
`pnt_correct` (|diff| ≤ 0.5; A and B pooled when the file holds their mean).
Primary sample: all matched; sensitivity: PNT–WAB gap ≤ 7 days.
Models (ridge fractional-logit, λ tuned in inner CV, 10 × 5 repeated CV, same folds):
- I0 accuracy only (`acc`)
- I1 accuracy + count composition (`acc+comp`)
- I2 accuracy + stage-1 latent scores θ
- I3 175 binary items (SCANT-style)

Pre-specified hypotheses (directional):
- H1: I2 > I0 for the comprehension subtests (Yes/No, Auditory Words, Sequential Commands).
- H2: I2 > I0 for Repetition.
Given n ≈ 60, these are exploratory; the result is reported whatever its direction.

## Revision analyses (added 2026-10-12, after all primary results; exploratory)

Added in response to methodological review; none changes the pre-specified deployed model.
- NL-Ridge: ridge regression on the same three accuracy terms as FL-acc with an identity link
  (unclipped primary; clipped sensitivity), same outer folds and inner tuning
  (`scripts/run_revision.py`). Separates nonlinear predictors from the bounded response.
- Fold-averaged R² (primary) and pooled out-of-fold R² (secondary) reported separately.
- Strict split-conformal intervals (70/30 split of each outer training partition) as a
  sensitivity analysis with a finite-sample guarantee; the primary intervals use
  cross-validated calibration and their coverage is empirical.

## Second-round revision analyses (added 2026-10-12; exploratory)

`scripts/run_revision2.py`:
- FL-spline: fractional-logit on a restricted cubic spline of proportion correct (4 knots).
- Prediction intervals: normalised CV+ made primary (guarantee >= 1 - 2*alpha); the earlier
  cross-validated approximation and split conformal kept as sensitivity analyses; spread-floor
  sensitivity (0.01-0.20); coverage by observed AQ severity band and by recorded aphasia type.
- Calibration-in-the-large added to calibration slope.
- TOST equivalence, FL-acc vs NL-Ridge, margin +/-0.02 R2 (chosen after the primary results).
- Bonferroni threshold changed to .05/14 per contrast family (14 outcomes incl. AQ).
The deployed model (FL-acc) is unchanged; its exported intervals now use CV+.
