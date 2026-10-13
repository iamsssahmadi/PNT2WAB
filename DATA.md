# Input data format

The analysis scripts read one Excel workbook (first sheet), one row per participant, with these columns (names must match exactly):

| Column | Content |
|---|---|
| `ParticipantID` | Unique identifier |
| `pntCorrect`, `pntFormal`, `pntMixed`, `pntNR`, `pntNeologism`, `pntSemantic`, `pntUnrelated` | PNT response-category counts; the seven must sum to the number of items (175) |
| `WABInformation Content` | 0–10 |
| `WABFluency Rating` | 0–10 |
| `WABComprehension Yes/No` | 0–60 |
| `WABComprehension Auditory Words` | 0–60 |
| `WABComprehension Sequential Commands` | 0–80 |
| `WABObject Naming` | 0–60 |
| `WABWord Fluency` | 0–20 |
| `WABSentence Completion` | 0–10 |
| `WABResponsive Speech` | 0–10 |
| `WABRepetition Subscore` | 0–10 |
| `WABSpontaneous Speech Total` | 0–20 (composite) |
| `WABComprehension Subscore` | 0–10 (composite) |
| `WABNaming subscore` | 0–10 (composite) |

Descriptive columns used only for reporting (not predictors): `Race`, `Sex`, `Days Post-Stroke`, `Aphasia Type from WAB`.

## Web tool batch CSV

`id,pnt_correct,pnt_total,semantic,formal,mixed,neologism,unrelated,no_response` — only `id` and `pnt_correct` are required; `pnt_total` defaults to 175. If a row has error counts, correct + errors must equal `pnt_total`; blank error cells mean the breakdown was not entered. A template can be downloaded from the tool's *Batch* tab.
