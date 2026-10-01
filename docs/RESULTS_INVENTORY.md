# Results inventory

Every generated table and figure in this repository, what it shows, and which
part of the paper it feeds. This is the map for writing and for a reader
checking a claim against its evidence.

Everything listed here regenerates from `results/` with one command per
analysis. No number anywhere is entered by hand. See [PROTOCOL.md](../PROTOCOL.md)
for the experimental decisions behind them.

---

## The three analyses

| Directory suffix | Role | Sensors | Subsets | Rows |
|---|---|---|---|---|
| `speed` | **PRIMARY** — held-out speed bands | 7 | 127 | 1,270 |
| `recording` | **SECONDARY** — held-out recordings, robustness | 7 | 127 | 1,270 |
| `secondary_8sensor_speed` | **SUPPLEMENTARY** — tachometer included, artefact demonstration | 8 | 255 | 2,550 |

`speed` is primary and `recording` is a robustness analysis; both are run in
full and both are reported, so no selection was made on which gave a more
interesting result (PROTOCOL.md §5.2). The 8-sensor run is the artefact
demonstration and is **not** a result of the study (PROTOCOL.md §2.2–2.5).

Each analysis has its own `results/<suffix>/`, `tables/<suffix>/` and
`figures/<suffix>/`. Outputs are separated by analysis so one cannot overwrite
another, and every figure caption names its evaluation protocol in the first
sentence so a figure cannot be misfiled.

**Regenerate:**

```bash
python 04_analyze.py --split speed --expect-full-run
python 04_analyze.py --split recording --expect-full-run
python scripts/supplementary_tables.py --split speed
python scripts/supplementary_tables.py --split recording
python scripts/secondary_8sensor.py --analyse-only
python scripts/tach_learnability.py
python scripts/artifact_exploitation.py
```

---

## Figures

All seven exist for each of the three analyses, as PDF and 300 dpi PNG.
Captions are generated into `figures/<suffix>/captions.tex` and `\input` by the
paper, so a caption cannot drift from what the code did.

| Figure | Shows | Feeds |
|---|---|---|
| `fig1_f1_vs_k` | Macro-F1 against sensor count *k*. Box plots over all C(7,*k*) subsets, with the oracle, permutation-selected and forward-selected curves and the full-configuration line. Whiskers span full min–max, not 1.5×IQR, because every subset is enumerated. | **RQ1, H1, H2** — the headline result |
| `fig2_sensor_importance` | Grouped permutation importance per sensor, mean ± sd across the 10 splits. Measures MARGINAL contribution. | **RQ3, H3** |
| `fig3_per_class_recall` | Recall for each of the 6 classes against *k*, permutation-selected, pooled over splits. | Fault sensitivity |
| `fig4_detection_errors` | Missed-detection, false-alarm and misdiagnosis rates against *k*. | Fault sensitivity; the practitioner-facing framing |
| `fig5_confusion` | Row-normalised confusion matrices, full configuration vs the fair H2 *k*. Repeat 0 only. | Fault sensitivity |
| `fig5b_confusion_k1` | Same, but full configuration vs *k*=1, where the panels actually differ. | Fault sensitivity — **pick one of fig5 / fig5b** |
| `fig6_sensor_correlation` | Spearman ρ of the RMS feature between sensors. Diagonal masked, diverging scale from the off-diagonal data. | Redundancy discussion |
| `fig6b_mean_abs_correlation` | Mean \|ρ\| over all 18 matched features. Diagonal masked, sequential 0–max scale. | Redundancy discussion — **pick one of fig6 / fig6b** |

Two deliberate choices left open: **fig5 vs fig5b** and **fig6 vs fig6b**. Both
pairs are kept so the choice can be made on the real figures.

---

## Tables — all three analyses

| File | Contains | Feeds |
|---|---|---|
| `table_main.csv` | One row per *k*: candidate count C(*n*,*k*), all-subsets mean ± sd, oracle best and its subset, permutation- and forward-selected means ± sd, percentile ranks, Wilcoxon p-values. | **RQ1, RQ2, H1–H3** — the main table |
| `table_main.tex` | The same, as a single-column IEEE booktabs table, `\input` directly. Caption states the oracle is an upper bound. | Results section |
| `hypotheses.txt` | H1, H2, H3 verdicts with every threshold stated inline; the fair/oracle distinction; the Holm post-hoc block; the non-independence note. | **Every hypothesis claim** |
| `table_per_class.csv` | Per-class recall against *k*, permutation-selected, pooled, plus mean recall. | Fault sensitivity; source for fig3 |
| `table_detection_errors.csv` | Missed detection, false alarm, misdiagnosis against *k*. | Fault sensitivity; source for fig4 |
| `sensor_importance_per_split.csv` | Grouped permutation importance, one row per split. | RQ3; source for fig2; stability claims |
| `sensor_rms_correlation.csv` | Spearman ρ of RMS between sensors. | Redundancy; source for fig6 |
| `sensor_mean_abs_correlation.csv` | Mean \|ρ\| over 18 features. | Redundancy; source for fig6b |
| `run_metadata.json` | Commit, dirty flag and its scope, library versions, seed, command line, inputs consumed. | Reproducibility statement |

### Supplementary, in `speed` and `recording`

| File | Contains | Feeds |
|---|---|---|
| `supp_sensor_profile.csv` | Per sensor: marginal permutation importance, standalone single-sensor macro-F1, mean \|ρ\| to the other sensors — the three views side by side. | RQ3 discussion: separating redundancy from absence of information |
| `supp_oracle_membership.csv` | How often each sensor appears in the oracle subset at each *k*. | RQ3; which sensors the upper bound relies on |
| `supp_selected_membership.csv` | The same counts for permutation- and forward-selected subsets. | RQ3; selection stability |

### Supplementary, in `secondary_8sensor_speed` only

| File | Contains | Feeds |
|---|---|---|
| `supp_single_sensor_scores.csv` | Single-sensor macro-F1 for all 8 channels including the tachometer. | Artefact subsection |
| `supp_single_sensor_per_class.csv` | Per-class recall for the tachometer and the `ob_rad` control, pooled. | Artefact subsection |
| `supp_confusion_tach.csv` | Confusion matrix, tachometer alone. | Artefact subsection |
| `supp_confusion_ob_rad.csv` | Confusion matrix, control sensor alone. | Artefact subsection |
| `supp_artifact_exploitation.csv` | **The decisive test.** Accuracy on affected vs unaffected bearing-class windows, for tach alone, the control, and the full 8- and 7-sensor models, with window counts. | **Artefact subsection — the central evidence** (PROTOCOL.md §2.5) |

---

## Raw results

`results/<suffix>/` holds what everything above is computed from:

| File | Contains |
|---|---|
| `scores_r{r}_f{f}.csv` | One row per subset: repeat, fold, *k*, subset name, macro-F1, accuracy |
| `preds_r{r}_f{f}.npz` | `test_idx`, `y_true` (int8), and one int8 prediction array per subset |
| `selection_r{r}_f{f}.json` | Permutation ranking and importances, forward order and path, the fold's label sets |
| `checkpoint_results.csv` | Go/no-go checks: leakage, baseline, single sensors, sanity baselines, class coverage |
| `run_metadata.json` | Provenance for the run |

Present for all 10 splits in each analysis. The 8-sensor directory has **no
`checkpoint_results.csv`**: the checkpoint was deliberately not run for a
configuration the study excludes.

`data/` and `features/` are **not** committed — the dataset is downloaded by the
user and `features.csv` (26 MB) regenerates from it.

---

## What is NOT here, deliberately

- **No checkpoint for the 8-sensor configuration.** Running it would mean
  reporting go/no-go statistics for a configuration the study excludes.
- **No `results/quick_*/` or `results/dryrun_*/`.** Smoke tests and timing
  probes, excluded from version control.
- **No numbers typed by hand anywhere.** If a figure and a table disagree, the
  table is wrong by construction and both should be regenerated.
