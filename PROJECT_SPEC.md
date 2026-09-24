# Build Specification: Sensor-Reduction Study for Machinery Fault Diagnosis

You are building the complete, tested software for an IEEE conference research paper written by a student. Read this entire document before writing any code. Work in the phases in Section 15, and stop for my confirmation at the end of each phase.

---

## 1. Project context

**Paper title:** How Many Sensors Are Enough? An Exhaustive Sensor-Subset Analysis for Rotating Machinery Fault Diagnosis

**Paper type:** An empirical, hypothesis-driven experimental study on a public dataset. It is NOT a new-method paper. We are not trying to maximize accuracy. We are measuring, as rigorously as possible, how diagnosis performance depends on the number and choice of sensors.

**What the software must do:** Train one fixed Random Forest classifier on every possible subset of the 8 sensors in the MaFaulDa dataset. Evaluate each subset with leakage-free cross-validation. Select sensors using training data only. Test three pre-registered hypotheses. Produce publication-ready figures and tables.

**What matters most, in order:**
1. **Validity:** no data leakage, and test data never influences any choice.
2. **Reproducibility:** identical results on every rerun.
3. **Traceability:** every number in the paper comes from a saved results file.
4. **Correctness:** verified by automated tests.
5. Clean, readable code a student can understand and explain to reviewers.

---

## 2. Research questions and hypotheses (fixed; do not change)

- **RQ1:** How does fault-diagnosis performance, overall and per fault class, change as the number of sensors k decreases from 8 to 1?
- **RQ2:** Can a sensor subset selected using only training data match the full 8-sensor configuration?
- **RQ3:** Which sensors contribute most, and does informed selection beat typical subsets of the same size?

- **H1:** Removing one sensor from the oracle-best subset never raises mean macro-F1 by more than `TOLERANCE = 0.01`. Check this for every consecutive pair of k values (k+1 → k).
- **H2:** There exists some k < 8 at which the permutation-selected subset's mean macro-F1 is ≥ (full-configuration mean − 0.01). Also report the oracle version, clearly labeled as an upper bound.
- **H3:** For each k from 1 to 7, the permutation-selected subset scores higher than the median of all k-sensor subsets. Use a one-sided Wilcoxon signed-rank test over the 10 paired split scores, with α = 0.05. Report the same test for forward selection.

---

## 3. Dataset: MaFaulDa (Machinery Fault Database, UFRJ)

Facts to rely on. Verify them against the data; never silently assume.

- **Size:** 1,951 recordings.
- **Recording format:** each recording is one CSV file with **no header**, **8 columns**, and **250,000 rows** (5 seconds at 50 kHz).
- **Column order**, which becomes the canonical `SENSORS` list:

  | Column | Sensor ID | Meaning |
  |---|---|---|
  | 1 | `tach` | Tachometer |
  | 2 | `ub_ax` | Underhang bearing accelerometer, axial |
  | 3 | `ub_rad` | Underhang bearing accelerometer, radial |
  | 4 | `ub_tan` | Underhang bearing accelerometer, tangential |
  | 5 | `ob_ax` | Overhang bearing accelerometer, axial |
  | 6 | `ob_rad` | Overhang bearing accelerometer, radial |
  | 7 | `ob_tan` | Overhang bearing accelerometer, tangential |
  | 8 | `mic` | Microphone |

- **Hardware:** three single-axis accelerometers, one triaxial accelerometer, an analog tachometer, and a microphone. Do not state in code or docs which bearing has the triaxial unit; the documentation does not make this explicit.
- **Classes:** the class label is the **top-level folder name** under the data root. There are 6 classes:

  | Label | Abbreviation (figures) |
  |---|---|
  | `normal` (the healthy class) | N |
  | `imbalance` | IMB |
  | `horizontal-misalignment` | HM |
  | `vertical-misalignment` | VM |
  | `underhang` | UB |
  | `overhang` | OB |

  Deeper subfolders encode fault subtype and severity. Store the relative subfolder path as `condition_path`, but do not use it as the label.
- **Speed:** rotation speeds span roughly 737–3686 rpm, about 12–62 Hz.
- **Location:** the data lives at `data/mafaulda/`. I will download and place it myself. **Do not download the dataset, and never commit data to Git.**
- **Class counts:** do not hardcode per-class counts. Print them, and warn if the total is not 1,951.

---

## 4. Environment and tooling

- Python 3.10 or newer, in a virtual environment (`.venv`).
- Dependencies in `requirements.txt`: numpy, pandas, scipy, scikit-learn (1.3 or newer), matplotlib, joblib, pytest. Add nothing else without asking me.
- After Phase 0, generate `requirements-lock.txt` with `pip freeze`.
- Initialize Git. The `.gitignore` must exclude `data/`, `results/`, `features/`, `.venv/`, `__pycache__/`, and `*.npz`.
- The core pipeline is plain Python scripts. No notebooks in the core pipeline.
- Use `pathlib` for all paths, relative to the repository root. No absolute paths.
- Use `argparse` for every script and the `logging` module for progress output.
- Add type hints and a docstring to every function.
- Fail loudly on bad input: raise clear errors. Never silently skip files.

---

## 5. Repository layout

```
sensor-reduction/
├── common.py                  # ALL constants and shared helpers (single source of truth)
├── 01_extract_features.py
├── 02_checkpoint.py
├── 03_run_experiments.py
├── 04_analyze.py
├── tests/
│   ├── conftest.py            # fixtures; builds synthetic dataset in a tmp dir
│   ├── make_synthetic_data.py
│   ├── test_features.py       # Level A
│   ├── test_data.py           # Level B (real data; skipped if absent)
│   ├── test_leakage.py        # Level C
│   ├── test_pipeline.py       # Level D
│   └── test_sanity.py         # Level E
├── paper/
│   ├── main.tex               # IEEEtran conference skeleton
│   └── refs.bib
├── PROTOCOL.md                # every fixed experimental decision, human-readable
├── README.md
├── requirements.txt
├── requirements-lock.txt
└── .gitignore
```

Generated folders, not committed: `data/`, `features/`, `results/`, `figures/`, `tables/`.

**If this folder already contains `common.py` and the `01_`–`04_` scripts:** treat them as a draft. Review them line by line against this specification, keep what is correct, and fix what is not. Report every discrepancy you find before changing anything.

---

## 6. `common.py`: fixed constants

Every script must import these. Nothing may be redefined elsewhere.

```python
FS = 50_000
SENSORS = ["tach", "ub_ax", "ub_rad", "ub_tan", "ob_ax", "ob_rad", "ob_tan", "mic"]
N_WINDOWS = 5
CLASS_ORDER = ["normal", "imbalance", "horizontal-misalignment",
               "vertical-misalignment", "underhang", "overhang"]
HEALTHY_CLASS = "normal"
SEED = 42
N_SPLITS = 5
N_REPEATS = 2          # 5 x 2 = 10 paired scores (5 alone cannot reach p < 0.05 in Wilcoxon)
N_TREES = 200
TOLERANCE = 0.01       # used by H1 and H2
PERM_REPEATS = 5
BANDS = [(5, 250), (250, 500), (500, 1000), (1000, 2000),
         (2000, 5000), (5000, 10000), (10000, 25000)]
```

The paths `DATA_DIR`, `FEATURES_FILE`, `RESULTS_DIR`, `FIGURES_DIR`, and `TABLES_DIR` are defined relative to the repository root.

Helpers that live in `common.py`:
- `make_model(n_trees=N_TREES)`: returns `RandomForestClassifier(n_estimators=n_trees, class_weight="balanced", n_jobs=-1, random_state=SEED)`. This is **the only model in the study.** No hyperparameter tuning, no other classifiers, no deep learning.
- `load_features()`
- `feature_columns(df, subset)`: returns the columns whose names start with `"<sensor>__"`.
- `encode_labels(df)`: integer codes in `CLASS_ORDER` order.
- `subset_name(subset)`: the canonical name, with sensors ordered as in `SENSORS` and joined by `"+"`. Any input order must give the same name.
- `outer_splits(df, y, split_mode)`: yields `(repeat, fold, train_idx, test_idx)`.
- `inner_split(df, y, train_idx)`: returns `(inner_train_idx, inner_val_idx)`.

---

## 7. Feature extraction: `01_extract_features.py`

For every CSV file under `DATA_DIR`:

1. Load it and validate the shape: 8 columns are required. Warn if there are not 250,000 rows.
2. Set `label` = the first path component relative to `DATA_DIR`, and `condition_path` = the relative parent folder.
3. Estimate `speed_hz` from the full tachometer column. Threshold at `(max + min) / 2`, count rising edges (below → above), and divide by the duration in seconds.
4. Split the recording into `N_WINDOWS` equal, non-overlapping windows of `len // N_WINDOWS` samples, which is 50,000 samples (1 s) each.
5. For every window and every sensor, compute **exactly these 18 features** after removing the mean (DC). Use `eps = 1e-12` in every denominator.

| # | Name | Definition |
|---|---|---|
| 1 | `rms` | sqrt(mean(x²)) |
| 2 | `peak` | max(\|x\|) |
| 3 | `p2p` | max(x) − min(x) |
| 4 | `crest` | peak / rms |
| 5 | `kurtosis` | scipy.stats.kurtosis(x), Fisher definition |
| 6 | `skewness` | scipy.stats.skew(x) |
| 7 | `shape` | rms / mean(\|x\|) |
| 8 | `impulse` | peak / mean(\|x\|) |
| 9 | `clearance` | peak / (mean(sqrt(\|x\|)))² |
| 10 | `spec_centroid` | Σ(f·P) / ΣP, with P = \|rfft(x · Hann window)\|² |
| 11 | `dominant_freq` | frequency of max(P), **excluding the DC bin** |
| 12–18 | `band_<lo>_<hi>` | ΣP inside [lo, hi) / ΣP, for each of the 7 `BANDS` |

**Do NOT include standard deviation.** After mean removal it equals RMS exactly, so it would be a duplicate feature.

6. Name columns `<sensor>__<feature>`, for example `ub_rad__kurtosis`. Metadata columns: `recording_id`, `window`, `label`, `condition_path`, `speed_hz`.
7. Process files in parallel with joblib, sorting files so `recording_id` assignment is deterministic.
8. Write `features/features.csv`. The expected size is about 9,755 rows × 149 columns.
9. Print the number of recordings per class, the total, and the speed range.
10. Command-line options: `--limit N` processes a random but seeded sample of N files, for testing. `--jobs` sets the number of workers.

---

## 8. Cross-validation protocol (leakage prevention)

- **Outer splits.** For each repeat r in 0..1, use `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + r)` with labels stratified and groups set by `split_mode`:
  - `"recording"` (the default): `groups = recording_id`. Windows from one recording never appear in both training and test data.
  - `"speed"`: `groups = pd.qcut(speed_hz, q=10, labels=False, duplicates="drop")`, so whole speed bands are held out. This is the fallback if a ceiling effect appears.
- **Inner split.** This happens only inside a training fold. Take the first split of `StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)` on the training indices, grouped by `recording_id`. That gives roughly 75% inner-train and 25% inner-validation.
- **Test fold isolation.** Test-fold indices must never be passed to any ranking, selection, or model-fitting function. Enforce this with assertions in the code, not just comments.

---

## 9. Checkpoint: `02_checkpoint.py`

This is the go/no-go check before the expensive run. Use repeat 0 only (5 folds) and support `--split recording|speed`.

1. **Leakage check.** Train the full-sensor model twice: once with a naive `StratifiedKFold` over windows, and once with the grouped split. Report both scores and the gap between them.
2. **Baseline.** Report the full-sensor macro-F1 (mean ± sd) with the grouped split.
3. **Single-sensor models.** Report macro-F1 for each of the 8 sensors alone.
4. **Sanity baselines.**
   - Shuffled labels: permute y within the training fold only. The result must be near chance.
   - `DummyClassifier(strategy="most_frequent")`.
5. **Decision guide.**
   - If full ≥ 0.99 **and** (full − best single) ≤ 0.01: print `CEILING EFFECT` and recommend `--split speed`.
   - Else if (full − best single) ≤ 0.01: print a warning.
   - Otherwise print `OK`.
   - If `tach` is the best single sensor, print a warning about a possible recording artifact.
6. Save `checkpoint_results.csv`.

---

## 10. Main experiment: `03_run_experiments.py`

For each outer split `(r, f)`:

1. **Grouped permutation importance** (training fold only):
   - Fit `make_model()` on inner-train with all sensors, and score macro-F1 on inner-validation. That is the base score.
   - For each sensor, apply **one shared row permutation to all 18 of its columns together**, re-predict, and record the base score minus the new score.
   - Repeat `PERM_REPEATS` times and average the drops. **Draw the permutations once per split** with `np.random.default_rng(SEED)` and **reuse the same list of permutations for all 8 sensors**, so the per-sensor drops are paired. Rationale: the sensor ranking comes from comparing sensors, so holding the randomness fixed (common random numbers) removes shuffle noise from that comparison.
   - Ranking = sensors sorted by mean drop, largest first.
2. **Forward selection** (training fold only):
   - Start empty. At each step, add the sensor whose addition gives the highest inner-validation macro-F1.
   - Break ties deterministically by order in `SENSORS`.
   - Continue until all 8 sensors are added. Record the order and the inner score at each step.
3. **Exhaustive evaluation.** For all 255 non-empty subsets, from `itertools.combinations` over k = 1..8, fit `make_model()` on the full outer training fold using only that subset's columns, and predict the test fold.
4. **Save, per split:**
   - `results/<split_mode>/scores_r{r}_f{f}.csv` with columns `repeat, fold, k, subset, macro_f1, accuracy`.
   - `results/<split_mode>/preds_r{r}_f{f}.npz` containing `test_idx`, `y_true` (int8), and one int8 prediction array per subset name.
   - `results/<split_mode>/selection_r{r}_f{f}.json` containing `perm_ranking`, `perm_importance`, `forward_order`, and `forward_path`.
5. **Resume.** If a split's scores file already exists, skip that split.
6. **Selected subsets are not refit.** Because the model and seed are fixed, a selected subset's test score is looked up from the exhaustive results.
7. **Command-line options:**
   - `--split recording|speed`
   - `--quick`: 50 trees, first split only, written to `results/quick_<split_mode>/` so quick results never mix with real ones.
8. Log the elapsed time per split.

The expected full run is 10 splits × 255 subsets = **2,550 score rows**.

---

## 11. Analysis: `04_analyze.py`

### Metrics

- Macro-F1 (primary), accuracy, and per-class recall.
- `missed_detection` = (fault windows predicted `normal`) / (fault windows)
- `false_alarm` = (normal windows predicted as any fault) / (normal windows)
- `misdiagnosis` = (fault windows predicted as a different fault) / (fault windows)

### Main table (`tables/table_main.csv` and `.tex`)

One row per k, with these columns:
- Mean ± sd across all k-subsets, where each subset's score is first averaged over the 10 splits.
- Oracle best score and the oracle subset's name.
- Permutation-selected: mean ± sd across splits.
- Forward-selected: mean ± sd across splits.
- Mean percentile rank of each selected subset among all k-subsets in each split. Show `--` when k = 8.
- One-sided Wilcoxon p-value, selected versus the per-split median. Show `--` when k = 8 or when all differences are zero.

The LaTeX version uses booktabs, has short headers (`$k$`, All subsets, Oracle, Perm.-sel., Fwd.-sel.), uses `$\pm$`, and must fit a single IEEE column.

### `tables/hypotheses.txt`

- The H1, H2, and H3 verdicts, using the exact rules in Section 2.
- For H1, list each violation with its magnitude.
- For H2, report both the fair k (permutation-selected) and the oracle k, and state that the fair number is the answer.
- A note that scores from repeated cross-validation are not fully independent.

### Other tables

`table_per_class.csv`, `table_detection_errors.csv`, `sensor_importance_per_split.csv`, and `sensor_rms_correlation.csv`.

### Figure rules

- Width: 3.5 in for single-column figures, 7.16 in for double-column.
- Serif font, 8 pt.
- Must be readable in grayscale: distinguish series by markers and line styles, not color alone.
- Save every figure as both PDF and PNG (300 dpi).

### Figures

1. `fig1_f1_vs_k`: box plots of subset-mean macro-F1 for each k. Overlay lines for oracle best, permutation-selected, and forward-selected, plus a horizontal line for the full configuration.
2. `fig2_sensor_importance`: bar chart of mean ± sd grouped permutation importance across splits, sorted from largest to smallest.
3. `fig3_per_class_recall`: recall for each class versus k for permutation-selected subsets, pooled over splits. Use the class abbreviations and put the legend above the plot.
4. `fig4_detection_errors`: missed-detection, false-alarm, and misdiagnosis rates versus k (permutation-selected).
5. `fig5_confusion` (double column): row-normalized confusion matrices for the full configuration and for the permutation-selected subset at the fair H2 k (use k = 2 if H2 fails). Use **repeat 0 only**, so each window is counted once.
6. `fig6_sensor_correlation`: Spearman correlation of the RMS feature between sensors, annotated with values.

### Hard rules for analysis

- The oracle-best subset is chosen using test scores. Always label it as an upper bound, never as the answer.
- Never hand-edit numbers. Everything must regenerate from `results/` with one command.
- Support `--split recording|speed`.

---

## 12. Testing requirements

Use pytest. All tests except Level B must run on the synthetic data and pass before any real-data run.

### Synthetic data (`tests/make_synthetic_data.py`)

- Mirror the MaFaulDa folder structure and the 8-column, header-less CSV format.
- Keep it small: 0.2 s recordings (10,000 rows) and about 8–14 recordings per class, placed in a temporary directory.
- Speeds are random between 12 and 60 Hz, with a tachometer pulse train at that speed.
- Inject fault signatures into specific channels so that correct behavior is predictable. For example: imbalance adds a 1× rotation-frequency component to the radial channels; misalignment adds a 2× component; bearing faults add bursts of high-frequency content on their own bearing's channels. Leave `ub_ax` and `ob_rad` mostly uninformative.
- The data generation must be seeded.

### Level A: unit tests (`test_features.py`)

| Test | Expected result |
|---|---|
| Sine wave, amplitude 2, 100 Hz, 1 s at 50 kHz | rms ≈ 1.4142; crest ≈ 1.4142; shape ≈ 1.1107; kurtosis ≈ −1.5; skewness ≈ 0; dominant_freq = 100.0 |
| Gaussian noise, seeded | kurtosis ≈ 0 (within 0.1) |
| Band fractions for a signal with no content below 5 Hz | Sum ≈ 1 |
| Sine plus constant offset | Same features as the plain sine (DC removal works) |
| Tachometer pulse train at 30 Hz | Estimated speed = 30 Hz (within 1%) |
| Feature count | 18 features per sensor; 8 × 18 = 144 feature columns in total |
| No duplicate features | Generate a seeded set of varied signal *shapes* (sines, Gaussian noise, impulse trains, sine plus noise, sums of harmonics). For every pair of features, the ratio between them must NOT be constant across these signals (for example, coefficient of variation of the ratio > 1e-3). This catches exact algebraic duplicates like std = rms. Do not use raw correlation: amplitude-scaled features such as rms, peak, and p2p are legitimately highly correlated. |
| `subset_name` | Returns the same canonical string for any permutation of the input |

### Level B: real-data validation (`test_data.py`)

Mark these tests `@pytest.mark.realdata` and skip them automatically if `data/mafaulda/` is absent.

- There are 1,951 CSV files.
- A seeded sample of at least 50 files all have shape (250,000, 8).
- The feature table has no NaN or infinite values.
- Every label is in `CLASS_ORDER`, and every recording has exactly 5 windows.
- Estimated speeds lie within 10–65 Hz.
- Print class counts for manual comparison with the documentation.

### Level C: leakage tests (`test_leakage.py`)

- For every outer split in both split modes, the training and test sets share no `recording_id`.
- In speed mode, the training and test sets share no speed band.
- Inner-train and inner-validation indices are subsets of the outer training indices and never touch test indices.
- Inner-train and inner-validation share no `recording_id`.
- The selection functions raise an error if they are given any test index (this tests the assertion guards).

### Level D: integration tests (`test_pipeline.py`)

- **End to end:** run scripts 01 → 02 → 03 (with reduced trees) → 04 on the synthetic data. All expected output files exist.
- **Determinism:** two identical quick runs into separate folders produce byte-identical score CSVs and identical selection JSONs.
- **Resume:** after an interrupted run is restarted (simulate by deleting one split's files), the final results are identical to an uninterrupted run.
- **Completeness:** a full synthetic run has exactly (number of splits × 255) score rows, with each subset appearing once per split.
- **Lookup:** each selected subset's reported score equals its exhaustive entry.

### Level E: sanity tests (`test_sanity.py`)

- Shuffled labels give macro-F1 at or near chance (< 0.35 on the synthetic data).
- The full-sensor model beats the dummy baseline by a wide margin.
- On the synthetic data, the injected informative channels rank above `ub_ax` and `ob_rad` in permutation importance.
- Rows of every confusion matrix sum to the number of test windows in that class.

### Level F: result validation

Implement these as runtime checks inside `04_analyze.py`, raising an error if one fails.

- 2,550 rows are present for a full real run (or the expected count for the current configuration).
- The full-configuration score is identical across the columns where it appears.
- All metrics lie in [0, 1], and the p-values in [0, 1] or are missing.

---

## 13. Supporting documents

- **`PROTOCOL.md`:** a plain-language list of every fixed decision in Sections 2, 3, 6, 7, 8, and 10, with a one-line justification for each. This becomes the source for the paper's methodology section.
- **`README.md`:** setup, dataset placement, run order with expected runtimes, how to run tests, the quick-mode and speed-mode options, and an output-to-paper mapping table:

  | Output | Paper element |
  |---|---|
  | Figure 1 and main table | RQ1, H1, H2 |
  | Figure 2 and main table | RQ3, H3 |
  | Figures 3–5 | Fault sensitivity |
  | Figure 6 | Redundancy discussion |

- **`paper/main.tex`:** an `IEEEtran` conference skeleton with sections I–VI: Introduction, Related Work, Dataset and Methodology, Results, Discussion, Conclusion.
  - It includes `\input{../tables/table_main.tex}`, and `\includegraphics` lines for the six figures with `figure` or `figure*` as appropriate.
  - It uses `\usepackage{booktabs,graphicx,amsmath,cite}` and an empty `refs.bib` with commented example entries.
  - Every result sentence is a `% TODO` placeholder. **Do not write any numbers, findings, or conclusions.**
  - Note that `IEEEtran.cls` may need to come from Overleaf or CTAN if it is not installed locally.

---

## 14. Things you must NOT do

- Do not invent, estimate, or pre-write any results or conclusions.
- Do not change any constant, feature, split, threshold, or model setting to improve scores. If you believe a protocol change is needed, **stop and explain why**, and wait for my decision.
- Do not add other models, hyperparameter tuning, feature scaling, deep learning, or extra dependencies.
- Do not rank or select sensors using data from test folds.
- Do not split windows randomly for any reported result. The naive split exists only in the leakage check.
- Do not download the dataset or commit data or results to Git.
- Do not silence warnings globally, and do not catch broad exceptions just to keep a run going.

---

## 15. Phases (stop after each one)

After each phase: run `pytest`, commit with a descriptive message, then give me a short report covering what was built, test results, any discrepancies or concerns, and what comes next. **Wait for my "continue" before starting the next phase.**

| Phase | Work | Exit criteria |
|---|---|---|
| **0. Setup and review** | Virtual environment, requirements files, Git, `.gitignore`, folder layout, `common.py`. If draft scripts exist, review them against this specification and list the discrepancies. | Environment installs; discrepancy report delivered |
| **1. Features** | `make_synthetic_data.py`, `01_extract_features.py`, Level A tests | Level A passes; the synthetic feature table has 144 feature columns |
| **2. Splits and checkpoint** | Split helpers with assertion guards, `02_checkpoint.py`, Level C tests, the shuffled-label and dummy parts of Level E | Levels A, C, and E (partial) pass |
| **3. Experiment runner** | `03_run_experiments.py`, Level D tests, the remaining Level E tests | Determinism, resume, completeness, and lookup tests pass |
| **4. Analysis** | `04_analyze.py`, all figures and tables, Level F checks. Show me the synthetic-data figures so I can review their layout. | The full synthetic pipeline runs with one command sequence; figures render correctly |
| **5. Docs and freeze** | `PROTOCOL.md`, `README.md`, `paper/` skeleton, `requirements-lock.txt`. Run the complete test suite, then create the Git tag `v1.0-protocol`. | All tests pass; tag created |

**After Phase 5, the real-data run is done together with me:**
1. I place the dataset in `data/mafaulda/`.
2. Run the Level B tests.
3. Run `01_extract_features.py`.
4. Run `02_checkpoint.py` and report the decision. **Stop here** so I can choose between `recording` and `speed`.
5. Run `03_run_experiments.py`, which will take several hours.
6. Run `04_analyze.py` and report the hypothesis verdicts **factually, without interpretation**.

---

## 16. Definition of done

- [ ] All tests (Levels A, C, D, E, F) pass on synthetic data, and Level B passes on real data.
- [ ] Protocol tag `v1.0-protocol` exists and was created before the first real run.
- [ ] Two independent runs produce identical results.
- [ ] Every figure and table regenerates from `results/` with one command.
- [ ] `PROTOCOL.md` fully documents the methodology.
- [ ] The repository is clean and ready for public release, with pinned versions.
