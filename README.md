# How Many Sensors Are Enough?

An exhaustive sensor-subset analysis for rotating machinery fault diagnosis on
the MaFaulDa dataset.

One fixed Random Forest is trained on **every one of the 127 non-empty subsets**
of the 7 sensors, evaluated with leakage-free cross-validation, with sensors
selected using training data only. The study measures how diagnosis performance
depends on the number and choice of sensors. It is not an attempt to maximise
accuracy.

Every fixed decision, and the reason for it, is in [PROTOCOL.md](PROTOCOL.md).

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # Linux/macOS
```

`requirements-lock.txt` holds the exact versions the results were produced with.
Use it to reproduce them exactly:

```bash
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
```

## Dataset

Download MaFaulDa from the UFRJ website yourself and unzip it so the six class
folders sit directly under `data/mafaulda/`:

```
data/mafaulda/
├── normal/
├── imbalance/
├── horizontal-misalignment/
├── vertical-misalignment/
├── underhang/
└── overhang/
```

The pipeline never downloads the dataset, and `data/`, `features/`, `results/`,
`figures/` and `tables/` are all excluded from Git.

## Run order

```bash
python 01_extract_features.py
python 02_checkpoint.py                 # stop and read the decision
python 03_run_experiments.py
python 04_analyze.py --expect-full-run
```

| Step | Command | Expected runtime | Output |
|---|---|---|---|
| 1 | `01_extract_features.py` | 30–60 min (1,951 files, all cores) | `features/features.csv` |
| 2 | `02_checkpoint.py` | 5–15 min | `results/<split>/checkpoint_results.csv` |
| 3 | `03_run_experiments.py` | **hours**; run overnight | `results/<split>/scores_*.csv`, `preds_*.npz`, `selection_*.json` |
| 4 | `04_analyze.py` | 1–5 min | `tables/`, `figures/` |

Runtimes are for a typical laptop and scale with core count. **Step 3 prints the
elapsed time for each split and a running estimate of the time remaining**, so
you can tell within the first split or two whether the full run is closer to
three hours or thirty.

Step 3 resumes automatically: if it is interrupted, rerun the same command and it
skips the splits already written.

### Before the real run

```bash
python 01_extract_features.py --limit 60    # seeded sample, for a smoke test
python 03_run_experiments.py --quick        # first split only, 50 trees
```

Quick results go to `results/quick_<split>/` and can never mix with real ones.
Rerunning `--quick` is a no-op rather than computing a different split.

### Choosing the split mode

`02_checkpoint.py` is the go/no-go gate. It reports the leakage inflation, the
full-sensor baseline, every single-sensor model, the shuffled-label and dummy
sanity baselines, and **class coverage per speed band and per test fold**. Read
the coverage section before choosing a split mode.

- `CEILING EFFECT` — one sensor already matches the full set and the score is at
  0.99. Rerun steps 2–4 with `--split speed`.
- `WARNING` — the curve may be flat; consider `--split speed`.
- `OK` — proceed with this split.

```bash
python 02_checkpoint.py --split speed
python 03_run_experiments.py --split speed
python 04_analyze.py --split speed --expect-full-run
```

Speed mode holds out whole rotation-speed bands. It refuses to split at all if no
banding puts every class in every test fold; see PROTOCOL.md §5.1.

## Tests

```bash
python -m pytest                    # everything (about 12 minutes)
python -m pytest -m "not slow"      # fast subset
python -m pytest -m realdata        # Level B; skipped unless data/mafaulda exists
```

All tests except Level B run on seeded synthetic data and must pass before any
real-data run.

| Level | File | What it checks |
|---|---|---|
| A | `tests/test_features.py` | Feature values against analytic signals; no duplicate features |
| B | `tests/test_data.py` | Real data: file count, shapes, labels, speed range |
| C | `tests/test_leakage.py` | No shared recordings or speed bands; guards fire |
| D | `tests/test_pipeline.py`, `tests/test_experiments.py` | Determinism, resume, completeness, lookup-equals-refit |
| E | `tests/test_sanity.py` | Shuffled labels at chance, dummy baseline, ranking, confusion totals |
| F | `tests/test_analysis.py` | Row counts, metric ranges, provenance; Level F lives inside `04_analyze.py` |

## Output to paper mapping

| Output | Paper element |
|---|---|
| Figure 1 and main table | RQ1, H1, H2 |
| Figure 2 and main table | RQ3, H3 |
| Figures 3–5 | Fault sensitivity |
| Figures 6 and 6b | Redundancy discussion |

| File | Contains |
|---|---|
| `tables/table_main.csv` / `.tex` | Macro-F1 against *k* for every strategy; `.tex` drops straight into the paper |
| `tables/hypotheses.txt` | H1, H2 and H3 verdicts, with every threshold stated inline |
| `tables/table_per_class.csv` | Per-class recall against *k* |
| `tables/table_detection_errors.csv` | Missed detection, false alarm, misdiagnosis |
| `tables/sensor_importance_per_split.csv` | Grouped permutation importance, one row per split |
| `tables/sensor_rms_correlation.csv` | Spearman ρ of RMS between sensors |
| `tables/sensor_mean_abs_correlation.csv` | Mean \|ρ\| over all 18 features |
| `figures/captions.tex` | Generated figure captions, `\input` by the paper |
| `*/run_metadata.json` | Commit, versions, timestamp and inputs for every step |

## Reading the results honestly

- **The oracle column and the oracle *k* are an upper bound**, selected using
  test scores. They are not achievable in practice and are never the answer. The
  permutation-selected and forward-selected results are.
- **H1 is judged on the fair curves.** The oracle curve is a maximum over
  C(7,*k*) candidates, so its upward bias grows with the number of candidates and
  peaks at the middle *k*. See PROTOCOL.md §7.
- **A sensor with low permutation importance is not necessarily useless.**
  Permutation importance measures marginal contribution, so a redundant sensor
  scores near zero. Read it alongside the forward-selection order and the
  correlation matrices. See PROTOCOL.md §8.2.
- **The 10 split scores are not mutually independent**, because the two repeats
  reuse the same recordings. Treat the p-values as supporting evidence alongside
  the effect sizes and percentile ranks.

## Repository layout

```
common.py                  all constants and shared helpers (single source of truth)
01_extract_features.py     raw CSV -> features/features.csv
02_checkpoint.py           go/no-go checks before the expensive run
03_run_experiments.py      selection + all 127 subsets on every split
04_analyze.py              tables, figures, hypothesis verdicts, Level F checks
tests/                     Levels A-F
paper/                     IEEEtran skeleton and bibliography
PROTOCOL.md                every fixed decision, with reasons
```
