# Experimental protocol

Every fixed decision in this study, with the reason for it. This document is the
source for the paper's methodology section.

**Every decision recorded here was fixed before any MaFaulDa data was
processed.** The protocol was frozen at the Git tag `v1.0.1-protocol`, which was
created before the first real-data run and is the exact code that produces
the real results. Nothing below was chosen, tuned or revised in response to a
result on the real dataset. Where a decision was revised during development, it
was revised on synthetic data only, and the reason is stated.

**Repository history.** On 2026-09-24, before any MaFaulDa data was processed,
the development history was squashed to a single commit and commit authorship
was corrected. `v1.0-protocol` was removed and `v1.0.1-protocol` was recreated
on the new commit. **No experimental decision changed in that operation**, and
no code, constant or threshold differs as a result: the squashed tree is
byte-identical to the pre-squash tree. The earlier two-tag scheme
(`v1.0-protocol` superseded by `v1.0.1-protocol`, differing only in the fig. 6b
colour scale, the `--dry-run` probe and the run script) no longer exists in the
history; that distinction is recorded here instead.

---

## 1. Research questions and hypotheses

Fixed in advance; not revised after seeing any result.

- **RQ1** How does fault-diagnosis performance, overall and per fault class,
  change as the number of sensors *k* decreases from 8 to 1?
- **RQ2** Can a sensor subset selected using only training data match the full
  8-sensor configuration?
- **RQ3** Which sensors contribute most, and does informed selection beat
  typical subsets of the same size?

- **H1** Removing one sensor never raises mean macro-F1 by more than
  `TOLERANCE = 0.01`, checked for every consecutive pair of *k* values
  (*k*+1 → *k*). See §7 for how this is judged.
- **H2** There exists some *k* < 8 at which the permutation-selected subset's
  mean macro-F1 is at least (full-configuration mean − 0.01).
- **H3** For each *k* from 1 to 7, the permutation-selected subset scores higher
  than the median of all *k*-sensor subsets. One-sided Wilcoxon signed-rank test
  over the 10 paired split scores, α = 0.05. The same test is reported for
  forward selection.

---

## 2. Dataset

| Decision | Value | Why |
|---|---|---|
| Dataset | MaFaulDa (UFRJ), 1,951 recordings | Public, multi-sensor, multi-fault; allows exhaustive subset analysis |
| Sampling rate | 50 kHz | Property of the dataset |
| Sensors | 8 channels, fixed order `tach, ub_ax, ub_rad, ub_tan, ob_ax, ob_rad, ob_tan, mic` | The CSV column order; defines the canonical subset naming |
| Classes | 6, from the top-level folder name | The documented class structure |
| Deeper folders | Stored as `condition_path`, never used as a label | Subtype and severity are not the classification target |
| Class counts | Printed, never hardcoded; a total other than 1,951 raises a warning | Detects an incomplete download instead of silently analysing partial data |

The hardware comprises three single-axis accelerometers, one triaxial
accelerometer, an analog tachometer and a microphone. **Which bearing carries
the triaxial unit is not stated anywhere in this repository**, because the
dataset documentation does not make it explicit.

### 2.1 What the two bearing classes mean

`underhang` and `overhang` denote bearing **positions** relative to the rotor,
not defect types. Defect types — ball, cage, outer race — appear in the
**subfolders** beneath them, and this study's 6-class choice **merges them into
the two position classes**. A model here therefore answers "which bearing is
faulty", not "what kind of bearing defect is it".

The label rule is unchanged and unconditional: the class is the top-level
folder name, and `condition_path` records the rest of the path without ever
being used as a label.

Two notes on provenance:

- The dataset's own web page words these same two classes differently, so the
  mapping must be stated explicitly for a reader arriving from that page.
- The claim that the subfolders carry defect types is **to be confirmed against
  the real directory tree**, not reasoned about. The arbiter is
  `tests/test_data.py::test_top_level_folders_are_the_six_classes`, which
  compares the actual top-level folder names against `CLASS_ORDER`. If the real
  names differ, that is reported as a finding; `CLASS_ORDER` is not adapted to
  fit them.

---

## 3. Features

18 features per sensor per window, computed after removing the mean, with
`eps = 1e-12` in every denominator.

| Decision | Value | Why |
|---|---|---|
| Windows | 5 equal, non-overlapping, `len // 5` samples each | More training rows than one per recording, without overlap-induced leakage |
| Features per sensor | Exactly 18, identical for every sensor | A subset's score must reflect the sensors, not a differing feature count |
| Standard deviation | **Excluded** | After DC removal it equals RMS exactly, so it would be a duplicate feature. `tests/test_features.py` enforces that no pair of features has a constant ratio, with std-vs-RMS as a negative control |
| Spectrum | `\|rfft(x · Hann)\|²` | Hann reduces spectral leakage |
| `dominant_freq` | Excludes the DC bin | After mean removal the DC bin carries only leakage |
| Frequency bands | 7 fixed bands, 5 Hz to 25 kHz | Fixed before any result; relative band energy is scale-free |
| Speed estimate | Rising edges of the tachometer above `(max + min) / 2`, divided by duration | Simple, robust, and needs no assumptions about pulse shape |
| `recording_id` | Assigned from the sorted file list | Makes the cross-validation groups identical on every run and machine |

---

## 4. Model

One classifier, used identically for every sensor subset:

```python
RandomForestClassifier(n_estimators=200, class_weight="balanced",
                       n_jobs=-1, random_state=42)
```

**No hyper-parameter tuning, no alternative classifiers, no feature scaling, no
deep learning.** This is not a new-method paper. Any difference between subsets
must come from the sensors, so the model is held fixed. `class_weight="balanced"`
is set because `normal` is a comparatively small class; it was chosen in advance,
not tuned.

---

## 5. Cross-validation

| Decision | Value | Why |
|---|---|---|
| Outer splits | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED+r)`, repeats r = 0, 1 | 5 × 2 = 10 paired scores; 5 alone cannot reach p < 0.05 in a one-sided Wilcoxon test |
| Default grouping | `recording_id` | The five windows of one recording are near-duplicates; splitting them across the boundary leaks |
| Fallback grouping | Speed band | Holds out whole rotation speeds, so the model is tested on speeds it never saw. Used only if the recording split shows a ceiling effect |
| Inner split | First split of `StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)` on the training fold, grouped by recording | Roughly 75/25; the only data available to selection |
| Test-fold isolation | Enforced by assertions in `common.assert_no_test_indices`, called by every selection and fitting path | A leak fails the run loudly instead of quietly inflating a score |

The naive random-window split exists **only** inside `02_checkpoint.py`, as the
measurement of how much leakage would inflate the score. No reported result
anywhere else splits windows at random.

### 5.1 Speed-band construction

A test fold missing a class is a split **defect**, not a result to be scored.
`choose_speed_q` therefore tries `q = 10, 8, 6` quantile bands in order and takes
the first that puts all 6 classes in every test fold, logging the `q` actually
used. If no `q` works it raises, rather than scoring a degenerate split.

On the real dataset every condition was recorded across the speed range, so
`q = 10` is expected to succeed immediately; this ladder is a guard, not an
expected path.

**Measured on synthetic data, and counterintuitive enough to record:** the ladder
is *not* monotonic in fixture size. Coarsening helps only in a middle regime.
With 5 folds and `q` bands, each fold holds roughly `q/5` bands, so a smaller `q`
means *fewer* bands per fold and therefore *more* chance of missing a class.

| Recordings per class | Total | q=10 | q=8 | q=6 |
|---|---|---|---|---|
| 10–14 | 71 | 3 bad folds | 4 bad | 2 bad |
| 18–22 | 123 | 2 bad | 2 bad | **OK** |
| 26–32 | 175 | **OK** | OK | 2 bad |
| 34–40 | 228 | OK | OK | OK |

Consequently the synthetic speed-mode tests use a larger fixture,
`SPEED_N_PER_CLASS = (26, 32)` — the smallest range at which `q = 10` covers
every class in every fold. The smaller default fixture (10–14 per class) cannot
support speed mode at all, and `choose_speed_q` correctly refuses to split it.
This is a test-fixture decision only; it does not touch the protocol applied to
real data.

---

## 6. Scoring

### 6.1 The per-fold label set

For each outer fold the scoring label set is computed **once**, as the sorted
unique values of `y_true` in that fold's test set, and the same label set is
passed to every scoring call for every subset in that fold. `FoldScorer`
re-derives the set from the `y_true` it is handed on every call and refuses to
score if it differs, so a subset cannot silently be measured on a different class
set from its neighbours.

**Why not scikit-learn's default?** The default label set is the union of
`y_true` and `y_pred`, so the denominator would vary between subsets *inside the
same fold*: a subset that never predicts a class would be scored over a different
number of classes from one that does. Every comparison in this study is paired
within a fold (subset against subset, selected against the per-fold median), so
that is unacceptable.

**Why not all 6 classes unconditionally?** That would penalise a fold for a class
it does not contain, scoring a forced 0 for a class that was never there. Fixing
the set from `y_true` keeps every subset in a fold comparable without inventing a
penalty. A class that *is* present but is never predicted still scores 0, which
is correct.

The scorer raises below 5 classes and warns loudly below 6, in both split modes.

`per_class_recall` returns a fixed-length array of 6, with **NaN** for a class
absent from `y_true`, so the per-class figure stays aligned across *k* and
absent classes are excluded from any averaging rather than dragged down by a
fictitious zero.

### 6.2 Metrics

Macro-F1 is primary. Accuracy and per-class recall are reported alongside.
Detection errors are defined as:

- `missed_detection` = fault windows predicted `normal` / fault windows
- `false_alarm` = normal windows predicted as any fault / normal windows
- `misdiagnosis` = fault windows predicted as a *different* fault / fault windows

---

## 7. How H1 is judged

The **primary H1 verdict is taken from the fair curves** — permutation-selected
and forward-selected — and both are reported. The oracle curve is reported as
**secondary, clearly caveated** evidence.

**Why.** The oracle at each *k* is a maximum over C(8,*k*) candidate subsets: 8,
28, 56, 70, 56, 28, 8, 1 for *k* = 1…8. A maximum over more candidates carries
more upward bias, so the oracle curve is inflated most at middle *k*, peaking at
*k* = 4 where C(8,4) = 70, purely from maximisation and independently of any
sensor effect. An apparent H1 violation at middle *k* can therefore be an
artefact of that bias rather than evidence of a redundant or harmful sensor. The
fair curves involve no test-set maximisation and so carry no such bias.

The main table includes a **Candidates C(8,k)** column so a reader can see the
bias source directly, and `hypotheses.txt` states the primary and secondary
verdicts separately with the bias explained inline.

Decided before any MaFaulDa run.

---

## 8. Selection, performed on training data only

### 8.1 Grouped permutation importance

Fit on inner-train with all sensors; score macro-F1 on inner-validation for the
base score. For each sensor, apply **one shared row permutation to all 18 of its
columns together** — so the sensor is destroyed as a unit rather than having its
own features scrambled against each other — re-predict, and record base minus
new.

**Common random numbers.** The `PERM_REPEATS = 5` permutations are drawn **once
per split** with `np.random.default_rng(SEED)` and the *same list is reused for
all 8 sensors*, so the per-sensor drops are paired. The ranking comes from
comparing sensors against each other, so holding the randomness fixed removes
shuffle noise from that comparison.

Ranking is by mean drop, largest first, ties broken by `SENSORS` order.

### 8.2 Documented limitation of the ranking

Permutation importance measures a sensor's **marginal** contribution. A sensor
near zero may carry no information, **or may merely be redundant** given the
sensors around it. **Permutation importance cannot distinguish the two on its
own.** This is why the study also runs forward selection and reports the sensor
correlation matrices.

Measured on the synthetic fixtures, where the generator decides which channels
carry signal. `mic` is injected with *weak* copies of every fault signature, so
it genuinely carries information, yet:

| Fixture | `mic` importance | Cause |
|---|---|---|
| Ceiling (full-amplitude signatures) | exactly **+0.0000** | The strong channels already separate the classes perfectly, so dropping `mic` costs nothing. Redundancy, not absence of information |
| Hard (`amplitude_scale = 0.20`) | **−0.0195 to +0.0160** across splits | Weak signature against a σ = 1 noise floor; the marginal contribution is within noise of zero |

On both fixtures `mic` is therefore indistinguishable from the two channels that
carry *only* noise, `ub_ax` and `ob_rad`. The four strongly injected channels
(`ub_rad`, `ub_tan`, `ob_tan`, `ob_ax`) outrank the noise channels in all 10
splits, with a worst margin of +0.0165.

**Consequence for reporting:** no generated text, file name, axis label or
caption anywhere in this repository describes a low-importance sensor as
uninformative or useless. The neutral phrase "low marginal contribution" is used
instead, and a test enforces it across every generated output.

### 8.3 Forward selection

Start empty; at each step add the sensor whose addition gives the highest
inner-validation macro-F1. Candidates are tried in `SENSORS` order and a later
candidate must be strictly better to displace an earlier one, so ties break
deterministically. Continue until all 8 are added.

### 8.4 Selected subsets are never refit

The model and the seed are fixed, so a selected subset's test score **is** its
entry in the exhaustive results and is looked up rather than recomputed. This
makes it impossible for a selected score to disagree with its exhaustive
counterpart. A Level D test confirms a lookup equals an explicit refit exactly.

---

## 9. Exhaustive evaluation

All 255 non-empty subsets of the 8 sensors are evaluated on every split:
10 splits × 255 subsets = **2,550 score rows** for a full run. For each subset,
`make_model()` is fit on the full outer training fold using only that subset's
columns and predicts the test fold.

Results are written per split, so an interrupted run resumes where it stopped.
`--quick` restricts the run to the first split at 50 trees and writes to
`results/quick_<split_mode>/`, so quick results can never mix with real ones. The
quick restriction is applied *before* the resume check, so repeating a quick run
is a no-op rather than silently computing a different split.

---

## 10. Figures

| Decision | Value | Why |
|---|---|---|
| Width | 3.5 in single column, 7.16 in double | IEEE conference format |
| Font | 8 pt serif | Matches the body text |
| Colour | Series distinguished by marker and line style | Must remain readable in grayscale |
| Formats | PDF and 300 dpi PNG | Vector for the paper, raster for review |
| Captions | Generated by `04_analyze.py` into `figures/captions.tex` | A caption that states the aggregation must match what the code actually did |

**Fig. 1 whiskers span the full minimum-to-maximum range** (`whis=(0, 100)`), not
1.5 × IQR, and no points are hidden as outliers. Every subset is enumerated, so
this is a complete population rather than a sample; the 1.5 × IQR convention and
hidden outliers would conceal the worst-case subsets, which are exactly what a
sensor-reduction study cares about.

**Fig. 3** does not offset or jitter overlapping lines. Several classes may sit
at recall 1.0 where their lines coincide exactly; the caption says so instead,
because nudging a line would misreport its value.

**Fig. 5** uses repeat 0 only, so every window is counted exactly once.

### 10.1 Correlation-matrix scaling

Both correlation figures **mask the diagonal** and take their colour limits from
the **off-diagonal** data: `vmax = max|ρ|` over the off-diagonal entries, rounded
up to the next 0.05, with `vmin = -vmax`.

**Why.** The diagonal is 1.0 by definition and carries no information, but on a
fixed −1…+1 scale it dominates the colour range and flattens the real
off-diagonal structure into indistinguishable grey. On the synthetic fixture the
genuine structure spans roughly −0.14 to +0.35, which is invisible against a unit
diagonal.

Two matrices are produced:

- **Fig. 6** — Spearman ρ of the **RMS** feature between sensors.
- **Fig. 6b** — **mean |Spearman ρ| over all 18 matched features** per sensor
  pair. A broader redundancy measure than RMS alone, which is a single amplitude
  summary.

Fig. 6 is a **signed** quantity and keeps the symmetric diverging scale about zero. Fig. 6b is **non-negative**, so it uses a sequential `0…vmax` scale with a sequential colormap: a symmetric scale would imply a meaningful zero crossing that mean |ρ| does not have, and would waste half the colour range.

Both are **descriptive only** and enter no hypothesis test. Which one appears in
the paper is chosen after seeing both on real data; that choice affects no
result.

---

## 11. Validation

| Level | What it checks |
|---|---|
| A | Feature values against analytic signals; no duplicate features |
| B | Real data: 1,951 files, shapes, labels, speed range (skipped when absent) |
| C | Leakage: no shared recordings or speed bands, inner splits inside the training fold, guards fire |
| D | Integration: determinism, resume, completeness, lookup-equals-refit |
| E | Sanity: shuffled labels at chance, dummy baseline, importance ranking, confusion-matrix totals |
| F | Result validation inside `04_analyze.py`: row counts, metric ranges, full-configuration agreement, p-value ranges, input provenance |

Level F runs **before** anything is written, so a failed check aborts with a clear
message rather than leaving a half-written set of tables and figures behind.

---

## 12. Traceability

`01_extract_features.py`, `03_run_experiments.py` and `04_analyze.py` each write
a `run_metadata.json` next to their outputs, recording the UTC timestamp, Git
commit and dirty flag, Python and library versions, split mode, seed, input
counts and the command line used. `04_analyze.py` additionally records the
metadata of the inputs it consumed, and warns loudly if the feature table and the
results carry different commit hashes, or if either was produced from a dirty
working tree.

Every number in the paper comes from a file under `results/` or `tables/`, and
every one of those files can be traced to the commit that produced it.

---

## 13. What this protocol deliberately excludes

- No hyper-parameter tuning, no model selection, no feature scaling.
- No alternative classifiers and no deep learning.
- No constant, feature, split, threshold or model setting changed to improve a
  score.
- No ranking or selection using data from a test fold.
- No random window splitting for any reported result.
- No hand-edited numbers anywhere: every table and figure regenerates from
  `results/` with one command.
