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

**Commit hashes changed on 2026-10-31.** Before the repository was made public,
the author, committer and tagger e-mail addresses were replaced with the
author's GitHub `users.noreply.github.com` address. Changing an identity changes
the commit hash, so every commit hash in the published history differs from the
one recorded in this document and in the `run_metadata.json` files. **No tree,
commit message, author date or committer date changed**, and no code, constant,
threshold or result differs as a result. The hashes recorded here were left
untouched — §12.1(a) quotes real log output, and the metadata files are written
by the pipeline, never by hand — so
[docs/COMMIT_MAP.md](docs/COMMIT_MAP.md) maps every pre-rewrite hash to its
published counterpart. Both protocol tags were recreated on the rewritten
commits with byte-identical messages and unchanged dates.

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
| Channels extracted | 8, fixed order `tach, ub_ax, ub_rad, ub_tan, ob_ax, ob_rad, ob_tan, mic` (`ALL_CHANNELS`) | The CSV column order |
| **Sensors studied** | **7: `ALL_CHANNELS` minus `tach`** (`SENSORS`) | The tachometer is excluded; see section 2.2 |
| Subsets | **127** non-empty subsets of 7 sensors | 2^7 - 1 |
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


### 2.2 The tachometer is excluded from the sensor set

**Decided 2026-09-29, AFTER data processing began and BEFORE any model was
fitted or any score consulted.** No classifier had been run on the real data at
the time of this decision; it rests entirely on the distributional evidence
below and on what the tachometer is for.

**The study evaluates 7 sensors and 127 non-empty subsets.** The tachometer
channel is still extracted and its 18 features remain in `features.csv`; the
exclusion is at the sensor-set level, not at extraction.

**Why.**

1. **The channel carries a recording artefact, not a fault signature.** On 67
   recordings the tachometer trace separates from non-ball-fault windows at
   Cohen's *d* up to **6.6**, and from OTHER `ball_fault` recordings at *d* up
   to **4.5**. A genuine fault signature would not separate a condition from
   other recordings of that same condition.
2. **The artefact is label-correlated.** All 67 affected recordings are labelled
   `underhang` (59, 10.6% of that class) or `overhang` (8, 1.6%); no other class
   is touched. Any model including `tach` can use it as a shortcut that has
   nothing to do with machine physics.
3. **RQ3 asks which sensors carry diagnostic information.** A ranking
   contaminated by an artefact would answer that question falsely, and the
   ranking is a headline result of this study.
4. **A partial fix is not defensible.** Dropping the 67 recordings, or dropping
   the affected features, would remove only the portion we happened to detect.
   We could not demonstrate that the artefact was gone rather than merely
   hidden below our detection threshold.
5. **The tachometer is instrumentation, not condition monitoring.** It measures
   shaft speed. Excluding it makes this a cleaner test of the sensor-placement
   question an engineer actually faces: which accelerometers and microphone to
   install.

**Measured evidence.** Three-way comparison, A = the 67 affected recordings
(335 windows), B = `ball_fault` recordings with a usable tachometer reading
(1,280 windows), C = all non-`ball_fault` windows (8,140):

| Tach feature | *d*(A vs C) | *d*(B vs C) | *d*(A vs B) |
|---|---|---|---|
| `band_1000_2000` | **6.584** | −0.358 | 3.856 |
| `band_5_250` | −5.386 | 0.310 | −3.661 |
| `band_10000_25000` | 5.332 | −0.430 | 3.809 |
| `rms` | 3.927 | −0.361 | 4.542 |
| `skewness` | −3.818 | 0.341 | −4.436 |

The tachometer does **not** separate `ball_fault` in general — every B-vs-C
effect is |*d*| ≤ 0.43. The effect lives entirely in the 67 affected
recordings. `scripts/diagnose_speed.py` reproduces these numbers.

**What this costs.** The full configuration is now 7 sensors rather than 8, and
the exhaustive sweep is 127 subsets per split rather than 255, so 10 × 127 =
**1,270 score rows** for a full run.

### 2.3 The secondary 8-sensor analysis, and what it showed

The deferred comparison has been **run**: the whole pipeline repeated with the
tachometer included, 8 sensors and 255 subsets, speed mode only, 10 splits and
2,550 rows. Outputs are under `results/`, `tables/` and
`figures/secondary_8sensor_speed/`, kept entirely separate from the primary
results.

**The exclusion changed no conclusion.** This is recorded because it establishes
that the decision in section 2.2 was **not result-driven**: it did not produce a
more favourable outcome. It does NOT establish that the exclusion was merely
precautionary — section 2.5 shows the channel carries a learnable shortcut, and
supersedes that earlier reading.

| | 7 sensors (primary) | 8 sensors (secondary) |
|---|---|---|
| H1 primary verdict | SUPPORTED, 0 violations | SUPPORTED, 0 violations |
| H2 fair *k* | 4 | 4 |
| H2 oracle *k* | 3 | 3 |
| Full configuration | 0.9917 ± 0.0066 | 0.9917 ± 0.0080 |
| Oracle score at every *k* | identical | identical |
| Permutation-selected curve | identical to 3 d.p. at every *k* | identical |

The oracle subsets are the same at every *k* in both configurations: the
tachometer never enters one. Had the exclusion been doing the work of producing
a favourable result, these would differ.

**Where the tachometer ranks.** Across the 10 splits it is last of 8 in
permutation importance in 8 splits and 7th of 8 in the other 2 (mean position
7.8, sd 0.40), with mean importance **+0.0002 ± 0.0003** — two orders of
magnitude below the next-lowest sensor (`ob_tan`, +0.0023). In forward selection
it lands at positions 4 to 7 (mean 5.6, sd 1.20).

### 2.4 The artefact is learnable, which marginal importance does not show

Near-zero marginal importance establishes only that the model did not **need**
the tachometer when seven good sensors were present. Whether its artefact is
learnable is a separate question, and it is answered by how the channel performs
**alone**. Measured from the saved predictions of the secondary run, with no
refitting:

| Sensor alone | macro-F1 (mean ± sd) |
|---|---|
| `ub_rad` | 0.8799 ± 0.1030 |
| … | … |
| `ob_rad` | 0.6404 ± 0.1091 |
| **`tach`** | **0.2471 ± 0.0491** |

Chance for 6 balanced classes is about 0.167, so the tachometer alone scores
**above chance**. Its per-class recall, pooled over the 10 splits, is not
uniform:

| Sensor alone | N | IMB | HM | VM | **UB** | **OB** |
|---|---|---|---|---|---|---|
| `tach` | 0.0286 | 0.2222 | 0.1898 | 0.2352 | **0.3366** | **0.5376** |
| `ob_rad` (control) | 0.3469 | 0.6979 | 0.5817 | 0.5282 | 0.8588 | 0.8386 |

Mean recall on the two bearing classes against the other three fault classes:

| Sensor alone | underhang + overhang | IMB + HM + VM | difference |
|---|---|---|---|
| `tach` | 0.4371 | 0.2158 | **+0.2213** |
| `ob_rad` (control) | 0.8487 | 0.6026 | +0.2461 |

**Two reasons the above is NOT by itself evidence of artefact exploitation.**

1. The elevated bearing-class recall for the tachometer (+0.2213) is **not
   distinguishable from the `ob_rad` control** (+0.2461). A control sensor with
   no known artefact shows the same direction and a slightly larger gap.
2. The confusion matrix shows **prediction bias rather than class skill**. The
   `OB` column absorbs mass from every row, including **0.310 of healthy
   windows**, and `underhang` and `overhang` are the two largest classes.
   Further, the artefact covers 59 of 558 `underhang` recordings (**10.6%**), so
   an artefact-driven model would show `underhang` recall near 0.11, not the
   observed 0.337.

**Reading of this section alone, SUPERSEDED by section 2.5:** the tachometer by
itself scores 0.2471 against a shuffled-label baseline of about 0.173 — weakly
above chance — with no class-specific pattern distinguishable from a control
sensor. **This reading is wrong, and section 2.5 corrects it.** The pooled
per-class recall used here cannot separate the 670 affected windows from the
10,040 unaffected ones; splitting them shows the artefact is learnable. The
section is kept rather than deleted so the correction is visible.
`scripts/tach_learnability.py` reproduces every figure above from the stored
predictions.

### 2.5 The decisive test: was the artefact exploited?

The question section 2.4 cannot answer is whether the artefact is used. It is
answered directly by splitting the bearing-class windows into those from the 67
**affected** recordings and those from the **unaffected** ones, and comparing
accuracy. If a model exploits the artefact, it should do markedly better on the
affected windows. Computed from saved predictions with no refitting
(`scripts/artifact_exploitation.py`).

Window counts, identical for every model (two repeats, so each window appears
twice):

| affected UB/OB | unaffected UB/OB | all other classes |
|---|---|---|
| 670 | 10,040 | 8,800 |

| Model | affected | unaffected | **difference** | other classes |
|---|---|---|---|---|
| **`tach` alone** | **0.8239** | **0.4068** | **+0.4171** | 0.2086 |
| `ob_rad` alone (control) | 1.0000 | 0.8390 | +0.1610 | 0.5943 |
| all 8 sensors (tach included) | 1.0000 | 0.9984 | +0.0016 | 0.9902 |
| all 7 sensors (tach excluded) | 1.0000 | 0.9972 | +0.0028 | 0.9911 |

**The tachometer-alone model classifies affected windows far better than
unaffected ones**: 0.8239 against 0.4068, a gap of +0.4171. The control sensor's
gap is +0.1610, about one third as large. By the stated criterion — that a
learnable artefact should show far higher accuracy on affected windows — the
artefact **is** learnable and **is** used by a model that has only that channel.

Two qualifications on the comparison, stated because they bear on how far it
can be pushed:

- The control's affected-window accuracy is **1.0000**, at ceiling, so its gap
  is compressed from above and the two gaps are not measured on equal footing.
- The full 8-sensor and 7-sensor models show **no meaningful gap** (+0.0016 and
  +0.0028), both at or near 1.0000 on affected and unaffected windows alike.
  With the other sensors present, affected windows are not classified
  differently from unaffected ones.

**This corrects the reading in section 2.4.** The artefact **is learnable**:
the tachometer alone reaches 0.8239 accuracy on affected underhang/overhang
windows against 0.4068 on unaffected ones, a gap 2.6 times the control's. The
earlier reading — prediction bias, no class-specific pattern — was drawn from
**pooled** per-class recall, which cannot separate the 670 affected windows from
the 10,040 unaffected ones; pooling averages the two together and hides exactly
the contrast that matters. The exclusion in section 2.2 was therefore **not
merely precautionary: it removed a learnable shortcut.** Equally important, and
inseparable from that finding: with the other seven sensors present the gap
disappears (+0.0016 for 8 sensors, +0.0028 for 7, both at or near 1.0000 on
affected and unaffected windows alike), so the artefact **did not distort the
full-configuration results** — which is why the primary and supplementary
analyses agree on every verdict. One caveat on the comparison: the control's
affected-window accuracy is 1.0000, at ceiling, so its gap is compressed from
above and the two gaps are not measured on equal footing.

A note on method, since it cost a round of analysis: a pooled per-class summary
could not answer this question, and only the affected/unaffected split could.
Where a suspected artefact is confined to an identifiable subset of recordings,
the subset has to be isolated before any comparison is informative.

**This is why the exclusion stands even though it changed no conclusion here.**
The channel carries learnable, label-correlated structure; that it happened not
to alter a verdict on this dataset is not a guarantee it would not on another.

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
| `speed_hz` | Parsed from the recording's file name | Ground truth from the dataset authors; see section 3.1 |
| `speed_hz_tach` | Rising edges of the tachometer above `(max + min) / 2`, divided by duration | Retained for comparison only; see section 3.1 |
| `recording_id` | Assigned from the sorted file list | Makes the cross-validation groups identical on every run and machine |


### 3.1 Where `speed_hz` comes from

**Changed on 2026-09-28, AFTER data processing began.** This is the only
protocol change made after the first real-data run, and it is recorded here in
full.

`speed_hz` is **parsed from the recording's file name**, which MaFaulDa uses to
record the rotation frequency in Hz (for example `12.288.csv`). The tachometer
estimate is still computed and kept as a second column, `speed_hz_tach`, but is
no longer used by the pipeline.

**Why the change was made.** Extracting the full dataset produced tachometer
estimates spanning 11.6–320.6 Hz. The rig is specified at 700–3600 rpm, about
12–60 Hz, so 320 Hz is **physically impossible**: the estimate was wrong, not
the specification. 67 of 1,951 recordings read outside 10–65 Hz.

**The decision rests on physical grounds, not on any score.** No model was
fitted, no accuracy was consulted, and no hypothesis test was run at any point
in reaching it. The reasoning is that a shaft cannot turn at 320 Hz on a rig
built for 60 Hz.

**Nothing in the recording-split results is affected.** `speed_hz` is metadata:
it is not among the 144 model features and no model has ever seen it. The
recording-split mode groups by `recording_id` and never reads it. The only
consumer is `choose_speed_q`, which builds the speed-band grouping for the
speed-split fallback — and that mode could not have been used correctly with
the old values, which is why this had to be fixed before it could be offered.

**Secondary benefit.** On the 1,884 recordings where the tachometer estimate was
in range, it read a systematic **2.3% low** (median ratio of estimate to file
name 0.977, median difference −0.79 Hz), because a recording that ends
mid-revolution yields one fewer rising edge than it has revolutions. Reading the
file name removes that bias as well.

### 3.2 Documented limitation of the tachometer channel

On **67 of 1,951 recordings — every one of them `ball_fault`** — the tachometer
estimator overcounted rising edges, reporting up to **320.6 Hz** against a true
50.79 Hz.

The cause is in the dataset's tachometer trace on those files, not in the
method. Measured over a seeded 40-file sample from each group:

| | tach min (median) | tach max (median) | tach std (median) | DC offset |
|---|---|---|---|---|
| Affected (67) | −1.925 | 4.525 | **2.118** | ~0 |
| Unaffected (1,884) | −1.376 | 4.947 | **1.742** | ~0 |

The affected traces swing about **22% wider** (higher standard deviation), with
a more negative minimum and a lower maximum. Because the estimator's threshold
is the midpoint `(max + min) / 2`, that pulls the threshold down from ~1.97 to
**0.80–1.27**, into the noise band, where the trace crosses it repeatedly within
a single revolution.

Evidence that the signal itself is fine: an FFT of the same tachometer channel
recovers the correct rotation frequency to within ~1.3 Hz on all three affected
recordings examined. The rising-edge intervals become strongly bimodal on those
files (one had a 1.04 ms median against a 14.38 ms 90th percentile, a 14-fold
spread) whereas unaffected files are near-constant to within about 1%. The ratio
of estimate to true speed clusters on small integers — 2×:23, 3×:17, 4×:11,
5×:8, 6×:4 — consistent with several counted crossings per revolution.

`scripts/diagnose_speed.py` reproduces every number above and is kept in the
repository as the evidence for this limitation.

**This is a property of the dataset's tachometer channel on those files.** It is
reported as a limitation rather than corrected, and the estimator is left
unchanged so that `speed_hz_tach` continues to record what it actually measures.

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
| Grouping rationale | Never split a unit of measurement | The five windows of one recording are near-duplicates; splitting them across the boundary leaks |
| **Primary grouping** | **Speed band** | Holds out whole rotation speeds, so the model is tested on speeds it never saw; see section 5.2 |
| Secondary grouping | `recording_id` | Reported as a robustness analysis; see section 5.2 |
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


### 5.2 Which split mode is primary

**Decided 2026-09-30, after the checkpoint and before any experiment was run.**

**`speed` is the PRIMARY protocol. `recording` is reported as a secondary
robustness analysis.** Both are run in full and both are reported. Neither is
selected on the basis of which produced a more interesting result, and this
decision was recorded before `03_run_experiments.py` was run in either mode.

The measured checkpoint figures behind the decision:

| | recording | speed |
|---|---|---|
| Full configuration (7 sensors) | 0.9971 ± 0.0038 | 0.9917 ± 0.0066 |
| Best single sensor | `ub_rad` 0.9770 | `ub_rad` 0.8799 |
| **Dynamic range (full − best single)** | **0.0201** | **0.1119** |
| Naive-split inflation | +0.0027 | +0.0081 |

**Why `speed` is primary.**

1. **Dynamic range.** Recording mode spans only 0.020 macro-F1 between the best
   single sensor and all seven — barely twice the 0.01 comparability threshold.
   The sensor-count curve has almost no room to move, so RQ1 cannot be answered
   informatively under it. Speed mode spans 0.112, five and a half times wider.
2. **External validity.** Holding out whole rotation-speed bands tests
   generalisation to unseen operating speeds, which is the realistic deployment
   condition for variable-speed machinery. Holding out recordings tests only
   generalisation to unseen recordings at speeds already observed.
3. **Leakage sensitivity.** The naive-split inflation is three times larger
   under speed grouping (+0.0081 against +0.0027), so speed mode is the more
   demanding evaluation.

**The formal `CEILING EFFECT` rule did not fire in either mode.** It requires
both a full score at or above 0.99 AND a gap to the best single sensor within
`TOLERANCE`. Both modes met the first condition; neither met the second
(0.0201 and 0.1119 against a 0.01 threshold), so the rule returned `OK` for
both. **This decision is therefore made on dynamic range, which that rule does
not capture.** The rule is left unchanged: it is a guard against a degenerate
study, not a split-mode selector.

Both modes use the same speed banding, which `choose_speed_q` accepted at the
first candidate: `q=10`, ten bands, every band carrying all six classes and
every test fold carrying all six.

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

**Why.** The oracle at each *k* is a maximum over C(7,*k*) candidate subsets: 7,
21, 35, 35, 21, 7, 1 for *k* = 1…7. A maximum over more candidates carries
more upward bias, so the oracle curve is inflated most at middle *k*, peaking at the middle *k*, purely from maximisation and independently of any
sensor effect. An apparent H1 violation at middle *k* can therefore be an
artefact of that bias rather than evidence of a redundant or harmful sensor. The
fair curves involve no test-set maximisation and so carry no such bias.

The main table includes a **Candidates C(7,k)** column so a reader can see the
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
every sensor*, so the per-sensor drops are paired. The ranking comes from
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
deterministically. Continue until all 7 are added.

### 8.4 Selected subsets are never refit

The model and the seed are fixed, so a selected subset's test score **is** its
entry in the exhaustive results and is looked up rather than recomputed. This
makes it impossible for a selected score to disagree with its exhaustive
counterpart. A Level D test confirms a lookup equals an explicit refit exactly.

---

## 9. Exhaustive evaluation

All 127 non-empty subsets of the 7 sensors are evaluated on every split:
10 splits × 127 subsets = **1,270 score rows** for a full run. The
tachometer is excluded from the sensor set; see section 2.2. For each subset,
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

The commit hashes recorded in those metadata files, and quoted throughout §12.1
below, are the ones that existed when each run happened. They predate the
2026-10-31 identity rewrite described at the top of this document and do not
resolve in the published history. [docs/COMMIT_MAP.md](docs/COMMIT_MAP.md)
translates each one; the trees they name are unchanged, so every provenance
reference below remains checkable.

---


## 12.1 Record of the real run

Three things happened during the real run that a reader should know about. All
three are recorded here factually; none changed a result.

### (a) The Level F provenance warning fired

Both analyses reported:

```
Level F: INPUTS COME FROM DIFFERENT COMMITS
         (features=af5b0605e7aa, results=1f82015dc875)
```

The warning is correct: the feature table was extracted at `af5b0605` and the
experiments were run at `1f82015d`. Two commits separate them, `ade6d23` and
`1f82015`.

**Verified difference.** `common.py` and `01_extract_features.py` are
**byte-identical** across that range (`git diff --quiet` reports no change for
either). `03_run_experiments.py` differs by **five lines, all inside
docstrings** — "255 of them for 8 sensors" became "127 … 7 studied sensors", and
"all 8 sensors" became "every sensor". **No executable statement changed.** The
remaining differences are in `02_checkpoint.py`, `04_analyze.py`, documentation,
the run script and tests.

The feature table is therefore exactly what the current code produces. The
warning is left in place and was not suppressed: it did its job, and the
verification above is the answer to it rather than a reason to weaken the check.

### (b) The speed run was terminated and resumed

The first `03_run_experiments.py --split speed` invocation exited with code 4
after completing **8 of 10 splits**, during `r1_f3`, with no Python traceback.
The process was terminated externally rather than raising. **The cause is not
known.**

The run was resumed with the same command. Resume skipped the 8 completed
splits and computed the remaining 2, as designed.

**Determinism check.** To confirm the termination left no damage, the
pre-termination split `r0_f0` was recomputed from scratch into a temporary
directory and compared against the stored files:

| File | Result |
|---|---|
| `scores_r0_f0.csv` | **byte-identical** (md5 `525d01bd370efabc79e592e278f4ed6a`) |
| `selection_r0_f0.json` | **byte-identical** |
| `preds_r0_f0.npz` | **byte-identical** |

The stored results were not overwritten by the check. The recording-mode run
completed in one invocation with no interruption.

### (c) A benign scikit-learn warning

497 instances of one warning appeared across the two runs:

```
sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed`
should be used with `sklearn.utils.parallel.Parallel` to make it possible to
propagate the scikit-learn configuration of the current thread to the joblib
workers.
```

It originates **inside scikit-learn**, from `RandomForestClassifier` fitting
with `n_jobs=-1`; no code in this repository calls `sklearn.utils.parallel`. It
concerns the propagation of scikit-learn's thread-local configuration to joblib
workers. This study sets no such configuration, so nothing is lost. It is
recorded as benign and was **not** suppressed, since Section 14 forbids
silencing warnings globally.

### (d) The dirty-tree warning on the secondary run

The secondary run's metadata records `git_dirty: true`. Verified the same way as
(a): at run time HEAD was `edb0cc6`, and the working tree differed from it
**only by untracked files, all of them the run's own outputs** under `results/`,
`tables/` and `figures/`. `git diff edb0cc6` over tracked files is empty, and
`common.py`, `03_run_experiments.py` and `04_analyze.py` are each byte-identical
to that commit.

The flag fired because the `.gitignore` change in `edb0cc6` made those output
directories tracked, so **any** run in progress necessarily dirties the tree
while it writes. No code difference is involved. The check is left as it is: a
dirty tree is worth flagging, and the verification is the answer to it.

No other warning appeared in any run or analysis log. No fold in either mode
reported fewer than 6 classes, and no Level F check failed.

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
