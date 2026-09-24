#!/usr/bin/env bash
#
# The exact command sequence for the real MaFaulDa study.
#
# THIS IS DOCUMENTATION, NOT AUTOMATION.
#
# It is deliberately not runnable end to end. Every STOP below is a decision
# that a person makes by reading output, and automating past one of them would
# hide the decision rather than record it. Run the commands one block at a
# time and read what each prints before moving on.
#
# `set -e` is intentionally absent for the same reason: a non-zero exit here is
# something to read and think about, not something to abort a pipeline on.
#
# Every step writes a run_metadata.json next to its outputs, recording the
# commit, the library versions and the command line, so the provenance of every
# number is recoverable afterwards.

PY=".venv/Scripts/python.exe"      # Windows
# PY=".venv/bin/python"            # Linux / macOS

SPLIT="recording"                  # set to "speed" only after step (d)

# ---------------------------------------------------------------------------
# (a) Place the dataset
# ---------------------------------------------------------------------------
# Unzip MaFaulDa so the six class folders sit directly under data/mafaulda/:
#   data/mafaulda/{normal,imbalance,horizontal-misalignment,
#                  vertical-misalignment,underhang,overhang}/
#
# The pipeline never downloads it, and data/ is excluded from Git.
#
# Confirm the protocol tag is the code you are about to run:
#
#   git describe --tags
#
# It must report the protocol tag with no trailing commit suffix. If it does
# not, you are not running the frozen protocol.

# ---------------------------------------------------------------------------
# (b) Level B: validate the raw dataset          >>> STOP AND REPORT <<<
# ---------------------------------------------------------------------------
# Checks the file count against 1,951, the shape of a seeded 50-file sample,
# and the class folders. -s shows the printed class-count table.
#
# Report: class counts and the total against the 1,951 expected.

$PY -m pytest -m realdata -s

# >>> STOP. Do not continue until the counts have been read. <<<

# ---------------------------------------------------------------------------
# (c) Extract features                            >>> STOP AND REPORT <<<
# ---------------------------------------------------------------------------
# Expect roughly 30-60 minutes on all cores. Writes features/features.csv and
# features/run_metadata.json.
#
# Expected shape: 9,755 rows x 149 columns (1,951 recordings x 5 windows).
#
# Report: the feature table shape, the speed range, and any warnings. A row
# count warning fires per file that is not 250,000 rows.

$PY 01_extract_features.py

# Re-run the feature-level Level B tests now that features.csv exists:
#   NaN/infinite check, labels, 5 windows per recording, speed range.
$PY -m pytest -m realdata -s

# >>> STOP. Do not continue until the shape and speed range have been read. <<<

# ---------------------------------------------------------------------------
# (d) Checkpoint, BOTH split modes                >>> STOP AND REPORT <<<
# ---------------------------------------------------------------------------
# The go/no-go gate. Each run reports:
#   0. class coverage per speed band and per test fold
#   1. the leakage check: naive random-window split vs the grouped split,
#      and the inflation between them
#   2. the full-sensor baseline
#   3. all 8 single-sensor scores
#   4. the shuffled-label and most-frequent-dummy sanity baselines
#   5. the decision: OK / WARNING / CEILING EFFECT, plus a warning if the
#      tachometer is the single best sensor
#
# Speed mode may refuse to split if no quantile banding puts all 6 classes in
# every test fold. That is a defined outcome, not a crash; see PROTOCOL.md 5.1.

$PY 02_checkpoint.py --split recording
$PY 02_checkpoint.py --split speed

# >>> STOP. The split mode is a human decision. <<<
#
# Choose from the evidence above and set SPLIT accordingly. If CEILING EFFECT
# fires, or the tachometer leads the single-sensor ranking, that is a decision
# to be made deliberately, not worked around.

# ---------------------------------------------------------------------------
# (e) Dry run, then the full experiment           >>> STOP AND REPORT <<<
# ---------------------------------------------------------------------------
# One split at the full 200-tree configuration, reporting measured wall time,
# peak memory and the projected total for all 10 splits. Written to
# results/dryrun_<split>/, which is never part of a study run.

$PY 03_run_experiments.py --split "$SPLIT" --dry-run

# >>> STOP. Approve the projection before committing to the full run. <<<

# The full run: 10 splits x 255 subsets = 2,550 score rows. Hours. It logs the
# elapsed time per split and a running estimate of the time remaining, and
# resumes from where it stopped if interrupted, so rerunning the same command
# after a failure is safe and skips completed splits.

$PY 03_run_experiments.py --split "$SPLIT"

# ---------------------------------------------------------------------------
# (f) Analysis                                    >>> REPORT VERDICTS <<<
# ---------------------------------------------------------------------------
# --expect-full-run makes Level F demand the complete 10-split, 2,550-row
# result set rather than accepting whatever is present. Level F runs before
# anything is written, so a failure leaves no partial output behind.
#
# Writes tables/, figures/ and figures/captions.tex.
#
# Report the H1, H2 and H3 verdicts factually, from tables/hypotheses.txt.
# The oracle column and the oracle k are an upper bound selected on test data;
# the fair, permutation-selected result is the answer.

$PY 04_analyze.py --split "$SPLIT" --expect-full-run

# ---------------------------------------------------------------------------
# Afterwards
# ---------------------------------------------------------------------------
# Confirm provenance: every run_metadata.json should carry the same commit and
# report git_dirty = false.
#
#   cat features/run_metadata.json
#   cat "results/$SPLIT/run_metadata.json"
#   cat tables/run_metadata.json
#
# Nothing under data/, features/, results/, figures/ or tables/ is committed.
