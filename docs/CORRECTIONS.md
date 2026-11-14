# Corrections and verification — 10 November 2026

## Applied to this source copy

- One preparation pipeline shared by both converter entry points.
- Actual frame-ID matching, strict MOT parsing, finite numbers, valid boxes and
  no silent omission of malformed annotations.
- Deterministic sequence-level 27/5 split for 32 TrainReal sequences; separate
  TestReal sequences and explicit leakage checks across all selected splits.
- New output directories only; raw images, existing data and checkpoints are
  preserved. Optional hardlinks are documented and never edited by the code.
- A generated absolute-path dataset YAML, annotation and label hashes, sequence
  lists and a manifest that subsequent validation checks.
- Missing annotated frames fail unless explicitly acknowledged and are recorded.
- Validation fails for missing/empty data, absent split definitions, mismatched
  image/label pairs, invalid classes, zero-size/nonfinite/out-of-image boxes,
  missing class coverage, leakage and modified manifests.
- Retired destructive image-level splitter. Training, validation and evaluation
  read the same chosen dataset YAML; default root is data/processed_v2.
- Explicit CPU/XPU selection, fine-tuned inference defaults, streaming sequence
  results, Windows training guard, pinned tested package versions and backend
  requirements. Optional Ultralytics analytics are disabled by CLI entry points.
- Explicit fresh training/resume. Corrected runs record dataset provenance;
  resume with another dataset or an unproven legacy checkpoint is rejected.
- Held-out evaluation records checkpoint/data hashes, class metrics, timing and
  separate AP and confusion-matrix thresholds. A two-class checkpoint is required.
- Historical progress report retained unchanged; new instructions are in README.

## Evidence from the original local project

The read-only deterministic spot audit checked 192 exported train/validation
frames (first/middle/last per sequence/split). All sampled image hashes matched
the raw same-frame image, and all sampled labels matched the same-frame CSV
boxes within 1.1e-6 normalized coordinate tolerance. There were zero mismatches.
This does not prove every annotation is correct. The old converter contains a
list-indexing defect, but that defect was not observed in the sampled exports.

The current train and validation folders each contain all 32 sequence IDs, with
32 overlapping. This differs from the progress report's documented 27/5 design.
These findings establish current contents; a report alone cannot establish an
older dataset's exact state or a checkpoint's complete training provenance.

The original installed validator passes its default AP confidence floor of
0.001 into the confusion matrix. This explains how the shared matrix can include
many low-confidence false positives while PR/AP remains high. The new evaluation
uses a separate matrix confidence (default 0.25) and match IoU (default 0.50).

## Checks completed

- Unit/regression suite: see verification.log for the exact final test count.
- Real-data image/annotation spot audit: 192 checks, zero mismatches; complete
  observations are in original_label_audit.json.
- Existing project sources and training checkpoints were not edited or replaced.

## Not completed or claimed

This session's approval settings rejected in-place edits to the original project
directory, so changes are delivered as a source-only correction package.

A full real-data rebuild was interrupted by automatic approval review over
telemetry to a Visual Studio endpoint with an unverified payload. Approval of
the endpoint did not satisfy the automatic review's payload requirements.
Its partial output has no manifest and is not a valid prepared dataset.

Full held-out evaluation did not complete. No new TestReal AP, precision or
recall is reported here. Existing weights were not retrained, fixed, or certified
for internet imagery or field deployment. The old trained model may be tested
as a frozen checkpoint on TestReal, but a new sequence-split validation claim
requires fresh training from weights that have not seen those validation sequences.

## Next local decisions

Apply the source files using INSTALL_CORRECTIONS.md. Review the raw dataset,
complete preparation into a new directory, validate it, then decide when to
train a fresh model. Keep the old checkpoint/results for comparison. Class
imbalance remedies should be selected only after per-class held-out evidence;
the sampled audit did not find reversed class labels.
