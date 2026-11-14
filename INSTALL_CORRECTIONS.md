# Applying this source-only correction package

The original project folder was readable but this session's write approval
settings rejected edits there. This package therefore contains a corrected
source copy, not an in-place modification of your original repository.

1. Back up the original `src`, `tests`, `configs`, `docs`, README and requirements.
2. Extract/merge this package into the original project root and replace the
   corresponding source/configuration files. No data, weights, virtual
   environments or training runs are included in this package.
3. Keep the old `data/processed` and trained checkpoints unchanged for comparison.
4. Use a fresh compatible environment as described in README.
5. Select the generated YAML for the corrected dataset, validate it, and start
   fresh training from the original COCO weights when you choose to retrain.

A real-data preparation attempt was interrupted at
`C:/workspace/wildlife-corrected-data-20261110`. It has no completed manifest
or generated dataset YAML and must NOT be used for training. The builder
deliberately refuses to overwrite partial/existing outputs. When proceeding
with preparation, choose a new output directory and inspect its final manifest.

Do not copy the old `.venv` into a new environment or commit data/model files.
No Git commit, push or automatic retraining was performed.
