# Repository setup validation — 2026-10-15

Executed locally against this code handoff:

- `python -m unittest discover -v`: **23 tests passed** (19 inherited guards, 4 repository-path regressions).
- `python -m compileall -q .`: passed.
- JSON parsing, Python syntax and source-only publish selection: passed.
- `git diff --check`: passed.
- Ignore-rule checks for source data, images, fitted artifacts, archives, `.env` and `kaggle.json`: passed.

Observed environment: pandas 2.2.3, NumPy 2.3.5, scikit-learn 1.8.0, Pillow 12.3.0 and SciPy 1.17.0. `requirements.txt` retains the earlier compatible ranges; this is not a complete lockfile or proof of compatibility with every allowed version.

The Pawpularity tests use synthetic images and rows. The new combined-wrapper path test mocks stage execution to check dependency selection; it is not a full data run. No GB-scale image pipeline, model training, Kaggle download, frontend or container test was performed during repository setup. Real-data aggregate findings under `prior_run/` come from the previously delivered data package.
