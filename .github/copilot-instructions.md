# FPGuard AI project instructions

## Architecture

- Keep the Streamlit interface in `app/streamlit_app.py`.
- Keep reusable Python code in the `src/fpguard/` package. The package owns
  data loading, preprocessing, training, threshold selection, evaluation, and
  prediction; the UI should call these modules rather than reimplementing them.
- `src/fpguard/real_data.py` provides local-only provenance and quality audits
  for explicitly supplied CSV files. It does not create splits or train models.
- Keep small, explicitly approved demo fixtures in `data/sample/`. Put original
  datasets in `data/raw/`; raw data and generated artifacts must not be
  committed.
- Put automated checks in `tests/`. Keep generated models and other runtime
  outputs in `artifacts/`.

## Coding standards

- Use type hints, focused functions, clear names, and actionable exceptions.
- Validate input files and schemas before processing. Never silently discard
  rows, columns, or features; make any intentional cleaning explicit and
  measurable.
- Keep transformations inside the fitted scikit-learn pipeline so imputers and
  other fitted preprocessing state are learned from training data only.
- Do not add dependencies, services, or infrastructure without a clear MVP
  requirement. Record runtime and test dependencies in `requirements.txt`.
- Keep the Streamlit layer thin and avoid reporting unmeasured model claims.

## Windows commands

Run these commands in PowerShell from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m fpguard.data
.\.venv\Scripts\python.exe -m fpguard.train
.\.venv\Scripts\python.exe -m fpguard.evaluate
.\.venv\Scripts\python.exe -m fpguard.evaluate --threshold-experiment --minimum-recall 0.80
.\.venv\Scripts\python.exe -m fpguard.compare_models --minimum-recall 0.80
.\.venv\Scripts\python.exe -m fpguard.real_data --help
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

The editable install makes `src/fpguard` importable to the application. Pytest
also has the `src` directory on its import path when run from the repository
root.

## Testing expectations

- Add focused tests for schema validation, preprocessing behavior, split
  boundaries, threshold selection, evaluation, and prediction as those features
  are implemented.
- Test edge cases and failures, not only the happy path. Keep demo fixtures
  small and label their origin and limitations.
- Before claiming a model improvement, run the same untouched test data through
  both strategies and report the measured metrics and threshold.

## Data leakage and evaluation rules

- The Stage 2 synthetic experiment is generated locally by
  `python -m fpguard.data`; it must not download external data. Persist and
  validate the exact dataset and split indices used by training/evaluation.
- Split data into training, validation, and test sets before fitting any
  preprocessing or model components. Fit preprocessing and the classifier on
  training data only. Validation may be used for diagnostics and threshold
  selection; test data remains untouched until final evaluation.
- Use validation labels to select a probability threshold subject to a stated
  minimum recall constraint. Never use test labels to choose a threshold,
  features, model settings, or preprocessing behavior.
- Model comparison and threshold selection must use validation results only.
  Diagnose test errors only after selection is frozen, and do not feed diagnosis
  or test metrics back into model/threshold selection.
- Evaluate the selected strategy exactly once on an untouched test set and
  compare it with the baseline on that same test set. Threshold adjustment is
  not guaranteed to improve performance; state results honestly.
- Exclude identifiers, post-event information, and target-derived features from
  predictive inputs. Document feature exclusions and their rationale.
- The real-data audit must reject unmapped label values and report possible
  identifiers, timestamps, attack categories, and target-derived fields for
  human review. An audit report is not proof that a dataset is leakage-free.
- Do not download large datasets automatically. Never commit raw datasets or
  real sensitive traffic data.
