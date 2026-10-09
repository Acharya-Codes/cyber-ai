# FPGuard AI

FPGuard AI is an offline prototype for classifying network-flow records and
measuring false-positive behavior. Stage 2 provides a reproducible synthetic
dataset and a Random Forest baseline. The synthetic results demonstrate
engineering behavior only; they are not evidence of real-world intrusion
detection accuracy.

## Intended architecture

- `app/streamlit_app.py` — thin Streamlit user interface.
- `src/fpguard/data.py` — deterministic synthetic generation, CSV validation,
  and stratified 60/20/20 split indices.
- `src/fpguard/preprocessing.py` — numeric infinity handling and
  training-fitted median imputation.
- `src/fpguard/train.py` — bounded Random Forest training and model metadata.
- `src/fpguard/threshold.py` — validation-only candidate threshold scoring
  under a minimum attack-recall constraint.
- `src/fpguard/evaluate.py` — validation diagnostics and untouched test
  metrics, including confusion matrix, class metrics, ROC-AUC, post-hoc error
  diagnosis, and exploratory feature-group summaries.
- `src/fpguard/compare_models.py` — training-only candidate fitting,
  validation-only model/threshold selection, and frozen final test comparison.
- `src/fpguard/predict.py` — load the saved fitted pipeline and predict on
  feature batches with schema validation.
- `src/fpguard/real_data.py` — local-only CSV provenance and quality audits;
  it does not split data or train models.
- `tests/` — unit tests for implemented behavior.
- `data/sample/` — only small, explicitly approved demo fixtures.
- `data/raw/` — local source datasets; never committed.
- `artifacts/` — local generated models and outputs; never committed.

The package uses a `src` layout. Pytest is configured to include `src`; an
editable install makes `fpguard` available to the app and module commands.

## Setup on Windows

Open PowerShell in the repository root. The commands below use the existing
`.venv` and install only the dependencies declared for this project:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

Run the complete test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Reproduce the baseline experiment

Run these commands from the repository root, in order. Generation writes the
dataset and stratified row-order index manifest under the ignored `artifacts/`
directory. The CSV contains only predictive flow features and the target; the
split indices are stored separately rather than as an identifier column.
Training requires that saved dataset and records its SHA-256 hash,
split sizes, feature names, positive class, and model configuration. Evaluation
refuses a dataset that differs from the one used by training.

```powershell
.\.venv\Scripts\python.exe -m fpguard.data
.\.venv\Scripts\python.exe -m fpguard.train
.\.venv\Scripts\python.exe -m fpguard.evaluate
```

The generated data has 2,000 numeric flow rows with `benign` and `attack`
labels. No external data is downloaded. The deterministic splits are
stratified: 60% training, 20% validation, and 20% test. Preprocessing and the
Random Forest are fitted together on training rows only. Validation metrics
are diagnostics; test metrics are the final baseline report. The test set is
not used for training or model selection.

Artifacts written locally (all ignored by Git):

- `artifacts/synthetic_network_flows.csv`
- `artifacts/split_indices.json`
- `artifacts/baseline_model.joblib`
- `artifacts/baseline_metadata.json`
- `artifacts/metrics.json`
- `artifacts/threshold_experiment.json` — created by the Stage 3 command below.
- Stage 4 writes separate comparison, diagnosis, per-row and feature-diagnosis
  files, plus a separate selected-model pipeline and metadata file.

The default saved model can be loaded and called on a small feature batch:

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd; from fpguard.predict import predict_network_flows; X = pd.read_csv('artifacts/synthetic_network_flows.csv').drop(columns=['label']).head(5); labels, attack_probabilities = predict_network_flows(X); print(pd.DataFrame({'predicted_label': labels, 'attack_probability': attack_probabilities}))"
```

This stage uses the baseline threshold of `0.5` only. It does not implement
false-positive reduction. Inference timing is a single local batch measurement,
not a universal performance claim.

## Stage 3: validation-driven threshold experiment

Run the experiment against the existing model, synthetic CSV, and saved split
manifest. This command does not train a model, regenerate data, or reshuffle
partitions:

```powershell
.\.venv\Scripts\python.exe -m fpguard.evaluate --threshold-experiment --minimum-recall 0.80
```

The policy checks fixed candidate thresholds from `0.00` to `1.00` in `0.01`
increments. It selects the candidate with the lowest validation false-positive
rate among those meeting the requested recall; ties favor higher precision,
then a higher threshold. The selected threshold is frozen before the test
partition is evaluated. The report is saved to
`artifacts/threshold_experiment.json`. Validation feasibility does not
guarantee that the recall target will be met on test data.

The following is the measured **synthetic-data experiment** with minimum
validation attack recall `0.80`. Both policies used the same 400 untouched test
rows and the same saved Random Forest probabilities:

| Synthetic test policy | Threshold | FPR | FNR | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.50 | 0.2951 | 0.5179 | 0.3885 | 0.4821 | 0.4303 | 0.6425 |
| Validation-selected alternative | 0.33 | 0.5938 | 0.2232 | 0.3372 | 0.7768 | 0.4703 | 0.5100 |

The threshold met the validation target at recall `0.8036`, but test recall was
`0.7768`, below the requested `0.80`. Test FPR increased by `29.86` percentage
points (relative FPR reduction `-101.18%`) while false negatives decreased from
58 to 25. **This experiment did not reduce false positives and did not meet
the recall target on test data.** Results are synthetic-only and are not
evidence of real-world intrusion detection performance.

## Stage 4: diagnosis and model comparison

Run against the same saved dataset, fitted baseline, and split manifest:

```powershell
.\.venv\Scripts\python.exe -m fpguard.compare_models --minimum-recall 0.80
```

Stage 4 compares the existing, unfitted-again Random Forest baseline with
Logistic Regression and Extra Trees. The latter two use deterministic
scikit-learn pipelines with training-only median imputation, conversion of
infinities to missing values, and (for Logistic Regression) feature scaling.
There is no hyperparameter search. Each candidate is scored on the same
validation partition at thresholds `0.00` through `1.00` in increments of
`0.01`. A candidate is eligible only if a threshold reaches minimum validation
recall. Threshold selection prefers lower validation FPR, then higher
precision, then higher threshold. Among eligible models, selection prefers
lower validation FPR, then higher precision, higher recall, and ascending
model name. If none is eligible, no selected model is claimed.

The Stage 4 comparison is distinct from Stage 3: Stage 3 tunes only the saved
Random Forest's threshold; Stage 4 also selects among candidate models, using
validation metrics only. The selected model and threshold are frozen and
persisted before the test labels are read for final comparison. Test metrics
must never be used to revise the selected configuration.

New outputs (all ignored by Git):

- `artifacts/stage4_model_comparison.json` — candidate validation results,
  selection outcome, frozen model/threshold, and test comparison.
- `artifacts/stage4_baseline_diagnosis.json` — post-hoc baseline test metrics
  and error counts; diagnosis is not model selection.
- `artifacts/stage4_baseline_diagnostics.csv` — original row index, true and
  predicted labels, attack probability, and error type for each test row.
- `artifacts/stage4_feature_diagnostics.csv` — exploratory numeric feature
  summaries comparing false positives with true negatives and false negatives
  with true positives, only when both groups have enough samples.
- `artifacts/stage4_selected_model.joblib` and
  `artifacts/stage4_selected_model_metadata.json` — written only when a
  candidate satisfies the validation recall constraint.

Measured **synthetic-data Stage 4** validation selection chose the existing
Random Forest at threshold `0.33` (validation FPR `0.6111`, recall `0.8036`).
Extra Trees selected threshold `0.35` (validation FPR `0.6389`, recall
`0.8036`); Logistic Regression selected threshold `0.38` (validation FPR
`0.6354`, recall `0.8036`). The Random Forest remained the validation winner.
On the shared untouched 400-row test partition:

| Synthetic test policy | Threshold | FPR | FP | Recall | FN | Precision | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Existing baseline | 0.50 | 0.2951 | 85 | 0.4821 | 58 | 0.3885 | 0.4303 | 0.6425 |
| Selected Random Forest | 0.33 | 0.5938 | 171 | 0.7768 | 25 | 0.3372 | 0.4703 | 0.5100 |

FPR increased by `29.86` percentage points and false positives increased by
86. Recall improved by `29.46` points and false negatives decreased by 33, but
test recall remained below `0.80`. The relative FPR reduction was `-101.18%`;
therefore this run **did not reduce false positives**. Stage 3's fixed-baseline
threshold result is also recorded separately in the Stage 4 report for
context.

Interpret `candidate FPR - baseline FPR` as a signed percentage-point change:
a negative value is lower FPR; a positive value is higher FPR. Relative FPR
reduction is `(baseline FPR - candidate FPR) / baseline FPR * 100`; it is
reported as undefined when baseline FPR is zero. Recall and false-negative
changes should be considered alongside false-positive changes. Raising recall
often classifies more benign records as attacks, so threshold tuning alone
cannot guarantee lower FPR.

Baseline diagnosis is post-hoc analysis on the test partition, not an input to
candidate or threshold selection. Feature-group differences are descriptive
synthetic-data associations, not causal explanations. All current outputs are
synthetic experiments and are not evidence of real-world NIDS performance.

## Stage 5: local Streamlit dashboard

Launch the dashboard from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Streamlit prints the local address; the default is `http://localhost:8501`.
The dashboard reads the existing baseline and Stage 4 selected-model artifacts
and the Stage 3/4 reports. It never trains models, regenerates data, or writes
to experiment artifacts. If a report or model is missing or invalid, the
affected view reports that artifact as unavailable and gives setup guidance.

The six dashboard sections provide:

- **Overview:** saved baseline and selected-policy metrics, model status, and
  the measured synthetic test comparison, including policy-rate and confusion-
  matrix charts when those report fields are available.
- **Detection Demo:** single-flow scoring or local CSV batch prediction. The
  single-flow form starts from saved synthetic feature medians when available.
  The selected model is used only after its metadata, saved dataset, split
  manifest, and Stage 4 report have been checked for consistency; otherwise the
  dashboard reports it unavailable rather than silently using a mismatched
  artifact.
  compatible numeric feature columns are `flow_duration`,
  `total_fwd_packets`, `total_backward_packets`, `total_fwd_bytes`,
  `total_backward_bytes`, `flow_bytes_per_second`, `packet_length_mean`,
  `packet_length_std`, `syn_flag_count`, and `destination_port`. CSV headers
  may appear in any order; every required feature must be present. Optional
  `label` and recognized identifier columns are output metadata only. Other
  unexpected columns, duplicate headers, invalid nonnumeric values, empty
  files, files over 5 MB, and batches over 5,000 rows are rejected.
  Missing and infinite numeric feature values are handled by the fitted
  pipeline's training-learned imputer and preprocessing.
- **Threshold Lab:** temporary threshold exploration on the saved validation
  partition. These retrospective chart metrics do not change the frozen
  experiment threshold or any saved report.
- **Error Analysis:** bounded, filterable Stage 4 diagnostic previews and
  downloadable subsets, when those diagnostic files are present.
- **Model Comparison:** Stage 4 validation candidate summaries and its frozen
  test-set comparison.
- **About & Limitations:** explanations of the synthetic-data scope and metric
  interpretation.

The attack threshold converts an attack score into a predicted class. Recall
is the fraction of attack rows detected; false-positive rate (FPR) is the
fraction of benign rows incorrectly flagged. Lowering a threshold often
increases recall but may also increase FPR. A false positive is a benign flow
flagged as attack; a false negative is an attack flow predicted benign.
Interactive test-set tuning is intentionally not offered: the Threshold Lab
uses saved validation rows and is explicitly retrospective. All uploaded CSV
processing is local to the running application. Scores are not calibrated
guarantees, and the synthetic data and measurements do not establish
real-world network intrusion detection effectiveness.

## Stage 6A: real-data ingestion and audit

The audit CLI accepts only explicit local CSV paths. It never downloads
datasets, creates training splits, fits models, or changes synthetic data or
the Stage 1–5 experiment reports. For help (which does not require a dataset):

```powershell
.\.venv\Scripts\python.exe -m fpguard.real_data --help
```

The audit supports UTF-8 CSV files with or without a BOM and normalizes column
name case and whitespace. Duplicate or colliding normalized headers and
unmapped labels stop the audit with a clear error. The default label aliases
are `benign`/`normal` and `attack`/`malicious`. Dataset-specific labels must be
mapped explicitly; repeat the relevant option for every source value present.
For example, after manually placing an official CSE-CIC-IDS2018 CSV at the
example path:

```powershell
.\.venv\Scripts\python.exe -m fpguard.real_data `
  --output-dir .\data\audit_reports\cic_ids2018 `
  --label-column Label `
  --benign-label BENIGN `
  --attack-label "DoS attacks-Hulk" `
  --attack-label "DDoS attacks-LOIC-HTTP" `
  .\data\raw\cic_ids2018\Monday-WorkingHours.csv
```

The example maps only the two attack labels shown; include every distinct
attack label in the selected input or the audit will reject the unrecognized
values. For multiple CSVs, add each explicit file path after the options.
The command writes `real_data_audit.json` only beneath the requested
`--output-dir`. It reports source path, file size and SHA-256, row/column
counts, raw and mapped label distributions, missing values, infinities,
numeric parsing issues, constant columns, duplicate rows, identical-feature
groups with conflicting labels, and a name-level comparison with the saved
baseline feature schema when metadata is available. It also flags possible
identifiers, timestamps, attack-category columns, and target/post-event fields
for review. Feature-name matching does not validate feature meaning or certify
that data is suitable for the saved model.

**Dataset source and attribution:** CSE-CIC-IDS2018 is published through the
official Canadian Institute for Cybersecurity (CIC), University of New
Brunswick dataset page, describing it as a collaboration between the
Communications Security Establishment (CSE) and CIC:
<https://www.unb.ca/cic/datasets/ids-2018.html>. Obtain files manually from
that official page and review its usage terms and citation guidance before
redistribution or research use. Attribute the dataset to CSE and the Canadian
Institute for Cybersecurity, University of New Brunswick, and identify it as
CSE-CIC-IDS2018. FPGuard AI does not bundle or automatically fetch it.

This audit is data-quality and provenance triage, not a model evaluation and
not proof that a dataset is leakage-free. The project’s current trained model,
metrics, and dashboard remain based on synthetic data. Until a genuine dataset
has actually been audited and evaluated separately, no real-data validation or
real-world NIDS effectiveness is established.

## Stage 6B: real-data training and evaluation

The separate real-data CLI requires both explicit CSV inputs and the Stage 6A
audit report for those exact files. The report's resolved source paths, SHA-256
hashes, row counts, and explicit label aliases are checked before training.
The program accepts CSV only; it does not convert workbooks, download data,
modify the synthetic pipeline, or overwrite outputs outside a dedicated
Stage 6B directory.

```powershell
.\.venv\Scripts\python.exe -m fpguard.real_train --help
.\.venv\Scripts\python.exe -m fpguard.real_train `
  --audit-report .\data\audit_reports\cic_ids2018\real_data_audit.json `
  --output-dir .\artifacts\real_data `
  --minimum-recall 0.80 `
  --random-state 42 `
  .\data\raw\cic_ids2018\Monday-WorkingHours.csv
```

Pass every CSV that was included in the same audit (and whose normalized
feature schema agrees). `--max-rows N` enables a reproducible, seeded
development sample; sample-based results are not a substitute for a full-data
evaluation. The classifier candidates are regularized Logistic Regression,
Random Forest, and Extra Trees, with bounded deterministic settings. Feature
names that appear to represent labels, attack categories, identifiers, IP
addresses, timestamps, or target-derived data are excluded and recorded for
review; this name-based policy cannot certify semantic safety. Numeric
conversions are validated, non-finite values become missing, and imputers,
scalers, and encoders are fitted within each model pipeline using training
rows only.

Exact duplicate predictive-feature rows are kept together in a deterministic,
stratified group split. When at least five usable capture sessions/days or
source-file groups are available, those contexts are also kept together. The
resulting five folds are assigned as folds 2–4 for training, fold 1 for
validation, and fold 0 for test. If no independent capture context is
available, the manifest says so: duplicate-only grouping cannot establish
performance on an unseen capture, network, or attack family. The group split
can reject data when it cannot place both classes in every partition rather
than silently weakening the evaluation.

For each candidate, threshold selection uses validation probabilities and
labels only. It checks thresholds from 0.00 to 1.00 in increments of 0.01,
selects the lowest validation FPR among thresholds meeting the configured
minimum attack recall, and breaks ties by precision and then higher threshold.
The default minimum validation recall is 0.80. If no candidate is feasible,
the report records that no model was selected and does not evaluate a chosen
test policy. Otherwise, the validation winner and threshold are frozen before
the test partition is evaluated for the final comparison. Meeting the recall
constraint on validation does not guarantee meeting it on test data.

Outputs are written only to the requested dedicated directory (by default
`artifacts/real_data/`): `split_manifest.json`, `model_comparison.json`,
`selected_model.joblib`, `selected_model_metadata.json`, `test_metrics.json`,
and `diagnostics.csv`. A marker file protects forced updates: `--force` refuses
to replace existing model/report filenames in a directory not already marked
as Stage 6B output. Do not put downloaded datasets in Git.

FPR is false positives divided by benign rows; recall is detected attack rows
divided by all attack rows; false positives are benign flows flagged as
attacks, and false negatives are attacks predicted benign. The report includes
confusion counts, FPR/FNR, precision, recall, F1, accuracy, ROC-AUC and
average precision where defined, and measured baseline-to-selected changes.
Undefined metrics are JSON `null` with notes. These measurements describe the
particular audited data and split only. Synthetic results remain synthetic;
real-data results—even when available—do not establish effectiveness on other
networks or detection of unseen attack families.

At the time of this implementation, the local CIC-IDS2018 item previously
identified in this workspace was an XLSX workbook rather than a processed CSV,
and no matching Stage 6A audit report was present. Therefore this stage's
training pipeline is infrastructure only until a processed CSV is explicitly
audited; no genuine dataset has been trained or evaluated by this stage.

## Data and leakage safeguards

- Validate CSV schemas before processing and report missing required columns.
  The loader does not silently drop rows or columns. The experiment pipeline
  also rejects unexpected columns rather than silently training on them.
- Exclude identifiers, post-event information, and target-derived columns from
  predictive features; document every feature decision.
- Split into training, validation, and test data before fitting preprocessing
  or a classifier. Fit preprocessing and the model using training data only.
- Threshold selection uses validation labels only and a documented minimum
  recall constraint. Never use test labels for threshold or model choices.
- Evaluate baseline and selected threshold on the same untouched test set.
  Report measured results honestly; threshold adjustment is not guaranteed to
  improve false-positive rates or any other metric.
- Do not automatically download large datasets. Keep raw data and generated
  artifacts out of Git.

## Planned milestones

1. Consider a clearly identified external demo fixture only if its provenance
   and license are suitable.
2. Improve model and threshold strategies using validation data only; keep the
   test partition untouched for final comparisons.
3. Extend the dashboard only with additional validated experiments, clear
   data provenance, and no unsupported claims.
#   c y b e r - a i  
 