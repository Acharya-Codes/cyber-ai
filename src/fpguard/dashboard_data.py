"""Local-only data and artifact helpers used by the Streamlit dashboard."""

import csv
import hashlib
import io
import json
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from fpguard.data import DEFAULT_DATA_PATH, LABEL_COLUMN, LABELS
from fpguard.evaluate import evaluate_binary_classifier
from fpguard.predict import load_model, predict_probabilities
from fpguard.train import DEFAULT_ARTIFACT_DIR

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_UPLOAD_ROWS = 5_000
IDENTIFIER_COLUMNS = {
    "id",
    "flow_id",
    "record_id",
    "src_ip",
    "dst_ip",
    "source_ip",
    "destination_ip",
    "source_address",
    "destination_address",
    "timestamp",
    "flow_start_time",
}
REPORT_FILES = {
    "baseline_metrics": "metrics.json",
    "stage3_threshold": "threshold_experiment.json",
    "stage4_comparison": "stage4_model_comparison.json",
    "stage4_diagnosis": "stage4_baseline_diagnosis.json",
}


def load_json_report(path: str | Path) -> dict[str, Any] | None:
    """Read a local JSON report, returning None only when it is absent."""
    report_path = Path(path)
    if not report_path.is_file():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read report '{report_path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Report '{report_path}' must contain a JSON object")
    return payload


def load_trusted_pipeline(path: str | Path) -> Pipeline:
    """Load only an explicitly supplied local project pipeline artifact."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at '{model_path}'. Run the documented "
            "training or Stage 4 experiment command first."
        )
    try:
        return load_model(model_path)
    except (
        OSError,
        EOFError,
        KeyError,
        ValueError,
        TypeError,
        pickle.UnpicklingError,
    ) as exc:
        raise ValueError(f"Model artifact '{model_path}' is invalid or incompatible: {exc}") from exc


def validate_selected_model_artifact(
    model: Pipeline,
    metadata_path: str | Path,
    comparison_report: dict[str, Any] | None,
    *,
    dataset_path: str | Path,
    manifest_path: str | Path,
) -> None:
    """Ensure the selected pipeline, sidecar, and comparison report agree."""
    path = Path(metadata_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Selected-model metadata is missing at '{path}'. Rerun the Stage 4 comparison."
        )
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Selected-model metadata is invalid: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Selected-model metadata must be a JSON object.")
    if not comparison_report:
        raise ValueError("Stage 4 comparison report is missing; selected model cannot be verified.")
    configuration = comparison_report.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("Stage 4 report is missing its experiment configuration.")
    expected_names = comparison_report.get("selected_model")
    expected_threshold = comparison_report.get("selected_threshold")
    report_hash = configuration.get("dataset_sha256")
    if not comparison_report.get("selection_feasible") or not expected_names:
        raise ValueError("Stage 4 report does not contain a feasible selected model.")
    if metadata.get("selected_model") != expected_names:
        raise ValueError("Selected model artifact does not match the Stage 4 report.")
    if metadata.get("selected_threshold") != expected_threshold:
        raise ValueError("Selected model threshold does not match the Stage 4 report.")
    if metadata.get("dataset_sha256") != report_hash:
        raise ValueError("Selected model metadata belongs to a different dataset.")
    data_file = Path(dataset_path)
    split_file = Path(manifest_path)
    if not data_file.is_file() or not split_file.is_file():
        raise FileNotFoundError("The saved dataset or split manifest needed to verify Stage 4 is missing.")
    dataset_hash = hashlib.sha256(data_file.read_bytes()).hexdigest()
    manifest_hash = hashlib.sha256(split_file.read_bytes()).hexdigest()
    if dataset_hash != report_hash or metadata.get("dataset_sha256") != dataset_hash:
        raise ValueError("Saved dataset does not match the selected-model experiment.")
    if metadata.get("split_manifest_sha256") != manifest_hash:
        raise ValueError("Saved split manifest does not match the selected-model experiment.")
    if metadata.get("positive_class") != "attack":
        raise ValueError("Selected model metadata has an unsupported positive class.")
    if metadata.get("feature_names") != list(model.feature_names_in_):
        raise ValueError("Selected model feature schema does not match its metadata.")


def read_uploaded_flow_csv(
    content: bytes,
    feature_names: Sequence[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Validate an uploaded CSV and return ordered features plus metadata columns.

    Existing row order is preserved. The target and explicitly recognized
    identifiers are metadata only; they are never passed to a classifier.
    Missing and infinite numeric values are accepted because the saved pipeline
    imputes them. Non-numeric non-empty feature values are rejected.
    """
    if not content:
        raise ValueError("The uploaded CSV is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"File is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit."
        )
    try:
        decoded = content.decode("utf-8-sig")
        header = next(csv.reader(io.StringIO(decoded)), None)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"Could not read the CSV header: {exc}") from exc
    if not header:
        raise ValueError("The uploaded CSV has no column header.")
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in header:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    if duplicates:
        raise ValueError(f"Duplicate CSV column name(s): {', '.join(sorted(duplicates))}")
    if any(not name.strip() for name in header):
        raise ValueError("The CSV contains a blank column name.")

    try:
        frame = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not parse the uploaded CSV: {exc}") from exc
    if frame.empty:
        raise ValueError("The uploaded CSV contains headers but no flow rows.")
    if len(frame) > MAX_UPLOAD_ROWS:
        raise ValueError(
            f"File contains {len(frame):,} rows; the limit is {MAX_UPLOAD_ROWS:,}."
        )

    required = list(feature_names)
    if not required or len(set(required)) != len(required):
        raise ValueError("The saved model feature schema is empty or contains duplicate names.")
    if LABEL_COLUMN in required:
        raise ValueError("The saved model feature schema must not include the target label.")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature column(s): {', '.join(missing)}")
    metadata_columns = [
        name
        for name in frame.columns
        if name == LABEL_COLUMN or name.lower() in IDENTIFIER_COLUMNS
    ]
    if LABEL_COLUMN in frame.columns:
        supplied_labels = set(frame[LABEL_COLUMN].dropna().unique())
        invalid_labels = supplied_labels - set(LABELS)
        if invalid_labels:
            raise ValueError(
                f"Optional '{LABEL_COLUMN}' values must be drawn from {LABELS}; "
                f"found {sorted(invalid_labels)}."
            )
    unexpected = [
        name
        for name in frame.columns
        if name not in required and name not in metadata_columns
    ]
    if unexpected:
        raise ValueError(
            "Unsupported column(s): "
            f"{', '.join(unexpected)}. Keep only required flow features, optional "
            "`label`, and recognized identifier columns."
        )

    ordered = frame.loc[:, required].copy()
    for name in required:
        original = ordered[name]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        if invalid.any():
            examples = original.loc[invalid].astype(str).head(3).tolist()
            raise ValueError(
                f"Feature '{name}' contains non-numeric value(s), for example: {examples}"
            )
        ordered[name] = converted
    return ordered, metadata_columns


def predict_uploaded_flows(
    content: bytes,
    model: Pipeline,
    *,
    threshold: float,
) -> pd.DataFrame:
    """Predict a validated upload while preserving its order and metadata."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Decision threshold must be between 0.00 and 1.00.")
    feature_names = list(model.feature_names_in_)
    ordered_features, _ = read_uploaded_flow_csv(content, feature_names)
    original = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
    probabilities = predict_probabilities(model, ordered_features)
    output = original.copy()
    output["predicted_label"] = np.where(probabilities.to_numpy() >= threshold, "attack", "benign")
    output["attack_probability"] = probabilities.to_numpy()
    output["decision_threshold"] = float(threshold)
    return output.reset_index(drop=True)


def validation_threshold_curve(
    dataset_path: str | Path = DEFAULT_DATA_PATH,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    *,
    model: Pipeline,
) -> pd.DataFrame:
    """Compute a retrospective threshold curve on the saved validation partition."""
    data_path = Path(dataset_path)
    artifacts = Path(artifact_dir)
    manifest_path = artifacts / "split_indices.json"
    if not data_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "The saved dataset or split manifest is missing. Run the documented "
            "data generation and baseline setup commands first."
        )
    frame = pd.read_csv(data_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if manifest.get("dataset_sha256") != dataset_hash:
        raise ValueError("Saved split manifest does not match the saved dataset.")
    validation_indices = manifest.get("indices", {}).get("validation")
    if (
        not isinstance(validation_indices, list)
        or not validation_indices
        or any(type(index) is not int for index in validation_indices)
        or len(set(validation_indices)) != len(validation_indices)
        or min(validation_indices) < 0
        or max(validation_indices) >= len(frame)
    ):
        raise ValueError("Saved split manifest is missing validation indices.")
    feature_names = list(model.feature_names_in_)
    if LABEL_COLUMN not in frame or any(name not in frame for name in feature_names):
        raise ValueError("Saved validation data does not match the model feature schema.")
    validation = frame.iloc[validation_indices]
    features = validation.loc[:, feature_names]
    labels = validation[LABEL_COLUMN]
    if model.feature_names_in_.tolist() != feature_names:
        raise ValueError("Loaded model feature schema is inconsistent.")
    probabilities = predict_probabilities(model, features)
    rows: list[dict[str, Any]] = []
    for step in range(101):
        threshold = step / 100
        metrics = evaluate_binary_classifier(
            labels.to_numpy(), probabilities.to_numpy(), threshold=threshold
        )
        rows.append(
            {
                "threshold": threshold,
                "false_positive_rate": metrics["false_positive_rate"],
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "f1_score": metrics["f1_score"],
                "false_positives": metrics["false_positives"],
                "false_negatives": metrics["false_negatives"],
            }
        )
    return pd.DataFrame(rows)
