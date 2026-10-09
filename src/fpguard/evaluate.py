"""Baseline metrics for validation diagnostics and untouched test evaluation."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from fpguard.data import (
    DEFAULT_DATA_PATH,
    DEFAULT_RANDOM_STATE,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    LABELS,
    load_csv,
    split_dataset,
    validate_dataset,
)
from fpguard.train import DEFAULT_ARTIFACT_DIR, MODEL_FILENAME
from fpguard.threshold import (
    DEFAULT_CANDIDATE_THRESHOLDS,
    select_threshold,
)

METRICS_FILENAME = "metrics.json"
THRESHOLD_EXPERIMENT_FILENAME = "threshold_experiment.json"
ERROR_TYPES = ("false_positive", "false_negative", "true_positive", "true_negative")


def evaluate_binary_classifier(
    labels: Sequence[str],
    probabilities: Sequence[float],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute binary metrics; confusion matrix order is benign, then attack.

    Undefined rates/metrics caused by zero denominators are represented by
    ``None`` rather than a fabricated zero.
    """
    y_true = np.asarray(labels)
    scores = np.asarray(probabilities, dtype=float)
    if y_true.ndim != 1 or scores.ndim != 1 or len(y_true) != len(scores):
        raise ValueError("labels and probabilities must be one-dimensional and equal length")
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty batch")
    if set(np.unique(y_true)) - set(LABELS):
        raise ValueError(f"Labels must be drawn from {LABELS}")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite value between 0 and 1")

    y_pred = np.where(scores >= threshold, "attack", "benign")
    matrix = confusion_matrix(y_true, y_pred, labels=list(LABELS))
    tn, fp, fn, tp = (int(matrix[0, 0]), int(matrix[0, 1]), int(matrix[1, 0]), int(matrix[1, 1]))
    attack_present = "attack" in y_true
    benign_present = "benign" in y_true
    auc = (
        float(roc_auc_score(y_true == "attack", scores))
        if attack_present and benign_present
        else None
    )
    return {
        "class_order": list(LABELS),
        "confusion_matrix": matrix.tolist(),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "false_negative_rate": fn / (fn + tp) if fn + tp else None,
        "precision": float(precision_score(y_true, y_pred, pos_label="attack", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label="attack", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, pos_label="attack", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": auc,
        "threshold": float(threshold),
    }


def evaluate_model(
    model: Pipeline,
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate an already-fitted model and record local inference duration."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("The fitted model must support predict_proba")
    started = time.perf_counter()
    probabilities_matrix = model.predict_proba(features)
    elapsed = time.perf_counter() - started
    classes = list(model.classes_)
    if "attack" not in classes:
        raise ValueError("Fitted model does not contain the required positive class 'attack'")
    probabilities = probabilities_matrix[:, classes.index("attack")]
    metrics = evaluate_binary_classifier(labels, probabilities, threshold=threshold)
    metrics["local_inference_benchmark"] = {
        "elapsed_seconds": elapsed,
        "rows": len(features),
        "milliseconds_per_row": elapsed * 1_000 / len(features) if len(features) else None,
        "note": "One local batch measurement; not a universal performance claim.",
    }
    return metrics


def evaluate_saved_baseline(
    dataset_path: str | Path = DEFAULT_DATA_PATH,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Write validation diagnostics and final test metrics for a saved baseline."""
    artifact_path = Path(artifact_dir)
    metadata_path = artifact_path / "baseline_metadata.json"
    model_path = artifact_path / MODEL_FILENAME
    source = Path(dataset_path)
    if not metadata_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(
            f"Baseline model artifacts were not found in '{artifact_path}'; "
            "run `python -m fpguard.train` first"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if dataset_hash != metadata["dataset_sha256"]:
        raise ValueError(
            "Dataset differs from the version used to train the saved model; "
            "regenerate and retrain before evaluating"
        )

    frame = load_csv(source, required_columns=(*FEATURE_COLUMNS, LABEL_COLUMN))
    frame = frame.reset_index(drop=True)
    validate_dataset(frame)
    splits = split_dataset(frame, random_state=random_state)
    if metadata["feature_names"] != list(splits.X_train.columns):
        raise ValueError("Dataset feature schema differs from the saved model metadata")
    if metadata["split_sizes"] != {
        "train": len(splits.y_train),
        "validation": len(splits.y_validation),
        "test": len(splits.y_test),
    }:
        raise ValueError("Dataset split sizes differ from the saved model metadata")

    model = joblib.load(model_path)
    threshold = float(metadata["training_configuration"]["threshold"])
    metrics = {
        "experiment_note": "Synthetic demonstration data; not real-world IDS evidence.",
        "positive_class": "attack",
        "confusion_matrix_class_order": list(LABELS),
        "validation_diagnostics": evaluate_model(
            model, splits.X_validation, splits.y_validation, threshold=threshold
        ),
        "test_baseline": evaluate_model(
            model, splits.X_test, splits.y_test, threshold=threshold
        ),
    }
    (artifact_path / METRICS_FILENAME).write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return metrics


def _load_experiment_inputs(
    dataset_path: Path,
    artifact_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, list[int]]]:
    metadata_path = artifact_path / "baseline_metadata.json"
    manifest_path = artifact_path / "split_indices.json"
    model_path = artifact_path / MODEL_FILENAME
    for required_path in (dataset_path, metadata_path, manifest_path, model_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required saved experiment file does not exist: {required_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if dataset_hash != metadata.get("dataset_sha256"):
        raise ValueError("Dataset differs from the version used to train the saved model")
    if dataset_hash != manifest.get("dataset_sha256"):
        raise ValueError("Saved split manifest does not match the dataset")

    frame = load_csv(dataset_path, required_columns=(*FEATURE_COLUMNS, LABEL_COLUMN))
    unexpected = [
        column for column in frame if column not in (*FEATURE_COLUMNS, LABEL_COLUMN)
    ]
    if unexpected:
        raise ValueError(f"Unsupported extra dataset column(s): {', '.join(unexpected)}")
    non_numeric = [
        column
        for column in FEATURE_COLUMNS
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise TypeError(f"Feature column(s) must be numeric: {', '.join(non_numeric)}")
    if metadata.get("feature_names") != list(FEATURE_COLUMNS):
        raise ValueError("Saved model metadata feature schema does not match the dataset")

    raw_indices = manifest.get("indices")
    if not isinstance(raw_indices, dict):
        raise ValueError("Saved split manifest is missing its indices mapping")
    expected_names = ("train", "validation", "test")
    if set(raw_indices) != set(expected_names):
        raise ValueError("Saved split manifest must contain train, validation, and test indices")
    indices: dict[str, list[int]] = {}
    for name in expected_names:
        values = raw_indices[name]
        if not isinstance(values, list) or any(
            not isinstance(index, int) or isinstance(index, bool) for index in values
        ):
            raise ValueError(f"Saved {name} split indices must be a list of integers")
        indices[name] = values

    index_sets = {name: set(values) for name, values in indices.items()}
    if any(len(index_sets[name]) != len(indices[name]) for name in expected_names):
        raise ValueError("Saved split manifest contains duplicate row indices")
    if (
        index_sets["train"] & index_sets["validation"]
        or index_sets["train"] & index_sets["test"]
        or index_sets["validation"] & index_sets["test"]
    ):
        raise ValueError("Saved split manifest partitions overlap")
    if set.union(*index_sets.values()) != set(range(len(frame))):
        raise ValueError("Saved split manifest does not partition every dataset row exactly once")
    expected_sizes = metadata.get("split_sizes")
    if expected_sizes != {name: len(indices[name]) for name in expected_names}:
        raise ValueError("Saved split sizes do not match model metadata")
    if manifest.get("random_state") != metadata.get("random_state"):
        raise ValueError("Saved split manifest seed does not match model metadata")
    return frame, metadata, indices


def _attack_probabilities(
    model: Pipeline,
    features: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    """Get positive-class probabilities using the estimator's actual class order."""
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise TypeError("The saved model must be a fitted probability pipeline")
    classes = list(model.classes_)
    if "attack" not in classes:
        raise ValueError("Fitted model does not contain the required positive class 'attack'")
    started = time.perf_counter()
    probabilities = model.predict_proba(features)[:, classes.index("attack")]
    elapsed = time.perf_counter() - started
    return np.asarray(probabilities, dtype=float), elapsed


def _with_local_benchmark(
    metrics: dict[str, Any],
    elapsed_seconds: float,
    row_count: int,
) -> dict[str, Any]:
    result = dict(metrics)
    result["local_inference_benchmark"] = {
        "elapsed_seconds": elapsed_seconds,
        "rows": row_count,
        "milliseconds_per_row": elapsed_seconds * 1_000 / row_count if row_count else None,
        "note": "One local batch measurement; not a universal performance claim.",
    }
    return result


def create_baseline_diagnostics(
    features: pd.DataFrame,
    labels: pd.Series,
    probabilities: Sequence[float],
    *,
    threshold: float = 0.5,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Build post-hoc test error and exploratory feature-group diagnosis.

    Feature summaries describe synthetic test-set associations only; they are
    not causal explanations or feature-selection evidence.
    """
    if not features.index.equals(labels.index):
        raise ValueError("Diagnostic features and labels must have matching row indices")
    scores = np.asarray(probabilities, dtype=float)
    if len(scores) != len(features):
        raise ValueError("Diagnostic probabilities must match the number of feature rows")
    metrics = evaluate_binary_classifier(labels.to_numpy(), scores, threshold=threshold)
    predicted = np.where(scores >= threshold, "attack", "benign")
    errors = np.select(
        [
            (labels.to_numpy() == "benign") & (predicted == "attack"),
            (labels.to_numpy() == "attack") & (predicted == "benign"),
            (labels.to_numpy() == "attack") & (predicted == "attack"),
        ],
        ["false_positive", "false_negative", "true_positive"],
        default="true_negative",
    )
    row_diagnostics = pd.DataFrame(
        {
            "original_row_index": features.index.to_numpy(),
            "true_label": labels.to_numpy(),
            "predicted_label": predicted,
            "attack_probability": scores,
            "error_type": errors,
        }
    )

    feature_records: list[dict[str, Any]] = []
    comparisons = (
        ("false_positives_vs_true_negatives", "false_positive", "true_negative"),
        ("false_negatives_vs_true_positives", "false_negative", "true_positive"),
    )
    finite_features = features.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    for comparison_name, first_error, second_error in comparisons:
        first_mask = row_diagnostics["error_type"] == first_error
        second_mask = row_diagnostics["error_type"] == second_error
        for feature_name in finite_features.columns:
            first_values = finite_features.loc[first_mask.to_numpy(), feature_name].dropna()
            second_values = finite_features.loc[second_mask.to_numpy(), feature_name].dropna()
            sufficient_samples = (
                len(first_values) >= 2 and len(second_values) >= 2
            )
            first_mean = float(first_values.mean()) if sufficient_samples else None
            second_mean = float(second_values.mean()) if sufficient_samples else None
            first_median = float(first_values.median()) if sufficient_samples else None
            second_median = float(second_values.median()) if sufficient_samples else None
            feature_records.append(
                {
                    "comparison": comparison_name,
                    "first_group": first_error,
                    "second_group": second_error,
                    "feature": feature_name,
                    "first_group_non_missing_n": int(len(first_values)),
                    "second_group_non_missing_n": int(len(second_values)),
                    "minimum_group_rows_for_summary": 2,
                    "first_group_mean": first_mean,
                    "second_group_mean": second_mean,
                    "mean_difference_first_minus_second": (
                        first_mean - second_mean
                        if first_mean is not None and second_mean is not None
                        else None
                    ),
                    "first_group_median": first_median,
                    "second_group_median": second_median,
                    "median_difference_first_minus_second": (
                        first_median - second_median
                        if first_median is not None and second_median is not None
                        else None
                    ),
                    "interpretation": (
                        "Exploratory association on synthetic test data; not causal."
                    ),
                }
            )

    diagnosis = {
        "purpose": "post-hoc baseline test-set diagnosis; not model selection",
        "data_note": "Synthetic demonstration data; not real-world IDS evidence.",
        "threshold": float(threshold),
        "class_order": list(LABELS),
        "metrics": metrics,
        "false_positive_count": metrics["false_positives"],
        "false_negative_count": metrics["false_negatives"],
        "false_positive_proportion_among_benign": metrics["false_positive_rate"],
        "false_negative_proportion_among_attack": metrics["false_negative_rate"],
        "error_type_counts": {
            error_type: int((row_diagnostics["error_type"] == error_type).sum())
            for error_type in ERROR_TYPES
        },
        "feature_comparison_note": (
            "Group summaries are exploratory associations only. Small group sizes "
            "and synthetic feature distributions limit interpretation."
        ),
    }
    return diagnosis, row_diagnostics, pd.DataFrame(feature_records)


def run_threshold_experiment(
    dataset_path: str | Path = DEFAULT_DATA_PATH,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    *,
    minimum_recall: float = 0.80,
) -> dict[str, Any]:
    """Select on validation only, then compare policies on shared untouched test rows."""
    artifact_path = Path(artifact_dir)
    source = Path(dataset_path)
    frame, metadata, split_indices = _load_experiment_inputs(source, artifact_path)
    model = joblib.load(artifact_path / MODEL_FILENAME)
    expected_features = list(metadata["feature_names"])
    if not isinstance(model, Pipeline) or list(model.feature_names_in_) != expected_features:
        raise ValueError("Saved model feature schema does not match its metadata")
    if "attack" not in list(model.classes_):
        raise ValueError("Saved model does not include 'attack' as a fitted class")

    validation_positions = split_indices["validation"]
    X_validation = frame.iloc[validation_positions].loc[:, expected_features]
    y_validation = frame.iloc[validation_positions][LABEL_COLUMN]
    validation_probabilities, validation_seconds = _attack_probabilities(model, X_validation)
    threshold_result = select_threshold(
        validation_probabilities,
        y_validation.to_numpy(),
        minimum_recall=minimum_recall,
        candidate_thresholds=DEFAULT_CANDIDATE_THRESHOLDS,
    )
    selected_threshold = threshold_result["selected_threshold"]

    # Test labels/features are read only after validation has frozen the threshold.
    test_positions = split_indices["test"]
    X_test = frame.iloc[test_positions].loc[:, expected_features]
    y_test = frame.iloc[test_positions][LABEL_COLUMN]
    test_probabilities, test_seconds = _attack_probabilities(model, X_test)
    baseline_threshold = float(metadata["training_configuration"]["threshold"])
    validation_baseline = evaluate_binary_classifier(
        y_validation.to_numpy(),
        validation_probabilities,
        threshold=baseline_threshold,
    )
    validation_report: dict[str, Any] = {
        "baseline": validation_baseline,
        "alternative_policy": (
            threshold_result["selected_validation_metrics"]
            if threshold_result["feasible"]
            else None
        ),
    }
    test_baseline = _with_local_benchmark(
        evaluate_binary_classifier(
            y_test.to_numpy(),
            test_probabilities,
            threshold=baseline_threshold,
        ),
        test_seconds,
        len(X_test),
    )

    test_report: dict[str, Any] = {"baseline": test_baseline, "alternative_policy": None}
    comparison: dict[str, Any] | None = None
    if threshold_result["feasible"]:
        alternative_test = _with_local_benchmark(
            evaluate_binary_classifier(
                y_test.to_numpy(),
                test_probabilities,
                threshold=float(selected_threshold),
            ),
            test_seconds,
            len(X_test),
        )
        alternative_fpr = alternative_test["false_positive_rate"]
        baseline_fpr = test_baseline["false_positive_rate"]
        test_report["alternative_policy"] = alternative_test
        comparison = {
            "test_rows_shared_by_both_policies": len(test_positions),
            "same_test_probability_vector_used": True,
            "fpr_change_percentage_points_alternative_minus_baseline": (
                (alternative_fpr - baseline_fpr) * 100
                if alternative_fpr is not None and baseline_fpr is not None
                else None
            ),
            "absolute_fpr_change_percentage_points": (
                abs(alternative_fpr - baseline_fpr) * 100
                if alternative_fpr is not None and baseline_fpr is not None
                else None
            ),
            "relative_fpr_reduction_percent": (
                (baseline_fpr - alternative_fpr) / baseline_fpr * 100
                if baseline_fpr not in (None, 0) and alternative_fpr is not None
                else None
            ),
            "recall_change_percentage_points_alternative_minus_baseline": (
                (alternative_test["recall"] - test_baseline["recall"]) * 100
            ),
            "false_negative_count_change_alternative_minus_baseline": (
                alternative_test["false_negatives"] - test_baseline["false_negatives"]
            ),
            "alternative_meets_minimum_recall_on_test": (
                alternative_test["recall"] >= minimum_recall
            ),
        }

    report = {
        "experiment": "validation-driven attack probability threshold selection",
        "data_note": "Synthetic demonstration data; not real-world IDS evidence.",
        "warning": (
            "Meeting the recall target on validation data does not guarantee "
            "the same recall on the untouched test data."
        ),
        "configuration": {
            "dataset_sha256": metadata["dataset_sha256"],
            "random_state": metadata["random_state"],
            "minimum_validation_recall": float(minimum_recall),
            "positive_class": "attack",
            "baseline_threshold": baseline_threshold,
            "candidate_thresholds": list(DEFAULT_CANDIDATE_THRESHOLDS),
            "threshold_tie_breaker": "lowest FPR, then highest precision, then highest threshold",
            "model_retrained": False,
            "saved_split_manifest_used": True,
        },
        "validation_target_feasible": threshold_result["feasible"],
        "selected_threshold": selected_threshold,
        "selection_reason": threshold_result["selection_reason"],
        "validation": validation_report,
        "test": test_report,
        "test_comparison": comparison,
        "threshold_selection_details": threshold_result,
        "validation_inference_seconds": validation_seconds,
    }
    report_path = artifact_path / THRESHOLD_EXPERIMENT_FILENAME
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the FPGuard baseline model.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument(
        "--threshold-experiment",
        action="store_true",
        help="Run validation-driven threshold selection and untouched test comparison.",
    )
    parser.add_argument("--minimum-recall", type=float, default=0.80)
    args = parser.parse_args()
    if args.threshold_experiment:
        results = run_threshold_experiment(
            args.dataset, args.artifacts, minimum_recall=args.minimum_recall
        )
    else:
        results = evaluate_saved_baseline(args.dataset, args.artifacts, random_state=args.seed)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
