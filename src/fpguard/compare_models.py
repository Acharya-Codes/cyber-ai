"""Training-only candidate fitting and validation-only model/threshold selection."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fpguard.data import DEFAULT_DATA_PATH, DEFAULT_RANDOM_STATE, LABEL_COLUMN
from fpguard.evaluate import (
    THRESHOLD_EXPERIMENT_FILENAME,
    _attack_probabilities,
    _load_experiment_inputs,
    create_baseline_diagnostics,
    evaluate_binary_classifier,
)
from fpguard.predict import load_model
from fpguard.threshold import DEFAULT_CANDIDATE_THRESHOLDS, select_threshold
from fpguard.train import DEFAULT_ARTIFACT_DIR, MODEL_FILENAME

COMPARISON_FILENAME = "stage4_model_comparison.json"
DIAGNOSIS_FILENAME = "stage4_baseline_diagnosis.json"
DIAGNOSTICS_CSV_FILENAME = "stage4_baseline_diagnostics.csv"
FEATURE_DIAGNOSTICS_CSV_FILENAME = "stage4_feature_diagnostics.csv"
SELECTED_MODEL_FILENAME = "stage4_selected_model.joblib"
SELECTED_METADATA_FILENAME = "stage4_selected_model_metadata.json"


def make_additional_candidate_pipelines(
    feature_names: Sequence[str],
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Pipeline]:
    """Create deterministic additional model pipelines for numeric flow data."""
    from fpguard.preprocessing import build_preprocessor

    return {
        "extra_trees": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(feature_names)),
                (
                    "classifier",
                    ExtraTreesClassifier(
                        n_estimators=100,
                        max_depth=12,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(feature_names)),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=500,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def select_winning_candidate(
    candidates: dict[str, dict[str, Any]],
) -> str | None:
    """Choose only from candidate validation summaries, never test metrics."""
    eligible = [
        (name, result)
        for name, result in candidates.items()
        if result["feasible"]
    ]
    if not eligible:
        return None
    winner, _ = min(
        eligible,
        key=lambda item: (
            item[1]["validation_metrics"]["false_positive_rate"],
            -item[1]["validation_metrics"]["precision"],
            -item[1]["validation_metrics"]["recall"],
            item[0],
        ),
    )
    return winner


def _load_stage3_threshold(artifact_path: Path, dataset_hash: str) -> float | None:
    stage3_path = artifact_path / THRESHOLD_EXPERIMENT_FILENAME
    if not stage3_path.is_file():
        return None
    stage3_report = json.loads(stage3_path.read_text(encoding="utf-8"))
    if stage3_report.get("configuration", {}).get("dataset_sha256") != dataset_hash:
        raise ValueError("Stage 3 threshold experiment belongs to a different dataset")
    if not stage3_report.get("validation_target_feasible"):
        return None
    threshold = stage3_report.get("selected_threshold")
    if not isinstance(threshold, (float, int)) or not 0 <= threshold <= 1:
        raise ValueError("Stage 3 experiment has an invalid selected threshold")
    return float(threshold)


def _metric_change(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    baseline_fpr = baseline["false_positive_rate"]
    candidate_fpr = candidate["false_positive_rate"]
    signed_fpr_change = (
        (candidate_fpr - baseline_fpr) * 100
        if baseline_fpr is not None and candidate_fpr is not None
        else None
    )
    relative_reduction = (
        (baseline_fpr - candidate_fpr) / baseline_fpr * 100
        if baseline_fpr not in (None, 0) and candidate_fpr is not None
        else None
    )
    return {
        "absolute_fpr_change_percentage_points_candidate_minus_baseline": signed_fpr_change,
        "absolute_magnitude_fpr_change_percentage_points": (
            abs(signed_fpr_change) if signed_fpr_change is not None else None
        ),
        "relative_fpr_reduction_percent": relative_reduction,
        "recall_change_percentage_points_candidate_minus_baseline": (
            (candidate["recall"] - baseline["recall"]) * 100
        ),
        "false_positive_count_change_candidate_minus_baseline": (
            candidate["false_positives"] - baseline["false_positives"]
        ),
        "false_negative_count_change_candidate_minus_baseline": (
            candidate["false_negatives"] - baseline["false_negatives"]
        ),
        "precision_change_candidate_minus_baseline": (
            candidate["precision"] - baseline["precision"]
        ),
        "f1_change_candidate_minus_baseline": (
            candidate["f1_score"] - baseline["f1_score"]
        ),
        "relative_fpr_reduction_undefined_reason": (
            "Baseline FPR is zero or undefined."
            if baseline_fpr in (None, 0)
            else None
        ),
    }


def _json_safe(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to standards-compliant JSON data."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def compare_saved_models(
    dataset_path: str | Path = DEFAULT_DATA_PATH,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    *,
    minimum_recall: float = 0.80,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Compare the saved baseline with new candidates using validation only.

    All new candidate pipelines are fitted exclusively on the saved training
    indices. Model and threshold selection use only validation labels and
    probabilities. Test labels are first read after the winning configuration
    has been frozen.
    """
    artifact_path = Path(artifact_dir)
    source = Path(dataset_path)
    frame, metadata, split_indices = _load_experiment_inputs(source, artifact_path)
    if random_state != metadata["random_state"]:
        raise ValueError("Stage 4 must reuse the random state recorded in baseline metadata")
    if not np.isfinite(minimum_recall) or not 0 < minimum_recall <= 1:
        raise ValueError("minimum_recall must be greater than 0 and at most 1")

    baseline_model = load_model(artifact_path / MODEL_FILENAME)
    feature_names = list(metadata["feature_names"])
    if list(baseline_model.feature_names_in_) != feature_names:
        raise ValueError("Saved baseline feature schema does not match its metadata")

    train_positions = split_indices["train"]
    validation_positions = split_indices["validation"]
    X_train = frame.iloc[train_positions].loc[:, feature_names]
    y_train = frame.iloc[train_positions][LABEL_COLUMN]
    X_validation = frame.iloc[validation_positions].loc[:, feature_names]
    y_validation = frame.iloc[validation_positions][LABEL_COLUMN]
    if set(y_train.unique()) != {"benign", "attack"}:
        raise ValueError("Training partition must contain benign and attack labels")
    if set(y_validation.unique()) != {"benign", "attack"}:
        raise ValueError("Validation partition must contain benign and attack labels")

    fitted_candidates: dict[str, Pipeline] = {"random_forest_baseline": baseline_model}
    new_candidates = make_additional_candidate_pipelines(
        feature_names, random_state=random_state
    )
    for name, pipeline in new_candidates.items():
        pipeline.fit(X_train, y_train)
        fitted_candidates[name] = pipeline

    validation_results: dict[str, dict[str, Any]] = {}
    for name, model in fitted_candidates.items():
        probabilities, _ = _attack_probabilities(model, X_validation)
        threshold_result = select_threshold(
            probabilities,
            y_validation.to_numpy(),
            minimum_recall=minimum_recall,
            candidate_thresholds=DEFAULT_CANDIDATE_THRESHOLDS,
        )
        selected_validation_metrics = threshold_result.get("selected_validation_metrics")
        validation_results[name] = {
            "feasible": threshold_result["feasible"],
            "selected_threshold": threshold_result["selected_threshold"],
            "selection_reason": threshold_result["selection_reason"],
            "validation_metrics": selected_validation_metrics,
            "max_candidate_validation_recall": max(
                metric["recall"] for metric in threshold_result["candidates"]
            ),
            "threshold_candidates_evaluated": len(threshold_result["candidate_thresholds"]),
        }

    winner_name = select_winning_candidate(validation_results)
    winner_model = fitted_candidates.get(winner_name) if winner_name is not None else None
    winner_threshold = (
        validation_results[winner_name]["selected_threshold"]
        if winner_name is not None
        else None
    )

    # Persist the selected pipeline before any test labels or metrics are read.
    selection_metadata = {
        "selected_model": winner_name,
        "selected_threshold": winner_threshold,
        "minimum_validation_recall": float(minimum_recall),
        "positive_class": "attack",
        "feature_names": feature_names,
        "dataset_sha256": metadata["dataset_sha256"],
        "split_manifest_sha256": hashlib.sha256(
            (artifact_path / "split_indices.json").read_bytes()
        ).hexdigest(),
        "random_state": random_state,
        "training_rows": len(train_positions),
        "validation_rows": len(validation_positions),
        "test_rows": len(split_indices["test"]),
        "selection_data": "validation only",
        "model_configuration": (
            metadata["training_configuration"]
            if winner_name == "random_forest_baseline"
            else {
                "extra_trees": {
                    "classifier": "ExtraTreesClassifier",
                    "n_estimators": 100,
                    "max_depth": 12,
                    "min_samples_leaf": 2,
                    "class_weight": "balanced",
                    "random_state": random_state,
                    "n_jobs": 1,
                },
                "logistic_regression": {
                    "classifier": "LogisticRegression",
                    "C": 1.0,
                    "class_weight": "balanced",
                    "max_iter": 500,
                    "random_state": random_state,
                    "scaling": "StandardScaler",
                },
            }.get(winner_name)
        ),
        "model_retrained_from_baseline": winner_name != "random_forest_baseline"
        if winner_name is not None
        else False,
    }
    selected_model_path = artifact_path / SELECTED_MODEL_FILENAME
    selected_metadata_path = artifact_path / SELECTED_METADATA_FILENAME
    if winner_model is not None:
        joblib.dump(winner_model, selected_model_path)
        _write_json(selected_metadata_path, selection_metadata)

    # Test rows/labels are not accessed until the selected model and threshold are frozen.
    test_positions = split_indices["test"]
    test_row_indices_sha256 = hashlib.sha256(
        json.dumps(test_positions, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    X_test = frame.iloc[test_positions].loc[:, feature_names]
    y_test = frame.iloc[test_positions][LABEL_COLUMN]
    baseline_probabilities, baseline_test_seconds = _attack_probabilities(
        baseline_model, X_test
    )
    baseline_threshold = float(metadata["training_configuration"]["threshold"])
    baseline_test_metrics = evaluate_binary_classifier(
        y_test.to_numpy(), baseline_probabilities, threshold=baseline_threshold
    )
    baseline_test_metrics["local_inference_benchmark"] = {
        "elapsed_seconds": baseline_test_seconds,
        "rows": len(X_test),
        "note": "One local batch measurement; not a universal performance claim.",
    }

    stage3_threshold = _load_stage3_threshold(
        artifact_path, metadata["dataset_sha256"]
    )
    stage3_test_metrics = (
        evaluate_binary_classifier(
            y_test.to_numpy(), baseline_probabilities, threshold=stage3_threshold
        )
        if stage3_threshold is not None
        else None
    )

    diagnosis, diagnosis_rows, feature_diagnostics = create_baseline_diagnostics(
        X_test,
        y_test,
        baseline_probabilities,
        threshold=baseline_threshold,
    )
    diagnosis["dataset_sha256"] = metadata["dataset_sha256"]
    diagnosis["split_manifest_sha256"] = selection_metadata["split_manifest_sha256"]
    if len(diagnosis_rows):
        diagnosis_rows["original_row_index"] = test_positions
    diagnosis_path = artifact_path / DIAGNOSIS_FILENAME
    row_diagnostics_path = artifact_path / DIAGNOSTICS_CSV_FILENAME
    feature_diagnostics_path = artifact_path / FEATURE_DIAGNOSTICS_CSV_FILENAME
    _write_json(diagnosis_path, diagnosis)
    diagnosis_rows.to_csv(row_diagnostics_path, index=False)
    feature_diagnostics.to_csv(feature_diagnostics_path, index=False)

    selected_test_metrics = None
    if winner_model is not None and winner_name is not None:
        if winner_name == "random_forest_baseline":
            selected_probabilities = baseline_probabilities
            selected_inference_seconds = baseline_test_seconds
        else:
            selected_probabilities, selected_inference_seconds = _attack_probabilities(
                winner_model, X_test
            )
        selected_test_metrics = evaluate_binary_classifier(
            y_test.to_numpy(),
            selected_probabilities,
            threshold=float(winner_threshold),
        )
        selected_test_metrics["local_inference_benchmark"] = {
            "elapsed_seconds": selected_inference_seconds,
            "rows": len(X_test),
            "note": "One local batch measurement; not a universal performance claim.",
        }

    comparison = {
        "experiment": "Stage 4 validation-only model and threshold comparison",
        "data_note": "Synthetic demonstration data; not real-world NIDS evidence.",
        "limitations": [
            "Synthetic data is not evidence of real-world intrusion detection performance.",
            "Validation recall feasibility does not guarantee test recall feasibility.",
            "Feature group differences are exploratory associations, not causal explanations.",
        ],
        "configuration": {
            "dataset_file": source.name,
            "dataset_sha256": metadata["dataset_sha256"],
            "split_manifest_sha256": selection_metadata["split_manifest_sha256"],
            "random_state": random_state,
            "minimum_validation_recall": float(minimum_recall),
            "positive_class": "attack",
            "test_set_used_for_selection": False,
            "baseline_refitted": False,
            "candidate_thresholds": list(DEFAULT_CANDIDATE_THRESHOLDS),
            "threshold_selection_tie_breaker": (
                "lowest candidate validation FPR, then highest precision, then highest threshold"
            ),
            "model_selection_tie_breaker": (
                "lowest validation FPR, then highest precision, then highest recall, "
                "then ascending model name"
            ),
            "threshold_policy_difference_from_stage3": (
                "Stage 3 selects a threshold for the fixed baseline model. Stage 4 "
                "selects both a candidate model and its threshold using validation-only "
                "FPR/precision/recall ordering."
            ),
            "training_rows": len(train_positions),
            "validation_rows": len(validation_positions),
            "test_rows": len(test_positions),
            "test_row_indices_sha256": test_row_indices_sha256,
            "same_test_rows_used_for_all_final_comparisons": True,
        },
        "candidate_validation_results": validation_results,
        "candidate_model_configurations": {
            "random_forest_baseline": metadata["training_configuration"],
            "extra_trees": {
                "classifier": "ExtraTreesClassifier",
                "n_estimators": 100,
                "max_depth": 12,
                "min_samples_leaf": 2,
                "class_weight": "balanced",
                "random_state": random_state,
                "n_jobs": 1,
            },
            "logistic_regression": {
                "classifier": "LogisticRegression",
                "C": 1.0,
                "class_weight": "balanced",
                "max_iter": 500,
                "random_state": random_state,
                "scaling": "StandardScaler after training-fitted median imputation",
            },
        },
        "selected_model": winner_name,
        "selected_threshold": winner_threshold,
        "selection_feasible": winner_name is not None,
        "selection_outcome": (
            f"Selected {winner_name} using validation data only."
            if winner_name is not None
            else "No candidate met the minimum validation recall target."
        ),
        "baseline_test_metrics_threshold_0_50": baseline_test_metrics,
        "stage3_baseline_threshold_test_metrics": stage3_test_metrics,
        "selected_candidate_test_metrics": selected_test_metrics,
        "test_comparison_selected_candidate_vs_baseline": (
            _metric_change(selected_test_metrics, baseline_test_metrics)
            if selected_test_metrics is not None
            else None
        ),
        "test_comparison_vs_stage3_threshold_policy": (
            _metric_change(selected_test_metrics, stage3_test_metrics)
            if selected_test_metrics is not None and stage3_test_metrics is not None
            else None
        ),
        "baseline_diagnosis_file": DIAGNOSIS_FILENAME,
        "baseline_error_rows_file": DIAGNOSTICS_CSV_FILENAME,
        "baseline_feature_diagnostics_file": FEATURE_DIAGNOSTICS_CSV_FILENAME,
        "selected_model_file": SELECTED_MODEL_FILENAME if winner_name is not None else None,
        "selected_model_metadata_file": (
            SELECTED_METADATA_FILENAME if winner_name is not None else None
        ),
    }
    report_path = artifact_path / COMPARISON_FILENAME
    _write_json(report_path, comparison)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare validation-selected FPGuard classifier pipelines."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--minimum-recall", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args()
    report = compare_saved_models(
        args.dataset,
        args.artifacts,
        minimum_recall=args.minimum_recall,
        random_state=args.seed,
    )
    selected_metrics = report["selected_candidate_test_metrics"]
    print(f"Selection feasible: {report['selection_feasible']}")
    print(f"Selected model: {report['selected_model']}")
    print(f"Selected threshold: {report['selected_threshold']}")
    if selected_metrics is not None:
        comparison = report["test_comparison_selected_candidate_vs_baseline"]
        print(
            "Test FPR: baseline "
            f"{report['baseline_test_metrics_threshold_0_50']['false_positive_rate']:.4f}, "
            f"selected {selected_metrics['false_positive_rate']:.4f}"
        )
        print(
            "Test recall: baseline "
            f"{report['baseline_test_metrics_threshold_0_50']['recall']:.4f}, "
            f"selected {selected_metrics['recall']:.4f}"
        )
        print(
            "Test FPR change (percentage points): "
            f"{comparison['absolute_fpr_change_percentage_points_candidate_minus_baseline']:.2f}"
        )
    print(f"Report written to {Path(args.artifacts) / COMPARISON_FILENAME}")


if __name__ == "__main__":
    main()
