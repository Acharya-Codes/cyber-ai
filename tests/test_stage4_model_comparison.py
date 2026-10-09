"""Fast tests for Stage 4 selection, diagnosis, and artifact boundaries."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import fpguard.compare_models as comparison
from fpguard.data import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    generate_demo_files,
)
from fpguard.evaluate import create_baseline_diagnostics
from fpguard.train import train_saved_baseline


def test_winner_selection_uses_only_validation_fields_and_stable_name_order():
    results = {
        "model_b": {
            "feasible": True,
            "validation_metrics": {
                "false_positive_rate": 0.1,
                "precision": 0.8,
                "recall": 0.9,
            },
            "test_metrics": {"false_positive_rate": 0.0},
        },
        "model_a": {
            "feasible": True,
            "validation_metrics": {
                "false_positive_rate": 0.1,
                "precision": 0.8,
                "recall": 0.9,
            },
            "test_metrics": {"false_positive_rate": 1.0},
        },
    }

    assert comparison.select_winning_candidate(results) == "model_a"
    assert comparison.select_winning_candidate(
        {"model_a": {"feasible": False, "validation_metrics": None}}
    ) is None


def test_metric_change_handles_fpr_reduction_increase_and_zero_denominator():
    baseline = {
        "false_positive_rate": 0.4,
        "false_positives": 4,
        "false_negatives": 5,
        "recall": 0.5,
        "precision": 0.6,
        "f1_score": 0.54,
    }
    candidate = {
        "false_positive_rate": 0.2,
        "false_positives": 2,
        "false_negatives": 3,
        "recall": 0.7,
        "precision": 0.65,
        "f1_score": 0.67,
    }
    improved = comparison._metric_change(candidate, baseline)

    assert improved["absolute_fpr_change_percentage_points_candidate_minus_baseline"] == -20
    assert improved["relative_fpr_reduction_percent"] == pytest.approx(50)
    assert improved["recall_change_percentage_points_candidate_minus_baseline"] == pytest.approx(20)
    assert improved["false_positive_count_change_candidate_minus_baseline"] == -2
    assert improved["false_negative_count_change_candidate_minus_baseline"] == -2
    no_baseline_fpr = comparison._metric_change(
        candidate, {**baseline, "false_positive_rate": 0.0}
    )
    assert no_baseline_fpr["relative_fpr_reduction_percent"] is None
    assert "zero" in no_baseline_fpr["relative_fpr_reduction_undefined_reason"]


def test_json_safe_converts_nonfinite_numpy_values_to_null():
    report = comparison._json_safe(
        {"valid": np.float64(1.5), "undefined": np.float64(np.inf), "count": np.int64(2)}
    )

    assert report == {"valid": 1.5, "undefined": None, "count": 2}
    json.dumps(report, allow_nan=False)


def test_diagnosis_reports_error_types_counts_and_numeric_feature_groups():
    features = pd.DataFrame(
        {"flow": [10.0, np.inf, 20.0, np.nan], "packets": [1, 2, 3, 4]},
        index=[100, 101, 102, 103],
    )
    labels = pd.Series(["benign", "benign", "attack", "attack"], index=features.index)
    diagnosis, rows, feature_summary = create_baseline_diagnostics(
        features, labels, [0.1, 0.8, 0.9, 0.1]
    )

    assert diagnosis["error_type_counts"] == {
        "false_positive": 1,
        "false_negative": 1,
        "true_positive": 1,
        "true_negative": 1,
    }
    assert diagnosis["metrics"]["confusion_matrix"] == [[1, 1], [1, 1]]
    assert rows["original_row_index"].tolist() == [100, 101, 102, 103]
    assert rows["error_type"].tolist() == [
        "true_negative",
        "false_positive",
        "true_positive",
        "false_negative",
    ]
    assert "flow" in set(feature_summary["feature"])
    assert "label" not in set(feature_summary["feature"])
    assert feature_summary["interpretation"].str.contains("not causal").all()


@pytest.fixture(scope="module")
def saved_stage4_experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("stage4")
    dataset_path = generate_demo_files(root / "synthetic.csv", n_samples=200, random_state=73)
    artifact_path = root / "artifacts"
    train_saved_baseline(dataset_path, artifact_path, random_state=73)
    stage3_path = artifact_path / "threshold_experiment.json"
    stage3_path.write_text(
        json.dumps(
            {
                "configuration": {
                    "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest()
                },
                "validation_target_feasible": True,
                "selected_threshold": 0.33,
            }
        ),
        encoding="utf-8",
    )
    return dataset_path, artifact_path, root


def test_stage4_fits_candidates_on_training_rows_and_preserves_prior_artifacts(
    saved_stage4_experiment, monkeypatch
):
    dataset_path, artifact_path, _ = saved_stage4_experiment
    before_hashes = {
        name: hashlib.sha256((artifact_path / name).read_bytes()).hexdigest()
        for name in (
            "baseline_model.joblib",
            "baseline_metadata.json",
            "split_indices.json",
            "threshold_experiment.json",
        )
    }
    original_factory = comparison.make_additional_candidate_pipelines
    pipelines = original_factory(FEATURE_COLUMNS, random_state=73)
    factory_calls = []

    def tracking_factory(feature_names, *, random_state):
        factory_calls.append((tuple(feature_names), random_state))
        return pipelines

    fit_row_indices = []
    original_pipeline_fit = comparison.Pipeline.fit
    original_selector = comparison.select_threshold
    original_probabilities = comparison._attack_probabilities
    selector_labels = []
    probability_row_indices = []

    def tracking_fit(self, X, y=None, **kwargs):
        fit_row_indices.append(tuple(X.index.tolist()))
        return original_pipeline_fit(self, X, y, **kwargs)

    def tracking_selector(probabilities, labels, **kwargs):
        selector_labels.append(np.asarray(labels).copy())
        return original_selector(probabilities, labels, **kwargs)

    def tracking_probabilities(model, features):
        probability_row_indices.append(tuple(features.index.tolist()))
        return original_probabilities(model, features)

    monkeypatch.setattr(comparison, "make_additional_candidate_pipelines", tracking_factory)
    monkeypatch.setattr(comparison.Pipeline, "fit", tracking_fit)
    monkeypatch.setattr(comparison, "select_threshold", tracking_selector)
    monkeypatch.setattr(comparison, "_attack_probabilities", tracking_probabilities)
    report = comparison.compare_saved_models(
        dataset_path, artifact_path, minimum_recall=0.8, random_state=73
    )

    manifest = json.loads((artifact_path / "split_indices.json").read_text(encoding="utf-8"))
    expected_train = tuple(manifest["indices"]["train"])
    assert factory_calls == [(FEATURE_COLUMNS, 73)]
    assert fit_row_indices == [expected_train, expected_train]
    expected_validation = pd.read_csv(dataset_path).iloc[
        manifest["indices"]["validation"]
    ][LABEL_COLUMN]
    expected_test = tuple(manifest["indices"]["test"])
    assert len(selector_labels) == 3
    for labels in selector_labels:
        np.testing.assert_array_equal(labels, expected_validation.to_numpy())
    assert len(report["candidate_validation_results"]) == 3
    assert report["configuration"]["test_set_used_for_selection"] is False
    assert report["configuration"]["same_test_rows_used_for_all_final_comparisons"] is True
    assert probability_row_indices.count(expected_test) >= 1
    assert all(
        row_indices == expected_train
        or row_indices == tuple(manifest["indices"]["validation"])
        or row_indices == expected_test
        for row_indices in fit_row_indices + probability_row_indices
    )
    assert report["selection_feasible"] is True
    assert (artifact_path / comparison.COMPARISON_FILENAME).is_file()
    assert (artifact_path / comparison.DIAGNOSIS_FILENAME).is_file()
    assert (artifact_path / comparison.DIAGNOSTICS_CSV_FILENAME).is_file()
    assert (artifact_path / comparison.FEATURE_DIAGNOSTICS_CSV_FILENAME).is_file()
    assert (artifact_path / comparison.SELECTED_MODEL_FILENAME).is_file()
    assert (artifact_path / comparison.SELECTED_METADATA_FILENAME).is_file()
    comparison_text = (artifact_path / comparison.COMPARISON_FILENAME).read_text(
        encoding="utf-8"
    )
    json.loads(
        comparison_text,
        parse_constant=lambda value: pytest.fail(f"Invalid JSON constant: {value}"),
    )
    assert "NaN" not in comparison_text
    assert "Infinity" not in comparison_text
    assert report["selected_candidate_test_metrics"] is not None
    assert report["configuration"]["test_rows"] > 0
    assert report["baseline_test_metrics_threshold_0_50"]["confusion_matrix"]
    assert (
        report["candidate_validation_results"]["random_forest_baseline"][
            "threshold_candidates_evaluated"
        ]
        == 101
    )
    for filename, expected_hash in before_hashes.items():
        assert hashlib.sha256((artifact_path / filename).read_bytes()).hexdigest() == expected_hash
