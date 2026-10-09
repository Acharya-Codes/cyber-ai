"""Tests for validation-driven threshold selection and test-set reporting."""

import json

import numpy as np
import pandas as pd
import pytest

import fpguard.evaluate as evaluation
from fpguard.data import LABEL_COLUMN, generate_demo_files
from fpguard.evaluate import run_threshold_experiment
from fpguard.threshold import select_threshold
from fpguard.train import train_saved_baseline


def test_threshold_metrics_calculate_attack_recall_and_false_positive_rate():
    result = select_threshold(
        [0.2, 0.8, 0.7, 0.9],
        ["benign", "benign", "attack", "attack"],
        minimum_recall=0.5,
        candidate_thresholds=[0.5],
    )

    metrics = result["selected_validation_metrics"]
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["true_positives"] == 2
    assert metrics["true_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 0
    assert metrics["false_positive_rate"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(1.0)


def test_selector_chooses_lowest_fpr_among_feasible_candidates():
    result = select_threshold(
        [0.2, 0.8, 0.6, 0.9],
        ["benign", "benign", "attack", "attack"],
        minimum_recall=0.5,
        candidate_thresholds=[0.5, 0.7, 0.85],
    )

    assert result["feasible"] is True
    assert result["selected_threshold"] == 0.85
    assert result["selected_validation_metrics"]["false_positive_rate"] == 0.0


def test_selector_tie_breaks_by_precision_then_higher_threshold():
    higher_precision = select_threshold(
        [0.1, 0.8, 0.4, 0.9],
        ["benign", "benign", "attack", "attack"],
        minimum_recall=0.5,
        candidate_thresholds=[0.3, 0.5],
    )
    higher_threshold = select_threshold(
        [0.1, 0.2, 0.8, 0.9],
        ["benign", "benign", "attack", "attack"],
        minimum_recall=1.0,
        candidate_thresholds=[0.3, 0.5],
    )

    assert higher_precision["selected_threshold"] == 0.3
    assert higher_threshold["selected_threshold"] == 0.5


def test_selector_reports_infeasible_without_lowering_recall_target():
    result = select_threshold(
        [0.1, 0.2, 0.3, 0.4],
        ["benign", "benign", "attack", "attack"],
        minimum_recall=0.8,
        candidate_thresholds=[0.9, 1.0],
    )

    assert result["feasible"] is False
    assert result["minimum_recall"] == 0.8
    assert result["selected_threshold"] is None
    assert "No candidate threshold" in result["selection_reason"]


@pytest.mark.parametrize(
    ("labels", "probabilities", "minimum_recall", "candidates", "message"),
    [
        (["benign", "attack"], [-0.1, 1.0], 0.8, [0.5], "between 0 and 1"),
        (["benign", "attack"], [np.nan, 0.5], 0.8, [0.5], "finite"),
        (["benign", "unknown"], [0.1, 0.9], 0.8, [0.5], "only both classes"),
        (["benign", "attack"], [0.1], 0.8, [0.5], "equal length"),
        (["benign", "attack"], [0.1, 0.9], 1.1, [0.5], "minimum_recall"),
        (["benign", "attack"], [0.1, 0.9], 0.8, [], "at least one"),
    ],
)
def test_selector_rejects_invalid_inputs(
    labels, probabilities, minimum_recall, candidates, message
):
    with pytest.raises(ValueError, match=message):
        select_threshold(
            probabilities,
            labels,
            minimum_recall=minimum_recall,
            candidate_thresholds=candidates,
        )


def test_attack_probability_uses_matching_model_class_index():
    class ReversedClassModel:
        classes_ = np.array(["attack", "benign"])

        @staticmethod
        def predict_proba(features):
            return np.array([[0.83, 0.17]])

    probabilities, _ = evaluation._attack_probabilities(
        ReversedClassModel(), pd.DataFrame({"flow": [1]})
    )

    assert probabilities.tolist() == [0.83]


def _make_saved_experiment(tmp_path):
    dataset_path = generate_demo_files(tmp_path / "flows.csv", n_samples=200, random_state=13)
    artifact_path = tmp_path / "artifacts"
    train_saved_baseline(dataset_path, artifact_path, random_state=13)
    return dataset_path, artifact_path


def test_experiment_selects_from_validation_and_reuses_identical_test_batch(
    tmp_path, monkeypatch
):
    dataset_path, artifact_path = _make_saved_experiment(tmp_path)
    saved_selector = evaluation.select_threshold
    selector_inputs = []
    metric_inputs = []

    def capture_selector(probabilities, labels, **kwargs):
        selector_inputs.append(np.asarray(labels).copy())
        return saved_selector(probabilities, labels, **kwargs)

    saved_metrics_function = evaluation.evaluate_binary_classifier

    def capture_metrics(labels, probabilities, *, threshold):
        metric_inputs.append((np.asarray(labels).copy(), np.asarray(probabilities).copy()))
        return saved_metrics_function(labels, probabilities, threshold=threshold)

    monkeypatch.setattr(evaluation, "select_threshold", capture_selector)
    monkeypatch.setattr(evaluation, "evaluate_binary_classifier", capture_metrics)

    report = run_threshold_experiment(dataset_path, artifact_path, minimum_recall=0.8)

    manifest = json.loads((artifact_path / "split_indices.json").read_text(encoding="utf-8"))
    saved_data = pd.read_csv(dataset_path)
    expected_validation = saved_data.iloc[manifest["indices"]["validation"]][LABEL_COLUMN]
    expected_test = saved_data.iloc[manifest["indices"]["test"]][LABEL_COLUMN]
    assert len(selector_inputs) == 1
    np.testing.assert_array_equal(selector_inputs[0], expected_validation.to_numpy())
    assert report["configuration"]["saved_split_manifest_used"] is True
    assert report["configuration"]["model_retrained"] is False
    assert len(metric_inputs) == 3
    np.testing.assert_array_equal(metric_inputs[0][0], expected_validation.to_numpy())
    np.testing.assert_array_equal(metric_inputs[1][0], expected_test.to_numpy())
    np.testing.assert_array_equal(metric_inputs[2][0], expected_test.to_numpy())
    np.testing.assert_array_equal(metric_inputs[1][1], metric_inputs[2][1])
    assert report["test_comparison"]["same_test_probability_vector_used"] is True
    assert report["test_comparison"]["test_rows_shared_by_both_policies"] == len(expected_test)
    assert (artifact_path / "threshold_experiment.json").is_file()


def test_infeasible_experiment_reports_no_alternative_test_result(tmp_path, monkeypatch):
    dataset_path, artifact_path = _make_saved_experiment(tmp_path)
    monkeypatch.setattr(
        evaluation,
        "select_threshold",
        lambda *args, **kwargs: {
            "feasible": False,
            "minimum_recall": kwargs["minimum_recall"],
            "selected_threshold": None,
            "selection_reason": "No feasible candidate.",
            "candidate_thresholds": [1.0],
            "candidates": [],
        },
    )

    report = run_threshold_experiment(dataset_path, artifact_path, minimum_recall=0.8)

    assert report["validation_target_feasible"] is False
    assert report["selected_threshold"] is None
    assert report["validation"]["alternative_policy"] is None
    assert report["test"]["baseline"] is not None
    assert report["test"]["alternative_policy"] is None
    assert report["test_comparison"] is None


def test_report_flags_alternative_that_misses_target_on_test(tmp_path, monkeypatch):
    dataset_path, artifact_path = _make_saved_experiment(tmp_path)
    monkeypatch.setattr(
        evaluation,
        "select_threshold",
        lambda *args, **kwargs: {
            "feasible": True,
            "minimum_recall": kwargs["minimum_recall"],
            "selected_threshold": 0.9,
            "selection_reason": "test fixture",
            "selected_validation_metrics": {"recall": 1.0, "false_positive_rate": 0.0},
            "candidate_thresholds": [0.9],
            "candidates": [],
        },
    )
    probability_calls = 0

    def fixed_probabilities(model, features):
        nonlocal probability_calls
        probability_calls += 1
        return np.full(len(features), 0.1), 0.0

    monkeypatch.setattr(evaluation, "_attack_probabilities", fixed_probabilities)

    report = run_threshold_experiment(dataset_path, artifact_path, minimum_recall=0.8)

    assert probability_calls == 2
    assert report["validation_target_feasible"] is True
    assert report["test"]["alternative_policy"]["recall"] == 0.0
    assert report["test_comparison"]["alternative_meets_minimum_recall_on_test"] is False
