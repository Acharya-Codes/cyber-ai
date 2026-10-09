"""Focused tests for the synthetic baseline experiment."""

import json

import numpy as np
import pandas as pd
import pytest

from fpguard.data import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    LABELS,
    generate_demo_files,
    generate_synthetic_dataset,
    load_csv,
    split_dataset,
    validate_dataset,
)
from fpguard.evaluate import evaluate_binary_classifier, evaluate_saved_baseline
from fpguard.predict import predict_network_flows
from fpguard.preprocessing import build_preprocessor
from fpguard.train import MODEL_FILENAME, fit_baseline, train_saved_baseline


def test_synthetic_generation_is_deterministic_and_has_safe_feature_schema():
    first = generate_synthetic_dataset(400, random_state=17)
    second = generate_synthetic_dataset(400, random_state=17)

    pd.testing.assert_frame_equal(first, second)
    assert set(first[LABEL_COLUMN].unique()) == set(LABELS)
    assert set(FEATURE_COLUMNS).issubset(first.columns)
    assert LABEL_COLUMN not in FEATURE_COLUMNS
    assert list(first.columns) == [*FEATURE_COLUMNS, LABEL_COLUMN]


def test_generated_csv_has_no_row_identifier(tmp_path):
    dataset_path = generate_demo_files(tmp_path / "demo.csv", n_samples=100)
    saved_frame = pd.read_csv(dataset_path)

    assert list(saved_frame.columns) == [*FEATURE_COLUMNS, LABEL_COLUMN]
    assert (tmp_path / "split_indices.json").is_file()


def test_split_dataset_is_stratified_disjoint_and_60_20_20():
    splits = split_dataset(generate_synthetic_dataset(500))

    assert (len(splits.y_train), len(splits.y_validation), len(splits.y_test)) == (
        300,
        100,
        100,
    )
    assert set(splits.y_train.unique()) == set(LABELS)
    assert set(splits.y_validation.unique()) == set(LABELS)
    assert set(splits.y_test.unique()) == set(LABELS)
    train, validation, test = map(
        set, (splits.train_indices, splits.validation_indices, splits.test_indices)
    )
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert LABEL_COLUMN not in splits.X_train


def test_split_rejects_non_unique_row_indices():
    frame = generate_synthetic_dataset(100)
    frame.index = [0] * len(frame)

    with pytest.raises(ValueError, match="indices must be unique"):
        split_dataset(frame)


@pytest.mark.parametrize("malformation", ["label", "feature_type", "missing_feature"])
def test_dataset_validation_rejects_malformed_input(malformation):
    frame = generate_synthetic_dataset(100)
    if malformation == "label":
        frame.loc[0, LABEL_COLUMN] = "unknown"
    elif malformation == "feature_type":
        frame["packet_length_mean"] = "not numeric"
    else:
        frame = frame.drop(columns=["packet_length_std"])

    with pytest.raises((ValueError, TypeError)):
        validate_dataset(frame)


def test_numeric_preprocessor_imputes_missing_and_infinite_values():
    features = pd.DataFrame({"f1": [1.0, np.inf, np.nan], "f2": [4.0, 6.0, 8.0]})
    processed = build_preprocessor(tuple(features.columns)).fit_transform(features)

    assert np.isfinite(processed).all()
    assert processed[1, 0] == pytest.approx(1.0)
    assert processed[2, 0] == pytest.approx(1.0)


def test_dataset_with_missing_numeric_values_can_be_preprocessed_after_split():
    frame = generate_synthetic_dataset(100)
    frame.loc[frame.index[0], "packet_length_mean"] = np.inf
    frame.loc[frame.index[1], "packet_length_std"] = np.nan

    splits = split_dataset(frame)
    transformer = build_preprocessor(tuple(splits.X_train.columns))
    transformed = transformer.fit_transform(splits.X_train)

    assert np.isfinite(transformed).all()


def test_numeric_preprocessor_rejects_categorical_features():
    with pytest.raises(TypeError, match="Categorical features are not supported"):
        build_preprocessor(("duration",), ("protocol",))


def test_csv_loader_retains_required_schema_checks(tmp_path):
    csv_path = tmp_path / "missing.csv"
    pd.DataFrame({"duration": [1.0]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing required column"):
        load_csv(csv_path, required_columns=("duration", "label"))


def test_binary_metrics_match_hand_constructed_confusion_matrix():
    metrics = evaluate_binary_classifier(
        ["benign", "benign", "attack", "attack"],
        [0.1, 0.8, 0.7, 0.2],
        threshold=0.5,
    )

    assert metrics["class_order"] == ["benign", "attack"]
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert (
        metrics["true_positives"],
        metrics["true_negatives"],
        metrics["false_positives"],
        metrics["false_negatives"],
    ) == (1, 1, 1, 1)
    assert metrics["false_positive_rate"] == pytest.approx(0.5)
    assert metrics["false_negative_rate"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1_score"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["roc_auc"] == pytest.approx(0.5)


def test_metrics_report_undefined_zero_denominators_as_null():
    benign_only = evaluate_binary_classifier(["benign", "benign"], [0.1, 0.2])
    attack_only = evaluate_binary_classifier(["attack", "attack"], [0.8, 0.9])

    assert benign_only["false_negative_rate"] is None
    assert benign_only["roc_auc"] is None
    assert benign_only["precision"] == 0.0
    assert attack_only["false_positive_rate"] is None
    assert attack_only["roc_auc"] is None


def test_metrics_reject_invalid_probabilities():
    with pytest.raises(ValueError, match="Probabilities must be finite"):
        evaluate_binary_classifier(["benign"], [np.inf])


def test_training_persistence_evaluation_and_prediction_round_trip(tmp_path):
    dataset_path = generate_demo_files(
        tmp_path / "demo.csv", n_samples=400, random_state=29
    )
    artifact_dir = tmp_path / "artifacts"

    model = train_saved_baseline(dataset_path, artifact_dir, random_state=29)
    assert (artifact_dir / MODEL_FILENAME).is_file()
    assert (artifact_dir / "baseline_metadata.json").is_file()
    assert (artifact_dir / "split_indices.json").is_file()
    assert set(model.classes_) == set(LABELS)

    result = evaluate_saved_baseline(dataset_path, artifact_dir, random_state=29)
    assert (artifact_dir / "metrics.json").is_file()
    saved_metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved_metrics == result
    assert result["test_baseline"]["confusion_matrix"]
    assert result["test_baseline"]["local_inference_benchmark"]["rows"] == 80

    features = generate_synthetic_dataset(40, random_state=31).drop(columns=[LABEL_COLUMN])
    expected_order = list(model.feature_names_in_)
    reordered = features.loc[:, list(reversed(expected_order))].iloc[:8]
    labels_first, probabilities_first = predict_network_flows(
        reordered, artifact_dir / MODEL_FILENAME
    )
    labels_second, probabilities_second = predict_network_flows(
        reordered, artifact_dir / MODEL_FILENAME
    )

    pd.testing.assert_series_equal(labels_first, labels_second)
    pd.testing.assert_series_equal(probabilities_first, probabilities_second)
    assert labels_first.isin(LABELS).all()
    assert probabilities_first.between(0, 1).all()
    assert labels_first.index.equals(reordered.index)


def test_prediction_validates_missing_and_extra_features(tmp_path):
    dataset_path = generate_demo_files(
        tmp_path / "demo.csv", n_samples=400, random_state=37
    )
    artifact_dir = tmp_path / "artifacts"
    train_saved_baseline(dataset_path, artifact_dir, random_state=37)
    features = generate_synthetic_dataset(40).drop(columns=[LABEL_COLUMN]).iloc[:2]

    with pytest.raises(ValueError, match="missing required feature"):
        predict_network_flows(features.drop(columns=[FEATURE_COLUMNS[0]]), artifact_dir / MODEL_FILENAME)
    with pytest.raises(ValueError, match="unsupported feature"):
        predict_network_flows(
            features.assign(unapproved_extra=1), artifact_dir / MODEL_FILENAME
        )


def test_training_rejects_invalid_labels_and_non_numeric_features():
    features = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]})
    invalid_labels = pd.Series(["benign", "attack", "other", "benign"])
    with pytest.raises(ValueError, match="labels must be drawn"):
        fit_baseline(features, invalid_labels)

    text_features = pd.DataFrame({"value": ["one", "two", "three", "four"]})
    valid_labels = pd.Series(["benign", "attack", "benign", "attack"])
    with pytest.raises(TypeError, match="must be numeric"):
        fit_baseline(text_features, valid_labels)


def test_evaluation_rejects_dataset_changed_after_training(tmp_path):
    dataset_path = generate_demo_files(
        tmp_path / "demo.csv", n_samples=400, random_state=41
    )
    artifact_dir = tmp_path / "artifacts"
    train_saved_baseline(dataset_path, artifact_dir, random_state=41)
    with dataset_path.open("a", encoding="utf-8") as dataset_file:
        dataset_file.write("\n")

    with pytest.raises(ValueError, match="differs from the version used to train"):
        evaluate_saved_baseline(dataset_path, artifact_dir, random_state=41)
