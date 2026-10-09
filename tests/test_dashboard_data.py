"""Tests for local dashboard data validation and artifact access."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from fpguard.dashboard_data import (
    load_json_report,
    load_trusted_pipeline,
    predict_uploaded_flows,
    read_uploaded_flow_csv,
    validate_selected_model_artifact,
    validation_threshold_curve,
)

FEATURES = ["flow_duration", "total_fwd_packets"]


@pytest.fixture
def fitted_model():
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(n_estimators=8, random_state=12, n_jobs=1),
            ),
        ]
    )
    model.fit(
        pd.DataFrame(
            {
                FEATURES[0]: [1.0, 2.0, 10.0, 12.0],
                FEATURES[1]: [1.0, 2.0, 8.0, 9.0],
            }
        ),
        ["benign", "benign", "attack", "attack"],
    )
    return model


def test_uploaded_features_are_ordered_and_metadata_is_excluded():
    content = (
        b"flow_id,total_fwd_packets,label,flow_duration\n"
        b"r2,2,benign,2\n"
        b"r1,9,attack,10\n"
    )

    features, metadata = read_uploaded_flow_csv(content, FEATURES)

    assert list(features.columns) == FEATURES
    assert features["flow_duration"].tolist() == [2, 10]
    assert metadata == ["flow_id", "label"]


def test_uploaded_csv_accepts_missing_and_infinite_numeric_values():
    content = b"flow_duration,total_fwd_packets\n,inf\n2,3\n"

    features, _ = read_uploaded_flow_csv(content, FEATURES)

    assert pd.isna(features.loc[0, "flow_duration"])
    assert np.isinf(features.loc[0, "total_fwd_packets"])


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"flow_duration\n1\n", "Missing required feature"),
        (
            b"flow_duration,total_fwd_packets,extra\n1,2,3\n",
            "Unsupported column",
        ),
        (
            b"flow_duration,total_fwd_packets,flow_duration\n1,2,3\n",
            "Duplicate CSV column",
        ),
        (
            b"flow_duration,total_fwd_packets\nbad,2\n",
            "non-numeric",
        ),
        (
            b"flow_duration,total_fwd_packets,label\n1,2,unknown\n",
            "Optional 'label' values",
        ),
    ],
)
def test_uploaded_csv_rejects_invalid_schema_or_values(content, message):
    with pytest.raises(ValueError, match=message):
        read_uploaded_flow_csv(content, FEATURES)


@pytest.mark.parametrize("content", [b"", b"flow_duration,total_fwd_packets\n"])
def test_uploaded_csv_rejects_empty_input(content):
    with pytest.raises(ValueError, match="empty|no flow rows"):
        read_uploaded_flow_csv(content, FEATURES)


def test_uploaded_prediction_preserves_rows_and_uses_only_feature_columns(fitted_model):
    content = (
        b"record_id,total_fwd_packets,flow_duration,label\n"
        b"second,9,10,attack\n"
        b"first,1,1,benign\n"
    )

    result = predict_uploaded_flows(content, fitted_model, threshold=0.5)

    assert result["record_id"].tolist() == ["second", "first"]
    assert result["predicted_label"].tolist() == ["attack", "benign"]
    assert result["attack_probability"].between(0, 1).all()
    assert result["decision_threshold"].tolist() == [0.5, 0.5]
    assert "label" in result.columns


def test_uploaded_prediction_rejects_invalid_threshold(fitted_model):
    with pytest.raises(ValueError, match="between 0.00 and 1.00"):
        predict_uploaded_flows(
            b"flow_duration,total_fwd_packets\n1,2\n",
            fitted_model,
            threshold=1.1,
        )


def test_json_report_missing_invalid_and_valid(tmp_path):
    missing = tmp_path / "missing.json"
    assert load_json_report(missing) is None

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not read report"):
        load_json_report(invalid)

    valid = tmp_path / "valid.json"
    valid.write_text('{"available": true}', encoding="utf-8")
    assert load_json_report(valid) == {"available": True}


def test_trusted_pipeline_errors_are_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run the documented"):
        load_trusted_pipeline(tmp_path / "missing.joblib")

    invalid = tmp_path / "invalid.joblib"
    invalid.write_bytes(b"not a joblib pipeline")
    with pytest.raises(ValueError, match="invalid or incompatible"):
        load_trusted_pipeline(invalid)


def test_selected_model_artifact_must_match_report_and_saved_data(
    tmp_path, fitted_model
):
    dataset_path = tmp_path / "flows.csv"
    manifest_path = tmp_path / "split_indices.json"
    metadata_path = tmp_path / "metadata.json"
    dataset_path.write_bytes(b"synthetic data")
    manifest_path.write_bytes(b"split manifest")
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    metadata_path.write_text(
        json.dumps(
            {
                "selected_model": "candidate",
                "selected_threshold": 0.4,
                "dataset_sha256": dataset_hash,
                "split_manifest_sha256": manifest_hash,
                "positive_class": "attack",
                "feature_names": FEATURES,
            }
        ),
        encoding="utf-8",
    )
    report = {
        "selection_feasible": True,
        "selected_model": "candidate",
        "selected_threshold": 0.4,
        "configuration": {"dataset_sha256": dataset_hash},
    }

    validate_selected_model_artifact(
        fitted_model,
        metadata_path,
        report,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
    )

    report["selected_threshold"] = 0.5
    with pytest.raises(ValueError, match="threshold does not match"):
        validate_selected_model_artifact(
            fitted_model,
            metadata_path,
            report,
            dataset_path=dataset_path,
            manifest_path=manifest_path,
        )


def test_selected_model_artifact_rejects_changed_split_manifest(
    tmp_path, fitted_model
):
    dataset_path = tmp_path / "flows.csv"
    manifest_path = tmp_path / "split_indices.json"
    metadata_path = tmp_path / "metadata.json"
    dataset_path.write_bytes(b"synthetic data")
    manifest_path.write_bytes(b"split manifest")
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    metadata_path.write_text(
        json.dumps(
            {
                "selected_model": "candidate",
                "selected_threshold": 0.4,
                "dataset_sha256": dataset_hash,
                "split_manifest_sha256": manifest_hash,
                "positive_class": "attack",
                "feature_names": FEATURES,
            }
        ),
        encoding="utf-8",
    )
    manifest_path.write_bytes(b"changed split manifest")
    report = {
        "selection_feasible": True,
        "selected_model": "candidate",
        "selected_threshold": 0.4,
        "configuration": {"dataset_sha256": dataset_hash},
    }

    with pytest.raises(ValueError, match="split manifest"):
        validate_selected_model_artifact(
            fitted_model,
            metadata_path,
            report,
            dataset_path=dataset_path,
            manifest_path=manifest_path,
        )


def test_validation_curve_uses_saved_partition_without_writing_artifacts(
    tmp_path, fitted_model
):
    frame = pd.DataFrame(
        {
            FEATURES[0]: [1.0, 2.0, 10.0, 12.0, 1.5, 11.0],
            FEATURES[1]: [1.0, 2.0, 8.0, 9.0, 1.5, 8.5],
            "label": ["benign", "benign", "attack", "attack", "benign", "attack"],
        }
    )
    dataset_path = tmp_path / "flows.csv"
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    manifest_path = artifact_dir / "split_indices.json"
    report_path = artifact_dir / "threshold_experiment.json"
    frame.to_csv(dataset_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                "indices": {"validation": [0, 2, 4, 5]},
            }
        ),
        encoding="utf-8",
    )
    report_path.write_text('{"preserve": true}', encoding="utf-8")
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (dataset_path, manifest_path, report_path)
    }

    curve = validation_threshold_curve(dataset_path, artifact_dir, model=fitted_model)

    assert len(curve) == 101
    assert curve.iloc[50]["threshold"] == 0.5
    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (dataset_path, manifest_path, report_path)
    }
    assert before == after


def test_validation_curve_rejects_invalid_manifest_indices(tmp_path, fitted_model):
    frame = pd.DataFrame(
        {
            FEATURES[0]: [1.0, 2.0],
            FEATURES[1]: [1.0, 2.0],
            "label": ["benign", "attack"],
        }
    )
    dataset_path = tmp_path / "flows.csv"
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    frame.to_csv(dataset_path, index=False)
    manifest = {
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "indices": {"validation": [0, 2]},
    }
    (artifact_dir / "split_indices.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="validation indices"):
        validation_threshold_curve(dataset_path, artifact_dir, model=fitted_model)
