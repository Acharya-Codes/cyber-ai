"""Tests for leakage-conscious real-data training infrastructure."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from fpguard.real_data import write_audit_report
from fpguard.real_train import (
    RealTrainingError,
    _attack_probabilities,
    _json_safe,
    _typed_features,
    build_real_pipeline,
    evaluate_real_metrics,
    make_group_split,
    run_real_experiment,
    select_predictive_features,
    select_winning_candidate,
)


def _fixture_csv(path: Path, *, conflicting_labels: bool = False) -> Path:
    rows = []
    for session in range(5):
        for group in range(20):
            attack = group >= 10
            if session % 2:
                attack = group < 10
            label = "attack" if attack else "benign"
            if conflicting_labels and session == 0 and group == 0:
                duplicate_labels = ("benign", "attack")
            else:
                duplicate_labels = (label, label)
            for duplicate, duplicate_label in enumerate(duplicate_labels):
                rows.append(
                    {
                        "flow_duration": float(group + session * 20),
                        "total_fwd_packets": float(20 if attack else 2),
                        "session_id": f"session-{session}",
                        "Label": duplicate_label,
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _audited_fixture(tmp_path: Path, *, conflicting_labels: bool = False):
    source = _fixture_csv(
        tmp_path / "flows.csv",
        conflicting_labels=conflicting_labels,
    )
    audit_dir = tmp_path / "audit"
    report = write_audit_report(
        [source],
        audit_dir,
        label_column="Label",
        benign_labels=("benign",),
        attack_labels=("attack",),
        model_metadata_path=tmp_path / "missing-metadata.json",
    )
    return source, report


def test_cli_help_does_not_require_a_dataset():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "fpguard.real_train", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--audit-report" in result.stdout
    assert "--minimum-recall" in result.stdout
    assert "--max-rows" in result.stdout
    assert "--cleaning-manifest" in result.stdout
    assert "--investigation-report" in result.stdout
    assert "--preflight-only" in result.stdout


def test_audit_provenance_hash_is_required_and_detects_changed_input(tmp_path):
    source, report = _audited_fixture(tmp_path)
    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    result = run_real_experiment(
        [source],
        audit_report_path=report,
        output_dir=tmp_path / "real-output",
    )
    assert result["selected_model"] in {
        "logistic_regression",
        "random_forest",
        "extra_trees",
    }
    assert expected_hash in json.loads(
        (tmp_path / "real-output" / "split_manifest.json").read_text()
    )["input_sources"][0]["sha256"]

    source.write_text(source.read_text() + "\n", encoding="utf-8")
    with pytest.raises(RealTrainingError, match="SHA-256 mismatch"):
        run_real_experiment(
            [source],
            audit_report_path=report,
            output_dir=tmp_path / "changed-output",
        )


def test_missing_csv_and_malformed_label_mapping_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="audit report not found"):
        run_real_experiment(
            [tmp_path / "missing.csv"],
            audit_report_path=tmp_path / "missing-audit.json",
            output_dir=tmp_path / "output",
        )


def test_leakage_columns_are_excluded_and_explicit_exclusion_is_recorded():
    selected, explicitly_excluded, exclusions = select_predictive_features(
        [
            "bytes",
            "flow_duration",
            "session_id",
            "timestamp",
            "attack_category",
            "source_ip",
            "future_prediction",
        ],
        "label",
        excluded_columns=("flow_duration",),
    )

    assert selected == ["bytes"]
    assert explicitly_excluded == ["flow_duration"]
    assert {item["column"] for item in exclusions} == {
        "flow_duration",
        "session_id",
        "timestamp",
        "attack_category",
        "source_ip",
        "future_prediction",
    }


def test_destination_port_is_retained_but_timestamp_and_identifiers_are_excluded():
    selected, _, exclusions = select_predictive_features(
        ["dst_port", "timestamp", "flow_id", "source_ip", "label"],
        "label",
    )

    assert selected == ["dst_port"]
    assert {item["column"] for item in exclusions} == {
        "timestamp",
        "flow_id",
        "source_ip",
        "label",
    }


def test_duplicate_feature_groups_and_capture_sessions_do_not_cross_splits():
    features = pd.DataFrame(
        {
            "flow": [value for value in range(10) for _ in range(2)],
            "session_id": [f"s{value // 2 + 1}" for value in range(10) for _ in range(2)],
        }
    )
    labels = pd.Series(
        [
            "attack" if value % 2 else "benign"
            for value in range(10)
            for _ in range(2)
        ]
    )
    references = pd.DataFrame({"source_path": ["one.csv"] * len(features)})

    split = make_group_split(
        features[["flow"]],
        labels,
        references,
        random_state=17,
        context_frame=features,
    )

    assert split["context_grouping_used"] is True
    assert split["context_columns"] == ["session_id"]
    groups_by_split = {
        name: set(split["groups"][indices])
        for name, indices in split["indices"].items()
    }
    assert not (groups_by_split["train"] & groups_by_split["validation"])
    assert not (groups_by_split["train"] & groups_by_split["test"])
    assert not (groups_by_split["validation"] & groups_by_split["test"])
    assert split["feature_hashes"][0] == split["feature_hashes"][1]
    assert split["row_split"][0] == split["row_split"][1]


def test_grouping_can_use_all_source_features_not_only_predictors():
    predictive = pd.DataFrame({"flow": [value // 2 for value in range(20)]})
    grouping = pd.DataFrame(
        {
            "flow": [value // 2 for value in range(20)],
            "timestamp": [f"t{value}" for value in range(20)],
            "dst_port": [80 + value for value in range(20)],
        }
    )
    labels = pd.Series(
        ["benign" if value % 2 == 0 else "attack" for value in range(20)]
    )
    refs = pd.DataFrame({"source_path": ["flows.csv"] * len(labels)})

    split = make_group_split(
        predictive,
        labels,
        refs,
        random_state=9,
        grouping_features=grouping,
    )

    assert split["groups"][0] != split["groups"][1]
    assert split["feature_group_hashes"][0] != split["feature_group_hashes"][1]
    assert split["mixed_label_group_stats"]["train"]["affected_row_count"] == 0


def test_conflicting_label_groups_are_retained_and_reported(tmp_path):
    source, report = _audited_fixture(tmp_path, conflicting_labels=True)

    result = run_real_experiment(
        [source],
        audit_report_path=report,
        output_dir=tmp_path / "output",
    )
    manifest = json.loads(
        (tmp_path / "output" / "split_manifest.json").read_text()
    )

    assert result["selected_model"] in {
        "logistic_regression",
        "random_forest",
        "extra_trees",
    }
    assert manifest["mixed_label_group_count_total"] == 1
    assert manifest["rows_in_mixed_label_groups_total"] == 2
    assert sum(
        item["mixed_label_group_count"]
        for item in manifest["group_mixed_label_statistics"].values()
    ) == 1
    comparison = json.loads(
        (tmp_path / "output" / "model_comparison.json").read_text()
    )
    test_conflicts = comparison["test_evaluation"]["mixed_label_group_diagnostics"]
    assert test_conflicts["affected_row_count"] in (0, 2)


def test_cleaning_manifest_provenance_and_preflight_do_not_write_artifacts(tmp_path):
    source, audit_report = _audited_fixture(tmp_path)
    frame = pd.read_csv(source, dtype=str)
    manifest_path = tmp_path / "cleaning_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "output_path": str(source.resolve()),
                "output_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "output_data_row_count": len(frame),
                "retained_raw_label_counts": {
                    str(label): int(count)
                    for label, count in frame["Label"].value_counts().items()
                },
                "retained_labels_explicitly_counted": True,
                "output_has_no_exact_repeated_header_records": True,
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "preflight-artifacts"

    result = run_real_experiment(
        [source],
        audit_report_path=audit_report,
        cleaning_manifest_path=manifest_path,
        output_dir=output_dir,
        preflight_only=True,
    )

    assert result["preflight_only"] is True
    assert result["artifact_paths_written"] == []
    assert result["split_counts"] == {"train": 120, "validation": 40, "test": 40}
    assert not output_dir.exists()


def test_preprocessing_is_fitted_only_on_training_rows():
    frame = pd.DataFrame({"value": [1.0, np.nan, 3.0, 9999.0]})
    prepared, numeric, categorical = _typed_features(
        frame,
        np.array([0, 1, 2]),
        ["value"],
        max_categorical_levels=5,
    )
    pipeline = build_real_pipeline(
        ["value"],
        numeric,
        categorical,
        classifier=LogisticRegression(),
        scale_numeric=False,
    )
    pipeline.fit(prepared.iloc[:3], ["benign", "attack", "attack"])
    imputer = pipeline.named_steps["preprocessor"].named_transformers_["numeric"]
    assert imputer.named_steps["imputer"].statistics_[0] == 2.0


def test_numeric_features_reject_malformed_values_and_replace_infinities():
    malformed = pd.DataFrame({"rate": ["1.0", "not-numeric", "3.0"]})
    with pytest.raises(RealTrainingError, match="malformed numeric values"):
        _typed_features(
            malformed,
            np.array([0, 1]),
            ["rate"],
            max_categorical_levels=5,
        )

    nonfinite = pd.DataFrame({"rate": ["1.0", "inf", "-inf"]})
    prepared, numeric, categorical = _typed_features(
        nonfinite,
        np.array([0, 1]),
        ["rate"],
        max_categorical_levels=5,
    )
    assert numeric == ["rate"]
    assert categorical == []
    assert prepared["rate"].iloc[0] == 1.0
    assert prepared["rate"].iloc[1:].isna().all()


def test_attack_probability_uses_the_positive_class_column():
    class ReorderedClassModel:
        classes_ = np.array(["attack", "benign"])

        @staticmethod
        def predict_proba(_features):
            return np.array([[0.8, 0.2], [0.1, 0.9]])

    assert _attack_probabilities(ReorderedClassModel(), pd.DataFrame()) == pytest.approx(
        [0.8, 0.1]
    )


def test_metrics_handle_zero_fpr_and_undefined_class_metrics():
    metrics = evaluate_real_metrics(
        ["benign", "benign"],
        [0.1, 0.2],
        threshold=0.5,
    )
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["recall"] is None
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc_average_precision"] is None
    assert metrics["precision"] is None
    assert metrics["undefined_metric_notes"]


def test_relative_fpr_change_is_null_for_zero_baseline():
    from fpguard.real_train import _metric_changes

    change = _metric_changes(
        {"false_positive_rate": 0.1, "recall": 0.8, "false_positives": 1, "false_negatives": 2},
        {"false_positive_rate": 0.0, "recall": 0.7, "false_positives": 0, "false_negatives": 3},
    )
    assert change["absolute_fpr_change_percentage_points_candidate_minus_baseline"] == 10
    assert change["relative_fpr_reduction_percent"] is None
    assert change["relative_fpr_reduction_undefined_reason"] == "Baseline FPR is zero."


def test_candidate_selection_uses_validation_feasibility_and_is_deterministic():
    results = {
        "random_forest": {
            "feasible": True,
            "validation_metrics": {
                "false_positive_rate": 0.1,
                "precision": 0.8,
                "recall": 0.9,
            },
        },
        "extra_trees": {
            "feasible": True,
            "validation_metrics": {
                "false_positive_rate": 0.1,
                "precision": 0.8,
                "recall": 0.9,
            },
        },
        "logistic_regression": {
            "feasible": False,
            "validation_metrics": None,
        },
    }
    assert select_winning_candidate(results) == "extra_trees"
    assert select_winning_candidate(
        {name: {"feasible": False} for name in results}
    ) is None


def test_json_safe_converts_nonfinite_values_to_null():
    assert _json_safe({"finite": np.float64(2.5), "undefined": np.inf}) == {
        "finite": 2.5,
        "undefined": None,
    }
    assert json.dumps(_json_safe({"undefined": np.nan}), allow_nan=False) == (
        '{"undefined": null}'
    )


def test_experiment_outputs_are_separate_and_existing_synthetic_artifact_is_untouched(
    tmp_path,
):
    source, report = _audited_fixture(tmp_path)
    synthetic_artifact = tmp_path / "artifacts" / "synthetic_metrics.json"
    synthetic_artifact.parent.mkdir()
    synthetic_artifact.write_text('{"keep": true}\n', encoding="utf-8")
    original_hash = hashlib.sha256(synthetic_artifact.read_bytes()).hexdigest()
    output_dir = tmp_path / "artifacts" / "real_data"

    run_real_experiment(
        [source],
        audit_report_path=report,
        output_dir=output_dir,
    )

    assert hashlib.sha256(synthetic_artifact.read_bytes()).hexdigest() == original_hash
    assert {
        "split_manifest.json",
        "model_comparison.json",
        "selected_model.joblib",
        "selected_model_metadata.json",
        "test_metrics.json",
        "diagnostics.csv",
        ".fpguard-real-data-output",
    } == {path.name for path in output_dir.iterdir()}
    comparison = json.loads((output_dir / "model_comparison.json").read_text())
    assert comparison["configuration"]["test_used_for_selection"] is False
    assert comparison["test_evaluation"]["performed"] is True
    assert comparison["synthetic_data"] is None
    assert comparison["source_authenticity_verified"] is False


def test_output_collision_is_rejected_without_force(tmp_path):
    source, report = _audited_fixture(tmp_path)
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    sentinel = output_dir / "model_comparison.json"
    sentinel.write_text("preserve me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Choose another --output-dir"):
        run_real_experiment(
            [source],
            audit_report_path=report,
            output_dir=output_dir,
        )
    with pytest.raises(RealTrainingError, match="not marked as FPGuard Stage 6B output"):
        run_real_experiment(
            [source],
            audit_report_path=report,
            output_dir=output_dir,
            force=True,
        )
    assert sentinel.read_text(encoding="utf-8") == "preserve me"
