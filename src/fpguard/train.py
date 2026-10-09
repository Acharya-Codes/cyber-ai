"""Training and persistence for the baseline Random Forest pipeline."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
from fpguard.preprocessing import build_preprocessor

DEFAULT_ARTIFACT_DIR = Path("artifacts")
MODEL_FILENAME = "baseline_model.joblib"
METADATA_FILENAME = "baseline_metadata.json"


def _validate_training_inputs(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    if not isinstance(X_train, pd.DataFrame) or X_train.empty:
        raise ValueError("X_train must be a non-empty pandas DataFrame")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series")
    if len(X_train) != len(y_train) or not X_train.index.equals(y_train.index):
        raise ValueError("X_train and y_train must have matching row indices")
    invalid_labels = sorted(set(y_train.dropna().unique()) - set(LABELS))
    if y_train.isna().any() or invalid_labels:
        raise ValueError(f"y_train labels must be drawn from {LABELS}; found {invalid_labels}")
    if set(y_train.unique()) != set(LABELS):
        raise ValueError(f"y_train must contain both classes: {LABELS}")
    if X_train.columns.has_duplicates:
        raise ValueError("Training features contain duplicate column names")
    non_numeric = [
        column
        for column in X_train
        if not pd.api.types.is_numeric_dtype(X_train[column])
    ]
    if non_numeric:
        raise TypeError(f"Feature column(s) must be numeric: {', '.join(non_numeric)}")


def fit_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Fit preprocessing and a bounded Random Forest using training data only."""
    _validate_training_inputs(X_train, y_train)
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(tuple(X_train.columns))),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=64,
                    max_depth=12,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def _read_experiment_dataset(path: Path) -> pd.DataFrame:
    frame = load_csv(path, required_columns=(*FEATURE_COLUMNS, LABEL_COLUMN))
    frame = frame.reset_index(drop=True)
    validate_dataset(frame)
    unexpected = [column for column in frame if column not in (*FEATURE_COLUMNS, LABEL_COLUMN)]
    if unexpected:
        raise ValueError(f"Unsupported extra dataset column(s): {', '.join(unexpected)}")
    return frame.loc[:, [*FEATURE_COLUMNS, LABEL_COLUMN]]


def train_saved_baseline(
    dataset_path: str | Path = DEFAULT_DATA_PATH,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Train from the previously generated CSV, save model, metadata, and splits."""
    source = Path(dataset_path)
    frame = _read_experiment_dataset(source)
    splits = split_dataset(frame, random_state=random_state)
    model = fit_baseline(splits.X_train, splits.y_train, random_state=random_state)

    destination = Path(artifact_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / MODEL_FILENAME
    joblib.dump(model, model_path)
    metadata = {
        "model_file": model_path.name,
        "dataset_file": source.name,
        "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "feature_names": list(splits.X_train.columns),
        "positive_class": "attack",
        "random_state": random_state,
        "split_sizes": {
            "train": len(splits.y_train),
            "validation": len(splits.y_validation),
            "test": len(splits.y_test),
        },
        "training_configuration": {
            "classifier": "RandomForestClassifier",
            "n_estimators": 64,
            "max_depth": 12,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "n_jobs": 1,
            "threshold": 0.5,
        },
        "validation_rows_used_for_training": False,
        "test_rows_used_for_training": False,
        "experiment_note": "Synthetic demonstration data; not real-world IDS evidence.",
    }
    (destination / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "split_indices.json").write_text(
        json.dumps(
            {
                "random_state": random_state,
                "dataset_sha256": metadata["dataset_sha256"],
                "indices": {
                    "train": splits.train_indices,
                    "validation": splits.validation_indices,
                    "test": splits.test_indices,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the FPGuard baseline Random Forest.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args()
    train_saved_baseline(args.dataset, args.artifacts, random_state=args.seed)
    print(f"Saved baseline pipeline and metadata in {args.artifacts}")


if __name__ == "__main__":
    main()
