"""Synthetic network-flow data generation, validation, and stratified splits."""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_RANDOM_STATE = 42
DEFAULT_DATA_PATH = Path("artifacts/synthetic_network_flows.csv")
LABEL_COLUMN = "label"
LABELS = ("benign", "attack")
FEATURE_COLUMNS = (
    "flow_duration",
    "total_fwd_packets",
    "total_backward_packets",
    "total_fwd_bytes",
    "total_backward_bytes",
    "flow_bytes_per_second",
    "packet_length_mean",
    "packet_length_std",
    "syn_flag_count",
    "destination_port",
)


@dataclass(frozen=True)
class DataSplits:
    """Feature and label partitions with source row indices."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    train_indices: list[int]
    validation_indices: list[int]
    test_indices: list[int]


def generate_synthetic_dataset(
    n_samples: int = 2_000,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Generate an overlapping, deterministic synthetic binary flow dataset.

    Feature distributions are illustrative only and are not derived from real
    network traffic. Class-conditional shifts are deliberately overlapping.
    """
    if n_samples < 40:
        raise ValueError("n_samples must be at least 40 to support stratified splits")

    rng = np.random.default_rng(random_state)
    attack_count = round(n_samples * 0.28)
    labels = np.array(["benign"] * (n_samples - attack_count) + ["attack"] * attack_count)
    rng.shuffle(labels)
    attack = labels == "attack"

    duration = rng.lognormal(mean=4.5 + 0.18 * attack, sigma=1.0, size=n_samples)
    fwd_packets = rng.negative_binomial(5, 0.12 - 0.012 * attack) + 1
    bwd_packets = rng.negative_binomial(4, 0.14 - 0.01 * attack) + 1
    packet_mean = rng.normal(430 + 35 * attack, 155, size=n_samples).clip(40, 1_500)
    packet_std = rng.gamma(shape=2.2 + 0.12 * attack, scale=52, size=n_samples)
    fwd_bytes = fwd_packets * packet_mean * rng.lognormal(0, 0.22, size=n_samples)
    bwd_bytes = bwd_packets * packet_mean * rng.lognormal(-0.08, 0.25, size=n_samples)
    duration_seconds = duration / 1_000
    total_bytes = fwd_bytes + bwd_bytes
    bytes_per_second = total_bytes / np.maximum(duration_seconds, 0.01)
    syn_flags = rng.poisson(0.8 + 0.45 * attack, size=n_samples)
    common_ports = np.array([53, 80, 443, 8080, 22, 25, 445, 3389, 8443, 123])
    port_weights = np.array([0.12, 0.20, 0.24, 0.09, 0.08, 0.05, 0.07, 0.06, 0.05, 0.04])
    port_weights /= port_weights.sum()
    destination_ports = rng.choice(common_ports, size=n_samples, p=port_weights)
    unusual_port = rng.random(n_samples) < (0.06 + 0.06 * attack)
    destination_ports[unusual_port] = rng.integers(1_024, 65_536, unusual_port.sum())

    frame = pd.DataFrame(
        {
            "flow_duration": duration,
            "total_fwd_packets": fwd_packets,
            "total_backward_packets": bwd_packets,
            "total_fwd_bytes": fwd_bytes,
            "total_backward_bytes": bwd_bytes,
            "flow_bytes_per_second": bytes_per_second,
            "packet_length_mean": packet_mean,
            "packet_length_std": packet_std,
            "syn_flag_count": syn_flags,
            "destination_port": destination_ports,
            LABEL_COLUMN: labels,
        },
        columns=(*FEATURE_COLUMNS, LABEL_COLUMN),
    )
    validate_dataset(frame)
    return frame


def validate_dataset(frame: pd.DataFrame) -> None:
    """Validate required columns, target labels, and numeric feature types.

    Missing and infinite numeric feature values are left for the fitted
    training-only preprocessing pipeline to handle.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Dataset must be a pandas DataFrame")

    missing = [column for column in (*FEATURE_COLUMNS, LABEL_COLUMN) if column not in frame]
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Dataset must contain at least one row")

    invalid_labels = sorted(set(frame[LABEL_COLUMN].dropna().unique()) - set(LABELS))
    if frame[LABEL_COLUMN].isna().any() or invalid_labels:
        raise ValueError(
            f"Invalid target labels; expected only {LABELS}, found "
            f"{invalid_labels or 'missing values'}"
        )
    if frame[LABEL_COLUMN].nunique() != 2:
        raise ValueError(f"Dataset must contain both target classes: {LABELS}")

    features = frame.drop(columns=[LABEL_COLUMN])
    non_numeric = [
        column for column in features if not pd.api.types.is_numeric_dtype(features[column])
    ]
    if non_numeric:
        raise TypeError(f"Feature column(s) must be numeric: {', '.join(non_numeric)}")
def split_dataset(
    frame: pd.DataFrame,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> DataSplits:
    """Create deterministic, stratified 60/20/20 partitions without fitting transforms."""
    validate_dataset(frame)
    if not frame.index.is_unique:
        raise ValueError("Dataset row indices must be unique before splitting")

    positions = np.arange(len(frame))
    labels = frame[LABEL_COLUMN].to_numpy()
    train_validation_positions, test_positions = train_test_split(
        positions,
        test_size=0.20,
        random_state=random_state,
        stratify=labels,
    )
    train_positions, validation_positions = train_test_split(
        train_validation_positions,
        test_size=0.25,
        random_state=random_state + 1,
        stratify=labels[train_validation_positions],
    )

    train_index = frame.index.take(train_positions)
    validation_index = frame.index.take(validation_positions)
    test_index = frame.index.take(test_positions)
    train_set, validation_set, test_set = map(
        set, (train_index.tolist(), validation_index.tolist(), test_index.tolist())
    )
    assert train_set.isdisjoint(validation_set)
    assert train_set.isdisjoint(test_set)
    assert validation_set.isdisjoint(test_set)

    partitions: list[tuple[pd.Index, pd.Series]] = [
        (train_index, frame.iloc[train_positions][LABEL_COLUMN]),
        (validation_index, frame.iloc[validation_positions][LABEL_COLUMN]),
        (test_index, frame.iloc[test_positions][LABEL_COLUMN]),
    ]
    for _, target in partitions:
        assert set(target.unique()) == set(LABELS)

    return DataSplits(
        X_train=frame.iloc[train_positions].drop(columns=[LABEL_COLUMN]),
        X_validation=frame.iloc[validation_positions].drop(columns=[LABEL_COLUMN]),
        X_test=frame.iloc[test_positions].drop(columns=[LABEL_COLUMN]),
        y_train=partitions[0][1],
        y_validation=partitions[1][1],
        y_test=partitions[2][1],
        train_indices=train_index.tolist(),
        validation_indices=validation_index.tolist(),
        test_indices=test_index.tolist(),
    )


def load_csv(path: str | Path, required_columns: Sequence[str]) -> pd.DataFrame:
    """Load a CSV after checking source headers and required columns.

    Rows are not dropped or otherwise cleaned by this function.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
    if not required_columns:
        raise ValueError("required_columns must contain at least one column name")

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            header = next(csv.reader(csv_file), None)
    except (csv.Error, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read a valid CSV header from '{csv_path}': {exc}") from exc

    if header is not None:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for column in header:
            if column in seen:
                duplicates.add(column)
            seen.add(column)
        if duplicates:
            raise ValueError(f"CSV contains duplicate column names: {sorted(duplicates)}")
        if any(not column.strip() for column in header):
            raise ValueError(f"CSV '{csv_path}' contains a blank column name")

    try:
        frame = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read a valid CSV from '{csv_path}': {exc}") from exc

    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"CSV '{csv_path}' is missing required column(s): {', '.join(missing)}. "
            f"Available columns: {', '.join(map(str, frame.columns))}"
        )
    return frame


def generate_demo_files(
    output_path: str | Path = DEFAULT_DATA_PATH,
    *,
    n_samples: int = 2_000,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Path:
    """Generate the fixed demo CSV and its reproducible split-index manifest."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset = generate_synthetic_dataset(n_samples, random_state=random_state)
    splits = split_dataset(dataset, random_state=random_state)
    dataset.to_csv(destination, index=False)
    manifest_path = destination.with_name("split_indices.json")
    manifest_path.write_text(
        json.dumps(
            {
                "random_state": random_state,
                "dataset_path": destination.name,
                "split_sizes": {
                    "train": len(splits.train_indices),
                    "validation": len(splits.validation_indices),
                    "test": len(splits.test_indices),
                },
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
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the FPGuard synthetic flow dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args()
    destination = generate_demo_files(
        args.output, n_samples=args.samples, random_state=args.seed
    )
    print(f"Generated {args.samples} synthetic rows at {destination}")
    print(f"Saved deterministic split indices to {destination.with_name('split_indices.json')}")


if __name__ == "__main__":
    main()
