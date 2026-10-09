"""Validation-only selection of a false-positive-conscious probability threshold."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix

LABELS = ("benign", "attack")
DEFAULT_CANDIDATE_THRESHOLDS = tuple(index / 100 for index in range(101))


def _validate_inputs(
    labels: Sequence[str],
    probabilities: Sequence[float],
    minimum_recall: float,
    candidate_thresholds: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, tuple[float, ...], float]:
    y_true = np.asarray(labels)
    if y_true.ndim != 1:
        raise ValueError("labels and probabilities must be one-dimensional and equal length")
    try:
        scores = np.asarray(probabilities, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Probabilities must be numeric values between 0 and 1") from exc
    if scores.ndim != 1 or len(y_true) != len(scores):
        raise ValueError("labels and probabilities must be one-dimensional and equal length")
    if not len(y_true):
        raise ValueError("Cannot select a threshold from an empty validation set")
    label_values = y_true.tolist()
    if any(
        not isinstance(label, (str, np.str_)) or label not in LABELS
        for label in label_values
    ):
        raise ValueError(f"Validation labels must contain only both classes: {LABELS}")
    if set(label_values) != set(LABELS):
        raise ValueError(f"Validation labels must contain only both classes: {LABELS}")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1")
    try:
        recall_target = float(minimum_recall)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "minimum_recall must be a number greater than 0 and at most 1"
        ) from exc
    if not np.isfinite(recall_target) or not 0 < recall_target <= 1:
        raise ValueError("minimum_recall must be greater than 0 and at most 1")

    try:
        thresholds = tuple(float(value) for value in candidate_thresholds)
    except (TypeError, ValueError) as exc:
        raise ValueError("Candidate thresholds must be numeric values between 0 and 1") from exc
    if not thresholds:
        raise ValueError("candidate_thresholds must contain at least one threshold")
    if not np.isfinite(thresholds).all() or any(
        value < 0 or value > 1 for value in thresholds
    ):
        raise ValueError("Candidate thresholds must be finite values between 0 and 1")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("candidate_thresholds must not contain duplicates")
    return y_true, scores, thresholds, recall_target


def _metrics_at_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predicted = np.where(probabilities >= threshold, "attack", "benign")
    matrix = confusion_matrix(labels, predicted, labels=list(LABELS))
    tn, fp, fn, tp = (
        int(matrix[0, 0]),
        int(matrix[0, 1]),
        int(matrix[1, 0]),
        int(matrix[1, 1]),
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": threshold,
        "confusion_matrix": matrix.tolist(),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "false_negative_rate": fn / (fn + tp) if fn + tp else None,
        "precision": precision,
        "recall": recall,
        "f1_score": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
    }


def select_threshold(
    validation_probabilities: Sequence[float],
    validation_labels: Sequence[str],
    *,
    minimum_recall: float = 0.80,
    candidate_thresholds: Sequence[float] = DEFAULT_CANDIDATE_THRESHOLDS,
) -> dict[str, Any]:
    """Select the lowest-FPR candidate meeting validation recall.

    By default, candidates are the inclusive 0.00--1.00 grid at 0.01
    increments. Ties are resolved by higher precision, then higher threshold.
    This function accepts validation labels/probabilities only; callers must
    freeze the returned threshold before evaluating on test data.
    """
    labels, probabilities, thresholds, recall_target = _validate_inputs(
        validation_labels,
        validation_probabilities,
        minimum_recall,
        candidate_thresholds,
    )
    candidate_metrics = [
        _metrics_at_threshold(labels, probabilities, threshold)
        for threshold in thresholds
    ]
    feasible = [
        metrics
        for metrics in candidate_metrics
        if metrics["recall"] >= recall_target
    ]
    if not feasible:
        return {
            "feasible": False,
            "minimum_recall": recall_target,
            "selected_threshold": None,
            "selection_reason": "No candidate threshold met the minimum validation recall.",
            "candidate_thresholds": list(thresholds),
            "candidates": candidate_metrics,
        }

    selected = min(
        feasible,
        key=lambda metrics: (
            metrics["false_positive_rate"],
            -metrics["precision"],
            -metrics["threshold"],
        ),
    )
    return {
        "feasible": True,
        "minimum_recall": recall_target,
        "selected_threshold": selected["threshold"],
        "selection_reason": (
            "Lowest validation FPR among candidates meeting the recall target; "
            "ties favor higher precision, then a higher threshold."
        ),
        "candidate_thresholds": list(thresholds),
        "selected_validation_metrics": selected,
        "candidates": candidate_metrics,
    }
