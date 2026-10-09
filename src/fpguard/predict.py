"""Prediction helpers for a saved, already-fitted baseline pipeline."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from fpguard.train import DEFAULT_ARTIFACT_DIR, MODEL_FILENAME


def load_model(model_path: str | Path = DEFAULT_ARTIFACT_DIR / MODEL_FILENAME) -> Pipeline:
    """Load a fitted pipeline without fitting or changing any preprocessing."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Saved model does not exist: {path}")
    model = joblib.load(path)
    if not isinstance(model, Pipeline) or not hasattr(model, "predict_proba"):
        raise TypeError(f"Saved object at '{path}' is not a fitted probability pipeline")
    if not hasattr(model, "feature_names_in_"):
        raise ValueError(f"Saved model at '{path}' does not expose its fitted feature schema")
    return model


def _ordered_features(model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("Prediction features must be provided as a pandas DataFrame")
    if features.empty:
        raise ValueError("Prediction batch must contain at least one row")
    expected = list(model.feature_names_in_)
    missing = [column for column in expected if column not in features.columns]
    extra = [column for column in features.columns if column not in expected]
    if missing:
        raise ValueError(f"Prediction batch is missing required feature(s): {', '.join(missing)}")
    if extra:
        raise ValueError(f"Prediction batch contains unsupported feature(s): {', '.join(extra)}")
    ordered = features.loc[:, expected]
    non_numeric = [
        column
        for column in ordered
        if not pd.api.types.is_numeric_dtype(ordered[column])
    ]
    if non_numeric:
        raise TypeError(f"Feature column(s) must be numeric: {', '.join(non_numeric)}")
    return ordered


def predict_probabilities(model: Pipeline, features: pd.DataFrame) -> pd.Series:
    """Return attack probabilities in the model's recorded feature order."""
    if not isinstance(model, Pipeline) or not hasattr(model, "feature_names_in_"):
        raise TypeError("model must be a fitted Scikit-learn Pipeline")
    ordered = _ordered_features(model, features)
    classes = list(model.classes_)
    if "attack" not in classes:
        raise ValueError("Fitted model does not contain the required positive class 'attack'")
    scores = model.predict_proba(ordered)[:, classes.index("attack")]
    if not np.isfinite(scores).all():
        raise ValueError("Model returned non-finite attack probabilities")
    return pd.Series(scores, index=ordered.index, name="attack_probability")


def predict_network_flows(
    features: pd.DataFrame,
    model_path: str | Path = DEFAULT_ARTIFACT_DIR / MODEL_FILENAME,
) -> tuple[pd.Series, pd.Series]:
    """Load the saved model and return predicted labels and attack probabilities."""
    model = load_model(model_path)
    probabilities = predict_probabilities(model, features)
    labels = pd.Series(
        np.where(probabilities.to_numpy() >= 0.5, "attack", "benign"),
        index=probabilities.index,
        name="predicted_label",
    )
    return labels, probabilities
