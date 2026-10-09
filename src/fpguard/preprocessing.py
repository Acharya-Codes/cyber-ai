"""Training-only preprocessing for numeric network-flow features."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def _replace_non_finite(features: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
    """Convert positive and negative infinity to NaN for median imputation."""
    if isinstance(features, pd.DataFrame):
        return features.replace([np.inf, -np.inf], np.nan)
    result = np.array(features, dtype=float, copy=True)
    result[~np.isfinite(result)] = np.nan
    return result


def _require_finite(features: np.ndarray) -> np.ndarray:
    if not np.isfinite(features).all():
        raise ValueError("Preprocessing produced non-finite feature values")
    return features


def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str] = (),
) -> Pipeline:
    """Build an unfitted numeric pipeline; fit it only with training rows.

    Missing values and positive/negative infinity are imputed with each
    feature's training-set median. Entirely empty training columns are retained
    as zero-valued columns by scikit-learn's ``keep_empty_features`` behavior.
    Categorical features are not supported by this numeric-only stage.
    """
    if categorical_features:
        raise TypeError("Categorical features are not supported; provide numeric features only")
    if not numeric_features:
        raise ValueError("numeric_features must contain at least one feature name")
    if len(set(numeric_features)) != len(numeric_features):
        raise ValueError("numeric_features must not contain duplicates")

    return Pipeline(
        steps=[
            (
                "non_finite_to_missing",
                FunctionTransformer(_replace_non_finite, validate=False),
            ),
            (
                "median_imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            (
                "finite_output_check",
                FunctionTransformer(_require_finite, validate=False),
            ),
        ]
    )
