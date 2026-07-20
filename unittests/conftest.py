"""Shared fixtures for the model_metrics test suite.

Forces a headless matplotlib backend before pyplot is imported anywhere, and
provides the model bundles used across the feature-selection tests.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_regression


@pytest.fixture(autouse=True)
def _mpl_hygiene():
    """Neutralize plt.show and close figures after every test."""
    orig_show = plt.show
    plt.show = lambda *a, **k: None
    try:
        yield
    finally:
        plt.close("all")
        plt.show = orig_show


class NaNRF(RandomForestClassifier):
    """RandomForest that tolerates NaN by zero-filling.

    The ablation sweep blanks excluded features to NaN, which a stock
    RandomForest rejects. Zero-filling keeps the tests fast and deterministic
    without pulling in an OpenMP-backed booster.
    """

    def predict_proba(self, X):
        return super().predict_proba(np.nan_to_num(np.asarray(X, dtype=float)))


@pytest.fixture(scope="module")
def rf_reg():
    """Plain RandomForestRegressor on named numeric columns."""
    X, y = make_regression(
        n_samples=300, n_features=8, n_informative=5, noise=8, random_state=0
    )
    cols = ["bmi", "s5", "bp", "s6", "age", "s1", "s2", "s3"]
    X = pd.DataFrame(X, columns=cols)
    return RandomForestRegressor(n_estimators=50, random_state=0).fit(X, y)


@pytest.fixture(scope="module")
def flat_clf():
    """NaN-tolerant classifier on six plain numeric columns: (model, X, y)."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(300, 6), columns=[f"f{i}" for i in range(6)])
    y = ((X["f0"] * 2 + X["f1"] + rng.randn(300) * 0.4) > 0).astype(int)
    return NaNRF(n_estimators=40, random_state=0).fit(X, y), X, y


@pytest.fixture(scope="module")
def encoded_clf():
    """One-hot encoded model for the grouping tests: (model, Xt, y, ct).

    The categorical column name contains a space ("CKD Stage") deliberately, to
    exercise the longest-prefix matching in the group inference. Splitting such
    names on underscores would mangle them.
    """
    rng = np.random.RandomState(0)
    n = 300
    raw = pd.DataFrame(
        {
            "age": rng.randn(n),
            "egfr": rng.randn(n),
            "CKD Stage": rng.choice(["S2", "S3b", "S4"], n),
        }
    )
    y = (
        ((raw["CKD Stage"] == "S4").astype(int) * 2 + raw["age"] + rng.randn(n) * 0.5)
        > 1
    ).astype(int)

    ct = ColumnTransformer(
        remainder="drop",
        transformers=[
            ("num", "passthrough", ["age", "egfr"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["CKD Stage"],
            ),
        ],
    ).fit(raw)

    Xt = pd.DataFrame(
        ct.transform(raw), columns=ct.get_feature_names_out(), index=raw.index
    )
    return NaNRF(n_estimators=40, random_state=0).fit(Xt, y), Xt, y, ct
