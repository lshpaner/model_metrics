"""
Tests for the helpers in ``feature_selection_utils``.

Covers the three supported final-estimator interfaces (CatBoost-style
get_feature_importance(), sklearn feature_importances_, linear coef_) plus
the multiclass coef_ collapse and the unsupported-estimator error path.

Covers importance resolution and the one-hot grouping helpers.
"""

import numpy as np
import pytest

from model_metrics.feature_selection_utils import (
    _resolve_feature_importance_values,
    _resolve_feature_groups,
    _group_importance_df,
    _expand_groups,
    _infer_groups_from_column_transformer,
)
import pandas as pd


# --------------------------------------------------------------------------- #
# Stubs                                                                       #
# --------------------------------------------------------------------------- #
class CatBoostStub:
    """Exposes get_feature_importance() like a CatBoost / model_tuner final step."""

    def __init__(self, importances):
        self._importances = np.asarray(importances, dtype=float)

    def get_feature_importance(self):
        return self._importances


class TreeStub:
    """Exposes feature_importances_ like an sklearn tree/ensemble."""

    def __init__(self, importances):
        self.feature_importances_ = np.asarray(importances, dtype=float)


class LinearStub:
    """Exposes coef_ like a linear model."""

    def __init__(self, coef):
        self.coef_ = np.asarray(coef, dtype=float)


class UnsupportedStub:
    """Exposes none of the three supported interfaces."""


# --------------------------------------------------------------------------- #
# get_feature_importance() path                                               #
# --------------------------------------------------------------------------- #
def test_catboost_style_returns_values_unchanged():
    est = CatBoostStub([40.0, 25.0, 15.0, 10.0])
    out = _resolve_feature_importance_values(est)
    np.testing.assert_array_equal(out, np.array([40.0, 25.0, 15.0, 10.0]))
    assert out.dtype == float


def test_catboost_style_takes_precedence_over_other_attrs():
    # If an object somehow exposes several interfaces, get_feature_importance wins.
    est = CatBoostStub([1.0, 2.0, 3.0])
    est.feature_importances_ = np.array([9.0, 9.0, 9.0])
    est.coef_ = np.array([[9.0, 9.0, 9.0]])
    out = _resolve_feature_importance_values(est)
    np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0]))


# --------------------------------------------------------------------------- #
# feature_importances_ path                                                    #
# --------------------------------------------------------------------------- #
def test_tree_style_returns_feature_importances():
    est = TreeStub([0.5, 0.3, 0.2])
    out = _resolve_feature_importance_values(est)
    np.testing.assert_allclose(out, np.array([0.5, 0.3, 0.2]))


def test_tree_style_preferred_over_coef():
    est = TreeStub([0.6, 0.4])
    est.coef_ = np.array([[10.0, -10.0]])  # should be ignored
    out = _resolve_feature_importance_values(est)
    np.testing.assert_allclose(out, np.array([0.6, 0.4]))


# --------------------------------------------------------------------------- #
# coef_ path                                                                   #
# --------------------------------------------------------------------------- #
def test_linear_binary_2d_uses_absolute_magnitude():
    # Shape (1, n): binary logistic regression.
    est = LinearStub([[-2.0, 1.5, -0.5]])
    out = _resolve_feature_importance_values(est)
    np.testing.assert_allclose(out, np.array([2.0, 1.5, 0.5]))
    assert out.shape == (3,)


def test_linear_1d_coef_uses_absolute_magnitude():
    # Some linear estimators expose a flat coef_.
    est = LinearStub([-2.0, 3.0, -1.0])
    out = _resolve_feature_importance_values(est)
    np.testing.assert_allclose(out, np.array([2.0, 3.0, 1.0]))


def test_linear_multiclass_collapses_to_per_feature_l2_norm():
    # Shape (k, n) with k > 1: collapse across classes via L2 norm per feature.
    coef = [[3.0, 0.0, 1.0], [4.0, 0.0, 0.0]]
    est = LinearStub(coef)
    out = _resolve_feature_importance_values(est)
    # column norms: sqrt(3^2+4^2)=5, sqrt(0)=0, sqrt(1)=1
    np.testing.assert_allclose(out, np.array([5.0, 0.0, 1.0]))
    assert out.shape == (3,)


def test_linear_output_is_non_negative():
    est = LinearStub([[-5.0, -4.0, -3.0]])
    out = _resolve_feature_importance_values(est)
    assert np.all(out >= 0)


# --------------------------------------------------------------------------- #
# Error path                                                                   #
# --------------------------------------------------------------------------- #
def test_unsupported_estimator_raises_attributeerror():
    with pytest.raises(AttributeError):
        _resolve_feature_importance_values(UnsupportedStub())


def test_error_message_names_the_estimator_type():
    with pytest.raises(AttributeError, match="UnsupportedStub"):
        _resolve_feature_importance_values(UnsupportedStub())


# --------------------------------------------------------------------------- #
# Grouping helpers                                                             #
# --------------------------------------------------------------------------- #
def test_group_inference_maps_onehot_to_parent(encoded_clf):
    _, Xt, _, ct = encoded_clf
    mapping = _resolve_feature_groups("auto", list(Xt.columns), ct)
    stage_cols = [c for c in Xt.columns if "CKD Stage" in c]
    assert len(stage_cols) == 3
    assert all(mapping[c] == "CKD Stage" for c in stage_cols)
    assert mapping["num__age"] == "age"


def test_group_inference_helper_directly(encoded_clf):
    _, Xt, _, ct = encoded_clf
    mapping = _infer_groups_from_column_transformer(ct, list(Xt.columns))
    assert set(mapping.values()) == {"age", "egfr", "CKD Stage"}


def test_expand_groups_roundtrip(encoded_clf):
    _, Xt, _, ct = encoded_clf
    mapping = _resolve_feature_groups("auto", list(Xt.columns), ct)
    members = _expand_groups(["CKD Stage"], mapping)
    assert len(members) == 3 and all("CKD Stage" in m for m in members)


def test_group_importance_df_helper():
    imp = pd.DataFrame(
        {"feature": ["a_1", "a_2", "b"], "importance": [0.2, 0.3, 0.5]}
    )
    grouped = _group_importance_df(imp, {"a_1": "a", "a_2": "a", "b": "b"})
    assert grouped.loc[grouped["feature"] == "a", "importance"].iloc[0] == pytest.approx(0.5)
    assert grouped["importance"].sum() == pytest.approx(1.0)
