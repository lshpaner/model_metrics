"""
Align an external validation frame to the exact feature contract of a fitted model.

Catches the three silent failure modes of hand-built mapping loops:
  1. source column absent from the incoming frame
  2. destination name that the model never asked for (pandas creates it silently)
  3. expected feature that no mapping ever touched (stays all-NaN)
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


class FeatureContractError(KeyError):
    """Raised when a mapping does not satisfy a model's expected feature set."""


# --------------------------------------------------------------------------- #
# 1. Discover what the model actually wants
# --------------------------------------------------------------------------- #
#: attributes that commonly hold the real estimator inside a wrapper
#: (model_tuner.Model, GridSearchCV, calibration, TransformedTargetRegressor,
#: and most hand-rolled wrappers)
_WRAPPER_ATTRS = (
    "best_estimator_",
    "estimator_",
    "estimator",
    "regressor_",
    "classifier_",
    "base_estimator",
    "calibrated_classifiers_",
    "model",
    "_model",
    "pipeline",
    "test_model",
)


def _raw_cols_from_column_transformer(ct: Any) -> list[str]:
    """
    Flatten a fitted ColumnTransformer's transformers_ into the ordered list of
    raw input column names it was fit on. Skips the 'remainder' entry and any
    transformer explicitly dropped. Preserves first-seen order without dupes.
    """
    out: list[str] = []
    seen: set[str] = set()
    for name, trans, cols in ct.transformers_:
        if name == "remainder" or trans == "drop":
            continue
        if isinstance(cols, str):
            cols = [cols]
        try:
            cols = list(cols)
        except TypeError:
            continue
        for c in cols:
            c = str(c)
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out


def get_expected_features(model: Any, _seen: set[int] | None = None) -> list[str]:
    """
    Return the ordered feature names a fitted estimator expects at predict time.

    Handles sklearn estimators and Pipelines, XGBoost (sklearn API and native
    Booster), LightGBM, CatBoost, statsmodels, and wrappers that hold the fitted
    estimator on an attribute. Raises if no name information is reachable.
    """
    _seen = _seen or set()
    if id(model) in _seen:
        raise FeatureContractError("cyclic wrapper chain while resolving features")
    _seen.add(id(model))

    # ColumnTransformer: reconstruct the RAW pre-transform input columns from
    # transformers_. This is what you build an external frame against, since the
    # booster's own names are post-transform (num__/cat__ prefixed).
    transformers = getattr(model, "transformers_", None)
    if transformers is not None:
        raw = _raw_cols_from_column_transformer(model)
        if raw:
            return raw

    # Pipeline: the raw contract is whatever the first step was fitted on.
    # Prefer feature_names_in_ (raw input to the pipeline). Only fall through to
    # the first step if the pipeline never recorded it.
    steps = getattr(model, "steps", None)
    if steps:
        names = getattr(model, "feature_names_in_", None)
        if names is not None and len(names):
            return [str(c) for c in names]
        return get_expected_features(steps[0][1], _seen)

    # Bare estimator with recorded input names
    for attr in ("feature_names_in_", "feature_name_", "feature_names_"):
        names = getattr(model, attr, None)
        if names is not None and len(names):
            return [str(c) for c in names]

    # XGBoost sklearn wrapper
    booster = getattr(model, "get_booster", None)
    if callable(booster):
        names = booster().feature_names
        if names:
            return [str(c) for c in names]

    # Native xgboost.Booster
    names = getattr(model, "feature_names", None)
    if names:
        return [str(c) for c in names]

    # LightGBM Booster
    fn = getattr(model, "feature_name", None)
    if callable(fn):
        names = fn()
        if names:
            return [str(c) for c in names]

    # statsmodels results
    names = getattr(model, "exog_names", None)
    if names:
        return [str(c) for c in names]

    # Wrapper objects: descend into whatever holds the fitted estimator
    for attr in _WRAPPER_ATTRS:
        inner = getattr(model, attr, None)
        if inner is None or isinstance(inner, (str, bytes, type)):
            continue
        if isinstance(inner, (list, tuple)) and inner:
            inner = inner[0]
            inner = getattr(inner, "estimator", inner)
        try:
            return get_expected_features(inner, _seen)
        except (FeatureContractError, AttributeError, TypeError):
            continue

    raise FeatureContractError(
        f"{type(model).__name__} exposes no feature names, and none of "
        f"{_WRAPPER_ATTRS} led to a fitted estimator that does. "
        "Pass expected=[...] explicitly."
    )


# --------------------------------------------------------------------------- #
# 2. Build the aligned frame
# --------------------------------------------------------------------------- #
def align_features(
    df: pd.DataFrame,
    model: Any = None,
    col_map: Mapping[str, str | Sequence[str]] | None = None,
    derived: Mapping[str, Callable[[pd.DataFrame], Any] | Any] | None = None,
    expected: Iterable[str] | None = None,
    *,
    on_unmapped: str = "warn",
    on_all_nan: str = "warn",
    dtype: str | None = "float",
    copy_passthrough: bool = False,
    fill: float | Mapping[str, Any] | Callable[[str], Any] | None = None,
    report_fill: bool = True,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df
        Incoming external frame, in its own naming convention.
    model
        Any fitted estimator. Used to resolve `expected` when not given.
    col_map
        {source_column: destination} or {source_column: [dest_a, dest_b]}.
    derived
        {destination: callable(df) -> array} or {destination: array-like},
        for columns computed rather than renamed.
    expected
        Explicit ordered feature list. Overrides the model.
    on_unmapped, on_all_nan
        One of "raise", "warn", "ignore".
    dtype
        Cast the result, or None to leave dtypes alone.
    copy_passthrough
        Also carry over any df column whose name already matches an expected
        feature, without needing an entry in col_map.
    fill
        Value(s) for expected features that no mapping touched, i.e. columns the
        external cohort does not collect at all. Default None leaves them NaN,
        which delegates to the model's own missing-value handling and asserts
        nothing.

        Accepts:
          * scalar          -> same value for every unmapped column
          * {name: value}   -> per-column; columns absent from the dict stay NaN
          * callable(name)  -> returns a value, or None to leave that column NaN

        Only unmapped columns are filled. A column that was mapped but happens to
        be NaN for some patients is genuine per-patient missingness and is left
        alone.

        Choosing a fill is a substantive modelling assumption, not a formatting
        detail: a structurally-absent flag filled with 0 asserts "condition not
        present" for every patient. State whichever fill you use, and prefer
        fixing it in advance rather than selecting it by comparing downstream
        performance.
    report_fill
        Emit a warning listing exactly which columns were filled with what, so
        the assumption is visible in logs and reproducible in a write-up.

    Returns
    -------
    DataFrame with exactly `expected`, in order, indexed like `df`.
    """
    if expected is None:
        if model is None:
            raise ValueError("Provide either `model` or `expected`.")
        expected = get_expected_features(model)
    expected = [str(c) for c in expected]

    if len(set(expected)) != len(expected):
        dupes = sorted({c for c in expected if expected.count(c) > 1})
        raise FeatureContractError(f"duplicate names in expected feature set: {dupes}")

    col_map = dict(col_map or {})
    derived = dict(derived or {})

    if copy_passthrough:
        for c in df.columns:
            if str(c) in expected and c not in col_map:
                col_map[c] = str(c)

    # --- validate sources -------------------------------------------------- #
    missing_src = [s for s in col_map if s not in df.columns]
    if missing_src:
        raise FeatureContractError(f"source columns absent from df: {sorted(missing_src)}")

    # --- validate destinations --------------------------------------------- #
    targets: dict[str, list[str]] = {}
    for src, dst in col_map.items():
        for d in np.atleast_1d(np.asarray(dst, dtype=object)).tolist():
            targets.setdefault(str(d), []).append(str(src))
    for d in derived:
        targets.setdefault(str(d), []).append("<derived>")

    exp_set = set(expected)
    unknown = sorted(d for d in targets if d not in exp_set)
    if unknown:
        raise FeatureContractError(
            f"destinations the model never asked for: {unknown}\n" f"expected features: {expected}"
        )

    collisions = {d: s for d, s in targets.items() if len(s) > 1}
    if collisions:
        raise FeatureContractError(f"destination written more than once: {collisions}")

    # --- materialize ------------------------------------------------------- #
    ext = pd.DataFrame(np.nan, index=df.index, columns=expected)

    for src, dst in col_map.items():
        vals = df[src].to_numpy()
        for d in np.atleast_1d(np.asarray(dst, dtype=object)).tolist():
            ext[str(d)] = vals

    for d, spec in derived.items():
        vals = spec(df) if callable(spec) else spec
        vals = vals.to_numpy() if isinstance(vals, (pd.Series, pd.Index)) else np.asarray(vals)
        if len(vals) != len(df):
            raise ValueError(f"derived column {d!r} has length {len(vals)}, expected {len(df)}")
        ext[str(d)] = vals

    # --- fill structurally-absent columns ---------------------------------- #
    unmapped = [c for c in expected if c not in targets]
    filled = _apply_fill(ext, unmapped, fill)

    if filled and report_fill:
        shown = ", ".join(f"{k}={v!r}" for k, v in list(filled.items())[:8])
        more = "" if len(filled) <= 8 else f" (+{len(filled) - 8} more)"
        warnings.warn(
            f"{len(filled)} unmapped column(s) filled rather than left missing: "
            f"{shown}{more}. This asserts a value the external cohort never "
            f"measured; record it wherever the results are reported.",
            stacklevel=3,
        )

    still_nan = [c for c in unmapped if c not in filled]

    # --- report gaps ------------------------------------------------------- #
    _report(
        on_unmapped,
        f"{len(still_nan)} expected feature(s) never mapped, left all-NaN: {still_nan}",
        still_nan,
    )

    all_nan = [
        c for c in expected if c not in unmapped and c not in filled and ext[c].isna().all()
    ]
    _report(
        on_all_nan,
        f"{len(all_nan)} mapped feature(s) are entirely NaN after alignment: {all_nan}",
        all_nan,
    )

    if dtype is not None:
        ext = ext.astype(dtype)

    return ext[expected]


def _apply_fill(
    ext: pd.DataFrame,
    unmapped: list[str],
    fill: float | Mapping[str, Any] | Callable[[str], Any] | None,
) -> dict[str, Any]:
    """
    Fill structurally-absent columns in place. Returns {column: value} for every
    column actually written, so the caller can report the assumption.
    """
    if fill is None or not unmapped:
        return {}

    filled: dict[str, Any] = {}

    if callable(fill) and not isinstance(fill, Mapping):
        for c in unmapped:
            v = fill(c)
            if v is not None:
                ext[c] = v
                filled[c] = v
        return filled

    if isinstance(fill, Mapping):
        stray = sorted(set(fill) - set(unmapped))
        if stray:
            raise FeatureContractError(
                f"fill targets column(s) that were mapped or are not model "
                f"features: {stray}. Fill only applies to unmapped columns; a "
                f"mapped column that is NaN for some patients is real missingness."
            )
        for c, v in fill.items():
            if v is not None:
                ext[c] = v
                filled[c] = v
        return filled

    for c in unmapped:  # scalar
        ext[c] = fill
        filled[c] = fill
    return filled


def _report(mode: str, msg: str, payload: list[str]) -> None:
    if not payload or mode == "ignore":
        return
    if mode == "raise":
        raise FeatureContractError(msg)
    if mode == "warn":
        warnings.warn(msg, stacklevel=3)
        return
    raise ValueError(f"unknown mode {mode!r}; use raise, warn, or ignore")
