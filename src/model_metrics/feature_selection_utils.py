"""Feature-selection helpers shared by the feature functions in
model_evaluator: importance resolution, ranking, and label cleaning, plus
the performance-sweep plotting and retention utilities."""

import numpy as np
import pandas as pd


def _resolve_feature_importance_values(estimator):
    """
    Return a 1-D array of non-negative importance magnitudes from a fitted
    final estimator, supporting three conventions:

    1. CatBoost / model_tuner wrapper style : ``get_feature_importance()``
    2. sklearn tree / ensemble style        : ``feature_importances_``
    3. linear model style                   : ``coef_`` (absolute magnitude)

    For linear models the magnitude is ``|coef_|`` (binary) or the per-feature
    L2 norm across classes (multiclass). This is only comparable across
    features when they are on a common scale (e.g. standardized upstream).

    Raises
    ------
    AttributeError
        If the estimator exposes none of the three interfaces.
    """
    if hasattr(estimator, "get_feature_importance"):
        return np.asarray(estimator.get_feature_importance(), dtype=float)

    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_, dtype=float)

    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        if coef.ndim == 1:
            return np.abs(coef)
        if coef.shape[0] == 1:  # binary classification / single-output
            return np.abs(coef.ravel())
        return np.linalg.norm(coef, axis=0)  # multiclass: L2 across classes

    raise AttributeError(
        f"{type(estimator).__name__} exposes none of get_feature_importance(), "
        "feature_importances_, or coef_; cannot compute feature importance."
    )


def _rank_features_by_importance(
    model,
    feature_names=None,
    feature_selection_step="feature_selection_RFE",
    preprocessor_step="preprocess_column_transformer_ColumnTransformer",
):
    """
    Return a DataFrame ``[feature, importance]`` sorted by importance
    (descending), working across three model shapes:

    * model_tuner-style wrapper (``.estimator.estimator`` pipeline, often with
      ``get_feature_names()`` and a CatBoost final step)
    * a plain fitted sklearn estimator (RandomForest, LogisticRegression, ...)
    * an sklearn ``Pipeline``

    Feature names are resolved, in order of preference, from: an explicit
    ``feature_names`` argument, the wrapper's ``get_feature_names()``, a named
    preprocessor's ``get_feature_names_out()``, the estimator's
    ``feature_names_in_``, and finally generic ``feature_i`` placeholders.
    """
    # Locate the underlying sklearn object and its final estimator.
    if hasattr(model, "estimator") and hasattr(
        getattr(model, "estimator"), "estimator"
    ):
        inner = model.estimator.estimator  # model_tuner wrapper
    else:
        inner = model  # plain estimator or Pipeline
    final_est = inner[-1] if hasattr(inner, "named_steps") else inner

    importances = _resolve_feature_importance_values(final_est)

    # Resolve feature names from the best available source.
    names = None
    if feature_names is not None:
        names = np.asarray(feature_names)
    elif hasattr(model, "get_feature_names"):
        try:
            names = np.asarray(model.get_feature_names())
        except Exception:
            names = None
    if (
        names is None
        and hasattr(inner, "named_steps")
        and preprocessor_step in inner.named_steps
    ):
        try:
            names = np.asarray(
                inner.named_steps[preprocessor_step].get_feature_names_out()
            )
        except Exception:
            names = None
    if names is None:
        for obj in (final_est, model):
            if hasattr(obj, "feature_names_in_"):
                names = np.asarray(obj.feature_names_in_)
                break

    # Apply a feature-selection mask only when it reconciles a length gap
    # (pre-selection names longer than the post-selection importances).
    if (
        names is not None
        and hasattr(inner, "named_steps")
        and any("feature_selection" in s for s in inner.named_steps)
        and len(names) != len(importances)
    ):
        mask = inner.named_steps[feature_selection_step].get_support()
        names = names[mask]

    if names is None:
        names = np.array([f"feature_{i}" for i in range(len(importances))])
    elif len(names) != len(importances):
        raise ValueError(
            f"Resolved {len(names)} feature names but the model reports "
            f"{len(importances)} importances. Pass feature_names=... explicitly."
        )

    return (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _clean_feature_labels(names, clean_names):
    """Map feature names to display labels.

    ``clean_names`` may be a dict ``{original: label}`` (unmapped names are
    left unchanged) or a callable ``name -> label``. None returns the names
    unchanged. Display only; ranking and returned data keep the originals.
    """
    if clean_names is None:
        return list(names)
    if callable(clean_names):
        return [clean_names(n) for n in names]
    return [clean_names.get(n, n) for n in names]


def _plot_perf_series(ax, xv, yv, label, smooth, curve_kwgs=None, marker_kwgs=None):
    """Plot one metric curve, optionally PCHIP-smoothed, with markers at the real
    points. ``curve_kwgs`` styles the line (e.g. ``linewidth``); ``marker_kwgs``
    styles the point markers (defaults to a size-5 circle inheriting the line
    color). Both are already-resolved flat dicts for this single curve. Falls
    back to a straight polyline when smoothing is off, there are too few points,
    or SciPy is unavailable. Returns the line handle."""
    xv = np.asarray(xv, dtype=float)
    yv = np.asarray(yv, dtype=float)
    ck = dict(curve_kwgs or {})
    mk = {"marker": "o", "markersize": 5}
    mk.update(marker_kwgs or {})
    if smooth and len(xv) >= 3:
        try:
            from scipy.interpolate import PchipInterpolator

            xs = np.linspace(xv.min(), xv.max(), 200)
            ys = PchipInterpolator(xv, yv)(xs)
            (line,) = ax.plot(xs, ys, label=label, **ck)
            mk.setdefault("color", line.get_color())          # markers follow line color
            ax.plot(xv, yv, linestyle="None", **mk)
            return line
        except Exception:
            pass
    (line,) = ax.plot(xv, yv, label=label, **{**ck, **mk})
    return line


def _retained_fractions(results_df, resolved):
    """Per-k fraction of full-model performance, per metric.

    Higher-is-better metrics use score_k / score_full; lower-is-better
    (Brier, MAE, RMSE, MSE) use score_full / score_k, so 1.0 always means
    "matches the full-feature model" and the curve rises toward it.
    """
    full = results_df.iloc[-1]
    frac = pd.DataFrame({"n_features": results_df["n_features"]})
    for label, _, hib, _ in resolved:
        fv = full[label]
        if hib:
            frac[label] = results_df[label] / fv if fv != 0 else np.nan
        else:
            frac[label] = fv / results_df[label].replace(0, np.nan)
    return frac


def _infer_groups_from_column_transformer(ct, names):
    """Map one-hot / expanded output columns back to their source column.

    Uses the fitted ColumnTransformer's own ``transformers_`` (the input column
    list per branch) and matches each output name against those columns by
    longest prefix, so names containing spaces or underscores (e.g.
    ``cat__CKD Stage_CKD Stage 4``) resolve correctly without string splitting.
    Returns ``{output_name: source_column}``; unmatched names map to themselves.
    """
    mapping = {}
    branches = []  # (prefix, [input columns])
    for tname, _trans, cols in getattr(ct, "transformers_", []):
        if cols is None or isinstance(cols, str):
            continue
        try:
            cols = list(cols)
        except TypeError:
            continue
        branches.append((f"{tname}__", [str(c) for c in cols]))

    for n in names:
        src = None
        for prefix, cols in branches:
            if not n.startswith(prefix):
                continue
            rest = n[len(prefix):]
            # longest matching input column wins ("CKD Stage" over "CKD")
            cands = [c for c in cols if rest == c or rest.startswith(c + "_")]
            if cands:
                src = max(cands, key=len)
                break
        mapping[n] = src if src is not None else n
    return mapping


def _resolve_feature_groups(feature_groups, names, column_transformer=None):
    """Normalize ``feature_groups`` into a ``{feature: parent}`` dict.

    Accepts a dict (unmapped names keep themselves), a callable
    ``name -> parent``, or ``"auto"`` to infer from ``column_transformer``.
    Returns None when ``feature_groups`` is None (no grouping).
    """
    if feature_groups is None:
        return None
    names = [str(n) for n in names]
    if isinstance(feature_groups, str):
        if feature_groups != "auto":
            raise ValueError(
                f"feature_groups must be a dict, a callable, or 'auto'; "
                f"got {feature_groups!r}."
            )
        if column_transformer is None:
            raise ValueError(
                "feature_groups='auto' requires `column_transformer=` (the "
                "fitted ColumnTransformer) so output columns can be mapped "
                "back to their source columns."
            )
        return _infer_groups_from_column_transformer(column_transformer, names)
    if callable(feature_groups):
        return {n: feature_groups(n) for n in names}
    if isinstance(feature_groups, dict):
        return {n: feature_groups.get(n, n) for n in names}
    raise ValueError(
        "feature_groups must be a dict, a callable, or 'auto'; got "
        f"{type(feature_groups).__name__}."
    )


def _group_importance_df(importance_df, mapping):
    """Collapse an importance table to parent groups, summing within each.

    One-hot expansion splits a categorical's importance across its level
    columns; summing restores the variable-level importance. Ties are broken by
    the group's best-ranked member so ordering stays stable.
    """
    df = importance_df.copy()
    df["_order"] = np.arange(len(df))
    df["feature"] = [mapping.get(f, f) for f in df["feature"]]
    grouped = (
        df.groupby("feature", as_index=False)
        .agg(importance=("importance", "sum"), _order=("_order", "min"))
        .sort_values(["importance", "_order"], ascending=[False, True])
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    return grouped


def _expand_groups(group_names, mapping):
    """Return the underlying feature columns for each parent group name."""
    members = {}
    for feat, parent in mapping.items():
        members.setdefault(parent, []).append(feat)
    out = []
    for g in group_names:
        out.extend(members.get(g, [g]))
    return out