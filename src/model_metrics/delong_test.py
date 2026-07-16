import numpy as np
from scipy import stats
from scipy.stats import norm

from sklearn.metrics import roc_auc_score


def _compute_midrank(x):
    """Midranks with proper handling of ties (average rank within a tie group)."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted_transposed, m):
    """
    Fast DeLong (Sun & Xu, 2014).

    Parameters
    ----------
    preds_sorted_transposed : ndarray, shape (k_models, n)
        Scores for each model, columns ordered positives-first.
    m : int
        Number of positive samples (the first m columns).

    Returns
    -------
    aucs : ndarray, shape (k_models,)
    delong_cov : ndarray, shape (k_models, k_models)
        Covariance matrix of the AUC estimates (includes the between-model
        covariance that a single-AUC standard error ignores).
    """
    n = preds_sorted_transposed.shape[1] - m
    k = preds_sorted_transposed.shape[0]
    positive = preds_sorted_transposed[:, :m]
    negative = preds_sorted_transposed[:, m:]

    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, np.atleast_2d(delong_cov)


def delong_roc_test(
    y_true,
    y_scores_1,
    y_scores_2,
    model_names=None,
    verbose=True,
    return_values=False,
    decimal_places=4,
):
    """
    DeLong test for the difference between two correlated ROC AUCs.

    This is the exact, nonparametric counterpart to
    ``hanley_mcneil_auc_test``. It uses empirical placement values (no
    distributional assumption) and, crucially, the full covariance between the
    two AUCs, so it is the correct test when both models are scored on the same
    samples. Returns ``(auc1, auc2, p_value)``.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0/1).
    y_scores_1, y_scores_2 : array-like
        Predicted probabilities or decision scores from the two models,
        aligned to ``y_true``.
    model_names : tuple of str, optional
        Names for the printed summary. Defaults to ("Model 1", "Model 2").
    verbose : bool, default=True
        Print a formatted summary.
    return_values : bool, default=False
        If True, return ``(auc1, auc2, p_value)``.
    decimal_places : int, default=4
        Rounding for the printed summary.

    Returns
    -------
    tuple of float, optional
        ``(auc1, auc2, p_value)`` when ``return_values=True``.
    """
    y_true = np.asarray(y_true).astype(int)
    if set(np.unique(y_true)) - {0, 1}:
        raise ValueError("y_true must be binary (0/1).")
    m = int(y_true.sum())
    if m == 0 or m == len(y_true):
        raise ValueError("y_true must contain both classes.")

    order = (-y_true).argsort(kind="mergesort")  # positives first, stable
    preds = np.vstack((np.asarray(y_scores_1)[order], np.asarray(y_scores_2)[order]))
    aucs, cov = _fast_delong(preds, m)

    L = np.array([[1.0, -1.0]])
    var = float((L @ cov @ L.T).item())
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    p = float(2 * stats.norm.sf(abs(z)))
    auc1, auc2 = float(aucs[0]), float(aucs[1])

    if model_names is None:
        model_names = ("Model 1", "Model 2")

    if verbose:
        print(
            f"\nDeLong Test for Two Correlated ROC AUCs:\n"
            f"  {model_names[0]} AUC = {auc1:.{decimal_places}f}\n"
            f"  {model_names[1]} AUC = {auc2:.{decimal_places}f}\n"
            f"  difference   = {auc1 - auc2:+.{decimal_places}f}\n"
            f"  p-value      = {p:.{decimal_places}f}\n"
        )

    if return_values:
        return auc1, auc2, p


def hanley_mcneil_auc_test(
    y_true,
    y_scores_1,
    y_scores_2,
    model_names=None,
    verbose=True,
    return_values=False,
    decimal_places=4,
):
    """
    Hanley & McNeil (1982) large-sample z-test for difference in correlated AUCs.
    Returns (auc1, auc2, p_value).

    Parameters
    ----------
    y_true : array-like
        True binary class labels.
    y_scores_1 : array-like
        Predicted probabilities or decision scores from the first model.
    y_scores_2 : array-like
        Predicted probabilities or decision scores from the second model.
    model_names : list or tuple of str, optional
        Optional names for the models, used for printed output.
        Defaults to ("Model 1", "Model 2") if not provided.
    verbose : bool, default=True
        If True, prints a formatted summary of the comparison, including AUCs
        and the computed p-value.
    return_values : bool, default=False
        If True, returns the tuple (auc1, auc2, p_value) instead of only
        printing the results. This is useful for programmatic access or when
        integrating into other functions such as `show_roc_curve()`.

    Returns
    -------
    tuple of floats, optional
        (auc1, auc2, p_value) — only returned if `return_values=True`.
    """
    auc1 = roc_auc_score(y_true, y_scores_1)
    auc2 = roc_auc_score(y_true, y_scores_2)
    n1 = np.sum(y_true)
    n2 = len(y_true) - n1
    q1 = auc1 / (2 - auc1)
    q2 = 2 * auc1**2 / (1 + auc1)
    se = np.sqrt(
        (auc1 * (1 - auc1) + (n1 - 1) * (q1 - auc1**2) + (n2 - 1) * (q2 - auc1**2))
        / (n1 * n2)
    )
    z = (auc1 - auc2) / se
    p = 2 * (1 - norm.cdf(abs(z)))

    if model_names is None:
        model_names = ("Model 1", "Model 2")

    if verbose:
        print(
            f"\nHanley & McNeil AUC Comparison (Approximation of DeLong's Test):\n"
            f"  {model_names[0]} AUC = {auc1:.{decimal_places}f}\n"
            f"  {model_names[1]} AUC = {auc2:.{decimal_places}f}\n"
            f"  p-value = {p:.{decimal_places}f}\n"
        )

    if return_values:
        return auc1, auc2, p
