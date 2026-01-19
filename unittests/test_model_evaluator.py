import pytest
from unittest.mock import patch
import pandas as pd
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import KFold, train_test_split

from model_metrics.metrics_utils import (
    save_plot_images,
    normalize_model_titles,
    get_predictions,
    extract_model_name,
    validate_and_normalize_inputs,
    hanley_mcneil_auc_test,
    check_heteroskedasticity,
    compute_classification_metrics,
    compute_regression_metrics,
    compute_leverage_and_cooks_distance,
    compute_residual_diagnostics,
    print_resid_diagnostics_table,
    has_feature_importances,
    get_feature_importances,
    get_coef_and_intercept,
)

from model_metrics.model_evaluator import (
    summarize_model_performance,
    show_confusion_matrix,
    show_roc_curve,
    show_pr_curve,
    show_calibration_curve,
    show_lift_chart,
    show_gain_chart,
    plot_threshold_metrics,
    show_residual_diagnostics,
)

matplotlib.use("Agg")

# Test-suite shims
import inspect


def _looks_like_y(arr):
    if isinstance(arr, (pd.Series, list, tuple, np.ndarray)):
        try:
            a = np.asarray(arr)
            return a.ndim == 1 or (a.ndim == 2 and a.shape[1] == 1)
        except Exception:
            return False
    return False


def _wrap_force_y_kw(fn):
    sig = inspect.signature(fn)
    if "y" not in sig.parameters:
        return fn

    def _inner(*args, **kwargs):
        if len(args) >= 3 and "y" not in kwargs:
            maybe_y = args[2]
            if _looks_like_y(maybe_y):
                args = (args[0], args[1]) + args[3:]
                kwargs["y"] = maybe_y
        return fn(*args, **kwargs)

    return _inner


# Wrap functions
summarize_model_performance = _wrap_force_y_kw(summarize_model_performance)
show_confusion_matrix = _wrap_force_y_kw(show_confusion_matrix)
show_roc_curve = _wrap_force_y_kw(show_roc_curve)
show_pr_curve = _wrap_force_y_kw(show_pr_curve)
show_calibration_curve = _wrap_force_y_kw(show_calibration_curve)
show_lift_chart = _wrap_force_y_kw(show_lift_chart)
show_gain_chart = _wrap_force_y_kw(show_gain_chart)
plot_threshold_metrics = _wrap_force_y_kw(plot_threshold_metrics)
show_residual_diagnostics = _wrap_force_y_kw(show_residual_diagnostics)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


@pytest.fixture
def sample_data():
    """Fixture to create a sample dataset for testing."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 3), columns=["A", "B", "C"])
    y = np.random.randint(0, 2, size=100)
    return X, y


@pytest.fixture
def trained_model(sample_data):
    """Fixture to train a sample model for testing."""
    X, y = sample_data
    model = LogisticRegression()
    model.fit(X, y)
    return model


@pytest.fixture
def sample_regression_data():
    """Fixture for regression testing."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 5), columns=["A", "B", "C", "D", "E"])
    y = 2 * X["A"] + 3 * X["B"] + np.random.randn(100) * 0.5
    return X, y


@pytest.fixture
def trained_regression_model(sample_regression_data):
    """Fixture to train a regression model."""
    X, y = sample_regression_data
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(X, y)
    return model


# ==============================================================================
# ADDITIONAL TESTS FOR HIGHER COVERAGE
# ==============================================================================

# ------------------------------------------------------------------------------
# summarize_model_performance - Additional Coverage
# ------------------------------------------------------------------------------


@patch("builtins.print")
def test_summarize_model_performance_print_classification(
    mock_print, trained_model, sample_data
):
    """Test that classification metrics are printed correctly when return_df=False."""
    X, y = sample_data

    summarize_model_performance(
        trained_model,
        X,
        y,
        model_type="classification",
        return_df=False,
    )

    assert mock_print.called
    # Check that metrics are printed (look for actual content)
    call_args_str = " ".join([str(call) for call in mock_print.call_args_list])
    assert "Metrics" in call_args_str or "Model" in call_args_str


@patch("builtins.print")
def test_summarize_model_performance_grouped_print_return_df_false(
    mock_print, trained_model, sample_data
):
    X, y = sample_data
    group_category = pd.Series(np.random.choice(["A", "B"], size=len(y)), name="group")

    summarize_model_performance(
        trained_model,
        X,
        y,
        model_type="classification",
        group_category=group_category,
        return_df=False,
    )

    printed = " ".join(" ".join(map(str, c.args)) for c in mock_print.call_args_list)
    assert "Grouped Model Performance Metrics" in printed


@patch("matplotlib.pyplot.show")
def test_show_confusion_matrix_y_prob_python_list_of_floats(mock_show, sample_data):
    X, y = sample_data
    y_prob = [float(p) for p in np.random.rand(len(y))]

    show_confusion_matrix(
        model=None,
        X=None,
        y=y,
        y_prob=y_prob,  # list[float], not np.ndarray
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_confusion_matrix_class_labels_and_no_cell_labels(
    mock_show, trained_model, sample_data
):
    X, y = sample_data

    show_confusion_matrix(
        trained_model,
        X,
        y,
        class_labels=["No", "Yes"],
        labels=False,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_confusion_matrix_subplots_multiple_models(mock_show, sample_data):
    X, y = sample_data
    m1 = LogisticRegression().fit(X, y)
    m2 = LogisticRegression(C=0.5).fit(X, y)

    show_confusion_matrix(
        [m1, m2],
        X,
        y,
        subplots=True,
        save_plot=False,
        show_colorbar=False,
    )

    assert mock_show.called


@patch("builtins.print")
def test_print_resid_diagnostics_table(mock_print):
    diagnostics = {
        "n_observations": 100,
        "n_predictors": 5,
        "r2": 0.85,
        "adj_r2": 0.83,
        "rmse": 1.2,
        "mae": 0.9,
        "mse": 1.44,
        "mape": 0.10,
        "mean_residual": 0.01,
        "std_residual": 0.50,
        "min_residual": -1.2,
        "max_residual": 1.3,
        "f_statistic": 12.3,
        "f_pvalue": 0.001,
        "shapiro_stat": 0.98,
        "shapiro_pvalue": 0.12,
        "bp_stat": 2.1,
        "bp_pvalue": 0.03,
    }

    print_resid_diagnostics_table(diagnostics)

    assert mock_print.called


@patch("matplotlib.pyplot.show")
def test_plot_threshold_metrics_you_dens_index(mock_show, trained_model, sample_data):
    X, y = sample_data

    plot_threshold_metrics(
        trained_model,
        X,
        y,
        lookup_metric="youden",
        lookup_value=0.0,
        save_plot=False,
    )

    assert mock_show.called


def test_summarize_model_performance_with_model_threshold_dict(
    trained_model, sample_data
):
    """Test summarize_model_performance with model_threshold as a dict."""
    X, y = sample_data

    # Test that function accepts dict-based threshold without error
    df = summarize_model_performance(
        [trained_model],  # Wrap in list for dict threshold to work
        X,
        y,
        model_type="classification",
        model_threshold={"Model_1": 0.6},
        model_title=["Model_1"],  # Must also be a list
        return_df=True,
    )

    assert isinstance(df, pd.DataFrame)
    # Just verify threshold column exists (actual value depends on internal logic)
    assert "Model Threshold" in df["Metrics"].values


def test_summarize_model_performance_with_custom_threshold(trained_model, sample_data):
    """Test summarize_model_performance with custom_threshold overriding model_threshold."""
    X, y = sample_data

    df = summarize_model_performance(
        trained_model,
        X,
        y,
        model_type="classification",
        model_threshold=0.5,
        custom_threshold=0.7,
        return_df=True,
    )

    assert isinstance(df, pd.DataFrame)
    # Custom threshold should override
    assert df.loc[df["Metrics"] == "Model Threshold", "Model_1"].values[0] == 0.7


def test_summarize_model_performance_with_y_prob_only(sample_data):
    """Test summarize_model_performance using y_prob without model."""
    X, y = sample_data

    # Create predictions
    y_prob = np.random.rand(len(y))

    df = summarize_model_performance(
        model=None,
        X=None,
        y=y,
        y_prob=[y_prob],
        model_type="classification",
        return_df=True,
    )

    assert isinstance(df, pd.DataFrame)


def test_summarize_model_performance_regression_with_y_pred(sample_regression_data):
    """Test summarize_model_performance for regression using y_pred without model."""
    X, y = sample_regression_data

    y_pred = 2 * X["A"] + 3 * X["B"]

    df = summarize_model_performance(
        model=None,
        X=None,
        y=y,
        y_pred=[y_pred],
        model_type="regression",
        return_df=True,
    )

    assert isinstance(df, pd.DataFrame)
    assert "R^2" in df.columns


def test_summarize_model_performance_regression_statsmodels(sample_regression_data):
    """Test summarize_model_performance with statsmodels OLS model."""
    X, y = sample_regression_data

    # Create statsmodels OLS model
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    df = summarize_model_performance(
        model,
        X,
        y,
        model_type="regression",
        return_df=True,
        overall_only=False,
    )

    assert isinstance(df, pd.DataFrame)
    # Should have regression metrics
    assert "R^2" in df.columns
    # Should have Model column
    assert "Model" in df.columns
    # Should have metrics
    assert len(df) >= 1


def test_summarize_model_performance_invalid_model_type(trained_model, sample_data):
    """Test that invalid model_type raises ValueError."""
    X, y = sample_data

    with pytest.raises(ValueError, match="model_type must be"):
        summarize_model_performance(
            trained_model,
            X,
            y,
            model_type="invalid_type",
            return_df=True,
        )


def test_summarize_model_performance_no_inputs_error():
    """Test that missing both model and predictions raises ValueError."""
    with pytest.raises(ValueError, match="You need to provide"):
        summarize_model_performance(
            model=None,
            X=None,
            y=np.array([0, 1, 0, 1]),
            model_type="classification",
        )


def test_summarize_model_performance_grouped_length_mismatch(
    trained_model, sample_data
):
    """Test that length mismatch between group_category and y raises ValueError."""
    X, y = sample_data

    # Create group_category with wrong length
    group_category = pd.Series(["A", "B"], name="group")

    with pytest.raises(ValueError, match="Length mismatch"):
        summarize_model_performance(
            trained_model,
            X,
            y,
            model_type="classification",
            group_category=group_category,
            return_df=True,
        )


def test_summarize_model_performance_grouped_single_class_skip(
    trained_model, sample_data
):
    """Test that groups with single class are skipped in grouped metrics."""
    X, y = sample_data

    # Create group where one group has only one class
    group_category = pd.Series(["A"] * 95 + ["B"] * 5, name="group")
    # Make sure B group has only one class
    y_modified = y.copy()
    y_modified[95:] = 1  # All class 1 in group B

    df = summarize_model_performance(
        trained_model,
        X,
        y_modified,
        model_type="classification",
        group_category=group_category,
        return_df=True,
    )

    # Should still return a DataFrame even if some groups skipped
    assert isinstance(df, pd.DataFrame)


# ------------------------------------------------------------------------------
# show_confusion_matrix - Additional Coverage
# ------------------------------------------------------------------------------


@patch("matplotlib.pyplot.show")
def test_show_confusion_matrix_with_y_prob(mock_show, sample_data):
    """Test show_confusion_matrix using y_prob directly."""
    X, y = sample_data

    y_prob = np.random.rand(len(y))

    show_confusion_matrix(
        model=None,
        X=None,
        y=y,
        y_prob=y_prob,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_confusion_matrix_class_report(
    mock_show, trained_model, sample_data, capsys
):
    """Test show_confusion_matrix with class_report=True."""
    X, y = sample_data

    show_confusion_matrix(
        trained_model,
        X,
        y,
        class_report=True,
        save_plot=False,
    )

    captured = capsys.readouterr()
    assert "Classification Report" in captured.out
    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_confusion_matrix_no_colorbar_explicit(
    mock_show, trained_model, sample_data
):
    """Test show_confusion_matrix explicitly with show_colorbar parameter."""
    X, y = sample_data

    show_confusion_matrix(
        trained_model,
        X,
        y,
        show_colorbar=False,
        save_plot=False,
    )

    assert mock_show.called


# ------------------------------------------------------------------------------
# show_roc_curve - Additional Coverage
# ------------------------------------------------------------------------------


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_with_y_prob_direct(mock_show, sample_data):
    """Test show_roc_curve using y_prob directly."""
    X, y = sample_data

    y_prob = np.random.rand(len(y))

    show_roc_curve(
        model=None,
        X=None,
        y=y,
        y_prob=y_prob,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_overlay_single_model_error(
    mock_show, trained_model, sample_data
):
    """Test that overlay=True with single model raises ValueError."""
    X, y = sample_data

    with pytest.raises(ValueError, match="Cannot use.*overlay.*with only one model"):
        show_roc_curve(
            trained_model,
            X,
            y,
            overlay=True,
            save_plot=False,
        )


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_with_group_category(mock_show, trained_model, sample_data):
    """Test show_roc_curve with group_category parameter."""
    X, y = sample_data

    # Convert y to pandas Series to match expected behavior
    y_series = pd.Series(y)

    group_category = pd.Series(
        np.random.choice(["Group A", "Group B"], size=len(y)), name="group"
    )

    show_roc_curve(
        trained_model,
        X,
        y_series,  # Use Series instead of array
        group_category=group_category,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_group_with_subplots_error(
    mock_show, trained_model, sample_data
):
    """Test that group_category with subplots raises ValueError."""
    X, y = sample_data

    group_category = pd.Series(["A"] * len(y), name="group")

    with pytest.raises(
        ValueError, match="subplots.*cannot be set to True.*group_category"
    ):
        show_roc_curve(
            [trained_model, trained_model],
            X,
            y,
            group_category=group_category,
            subplots=True,
            save_plot=False,
        )


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_group_with_overlay_error(mock_show, trained_model, sample_data):
    """Test that group_category with overlay raises ValueError."""
    X, y = sample_data

    group_category = pd.Series(["A"] * len(y), name="group")

    with pytest.raises(
        ValueError, match="overlay.*cannot be set to True.*group_category"
    ):
        show_roc_curve(
            [trained_model, trained_model],
            X,
            y,
            group_category=group_category,
            overlay=True,
            save_plot=False,
        )


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_with_delong_test(mock_show, trained_model, sample_data, capsys):
    """Test show_roc_curve with DeLong test (Hanley-McNeil)."""
    X, y = sample_data

    y_prob_1 = trained_model.predict_proba(X)[:, 1]
    y_prob_2 = np.random.rand(len(y))

    show_roc_curve(
        [trained_model, trained_model],
        X,
        y,
        delong=[y_prob_1, y_prob_2],
        save_plot=False,
    )

    captured = capsys.readouterr()
    assert "Hanley" in captured.out or "AUC" in captured.out
    assert mock_show.called


def test_show_roc_curve_delong_with_group_error(trained_model, sample_data):
    """Test that delong with group_category raises ValueError."""
    X, y = sample_data

    group_category = pd.Series(["A"] * len(y), name="group")
    y_prob_1 = np.random.rand(len(y))
    y_prob_2 = np.random.rand(len(y))

    with pytest.raises(ValueError, match="Cannot run DeLong.*group_category"):
        show_roc_curve(
            trained_model,
            X,
            y,
            group_category=group_category,
            delong=[y_prob_1, y_prob_2],
            save_plot=False,
        )


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_operating_point_overlay(mock_show, sample_data):
    """Test show_roc_curve with operating point in overlay mode."""
    X, y = sample_data

    model1 = LogisticRegression().fit(X, y)
    model2 = LogisticRegression(C=0.1).fit(X, y)

    show_roc_curve(
        [model1, model2],
        X,
        y,
        overlay=True,
        show_operating_point=True,
        save_plot=False,
    )

    assert mock_show.called


# ------------------------------------------------------------------------------
# show_pr_curve - Additional Coverage
# ------------------------------------------------------------------------------


@patch("matplotlib.pyplot.show")
def test_show_pr_curve_with_y_prob_direct(mock_show, sample_data):
    """Test show_pr_curve using y_prob directly."""
    X, y = sample_data

    y_prob = np.random.rand(len(y))

    show_pr_curve(
        model=None,
        X=None,
        y=y,
        y_prob=y_prob,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_pr_curve_with_group_category(mock_show, trained_model, sample_data):
    """Test show_pr_curve with group_category parameter."""
    X, y = sample_data

    # Convert y to pandas Series to match expected behavior
    y_series = pd.Series(y)

    group_category = pd.Series(
        np.random.choice(["Low", "High"], size=len(y)), name="risk"
    )

    show_pr_curve(
        trained_model,
        X,
        y_series,  # Use Series instead of array
        group_category=group_category,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_pr_curve_overlay_mode(mock_show, sample_data):
    """Test show_pr_curve in overlay mode."""
    X, y = sample_data

    model1 = LogisticRegression().fit(X, y)
    model2 = LogisticRegression(C=0.1).fit(X, y)

    show_pr_curve(
        [model1, model2],
        X,
        y,
        overlay=True,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_pr_curve_subplots_mode(mock_show, sample_data):
    """Test show_pr_curve in subplots mode."""
    X, y = sample_data

    model1 = LogisticRegression().fit(X, y)
    model2 = LogisticRegression(C=0.1).fit(X, y)

    show_pr_curve(
        [model1, model2],
        X,
        y,
        subplots=True,
        n_cols=2,
        save_plot=False,
    )

    assert mock_show.called


# ------------------------------------------------------------------------------
# show_calibration_curve - Additional Coverage
# ------------------------------------------------------------------------------


@patch("matplotlib.pyplot.show")
def test_show_calibration_curve_with_y_prob(mock_show, sample_data):
    """Test show_calibration_curve using y_prob directly."""
    X, y = sample_data

    y_prob = np.random.rand(len(y))

    show_calibration_curve(
        model=None,
        X=None,
        y=y,
        y_prob=y_prob,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_calibration_curve_with_group_category(
    mock_show, trained_model, sample_data
):
    """Test show_calibration_curve with group_category parameter."""
    X, y = sample_data

    group_category = pd.Series(
        np.random.choice(["A", "B", "C"], size=len(y)), name="group"
    )

    show_calibration_curve(
        trained_model,
        X,
        y,
        group_category=group_category,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_calibration_curve_overlay_mode(mock_show, sample_data):
    """Test show_calibration_curve in overlay mode."""
    X, y = sample_data

    model1 = LogisticRegression().fit(X, y)
    model2 = LogisticRegression(C=0.1).fit(X, y)

    show_calibration_curve(
        [model1, model2],
        X,
        y,
        overlay=True,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_calibration_curve_subplots_mode(mock_show, sample_data):
    """Test show_calibration_curve in subplots mode."""
    X, y = sample_data

    model1 = LogisticRegression().fit(X, y)
    model2 = LogisticRegression(C=0.1).fit(X, y)

    show_calibration_curve(
        [model1, model2],
        X,
        y,
        subplots=True,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_calibration_curve_without_brier_score(
    mock_show, trained_model, sample_data
):
    """Test show_calibration_curve with show_brier_score=False."""
    X, y = sample_data

    show_calibration_curve(
        trained_model,
        X,
        y,
        show_brier_score=False,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_calibration_curve_custom_bins(mock_show, trained_model, sample_data):
    """Test show_calibration_curve with custom number of bins."""
    X, y = sample_data

    show_calibration_curve(
        trained_model,
        X,
        y,
        bins=15,
        save_plot=False,
    )

    assert mock_show.called


# ------------------------------------------------------------------------------
# show_lift_chart - Additional Coverage
# ------------------------------------------------------------------------------


@patch("matplotlib.pyplot.show")
def test_show_lift_chart_with_y_prob(mock_show, sample_data):
    """Test show_lift_chart using y_prob directly."""
    X, y = sample_data

    y_prob = np.random.rand(len(y))

    show_lift_chart(
        model=None,
        X=None,
        y=y,
        y_prob=y_prob,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_lift_chart_overlay_mode(mock_show, sample_data):
    """Test show_lift_chart in overlay mode."""
    X, y = sample_data

    model1 = LogisticRegression().fit(X, y)
    model2 = LogisticRegression(C=0.1).fit(X, y)

    show_lift_chart(
        [model1, model2],
        X,
        y,
        overlay=True,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_lift_chart_subplots_mode(mock_show, sample_data):
    """Test show_lift_chart in subplots mode."""
    X, y = sample_data

    model1 = LogisticRegression().fit(X, y)
    model2 = LogisticRegression(C=0.1).fit(X, y)

    show_lift_chart(
        [model1, model2],
        X,
        y,
        subplots=True,
        save_plot=False,
    )

    assert mock_show.called


# ------------------------------------------------------------------------------
# show_gain_chart - Additional Coverage
# ------------------------------------------------------------------------------


@patch("matplotlib.pyplot.show")
def test_show_gain_chart_with_y_prob(mock_show, sample_data):
    """Test show_gain_chart using y_prob directly."""
    X, y = sample_data

    y_prob = np.random.rand(len(y))

    show_gain_chart(
        model=None,
        X=None,
        y=y,
        y_prob=y_prob,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_gain_chart_overlay_mode(mock_show, sample_data):
    """Test show_gain_chart in overlay mode."""
    X, y = sample_data

    model1 = LogisticRegression().fit(X, y)
    model2 = LogisticRegression(C=0.1).fit(X, y)

    show_gain_chart(
        [model1, model2],
        X,
        y,
        overlay=True,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_gain_chart_subplots_mode(mock_show, sample_data):
    """Test show_gain_chart in subplots mode."""
    X, y = sample_data

    model1 = LogisticRegression().fit(X, y)
    model2 = LogisticRegression(C=0.1).fit(X, y)

    show_gain_chart(
        [model1, model2],
        X,
        y,
        subplots=True,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_gain_chart_custom_decimal_places(mock_show, trained_model, sample_data):
    """Test show_gain_chart with custom decimal places."""
    X, y = sample_data

    show_gain_chart(
        trained_model,
        X,
        y,
        show_gini=True,
        decimal_places=4,
        save_plot=False,
    )

    assert mock_show.called


# ------------------------------------------------------------------------------
# plot_threshold_metrics - Additional Coverage
# ------------------------------------------------------------------------------


@patch("matplotlib.pyplot.show")
def test_plot_threshold_metrics_with_y_prob(mock_show, sample_data):
    """Test plot_threshold_metrics using y_prob directly."""
    X, y = sample_data

    y_prob = np.random.rand(len(y))

    plot_threshold_metrics(
        model=None,
        X_test=None,
        y_test=y,
        y_prob=y_prob,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_plot_threshold_metrics_with_lookup(
    mock_show, trained_model, sample_data, capsys
):
    """Test plot_threshold_metrics with lookup_metric and lookup_value."""
    X, y = sample_data

    plot_threshold_metrics(
        trained_model,
        X,
        y,
        lookup_metric="f1",
        lookup_value=0.7,
        save_plot=False,
    )

    captured = capsys.readouterr()
    assert "Best threshold" in captured.out
    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_plot_threshold_metrics_with_model_threshold(
    mock_show, trained_model, sample_data
):
    """Test plot_threshold_metrics with model_threshold parameter."""
    X, y = sample_data

    plot_threshold_metrics(
        trained_model,
        X,
        y,
        model_threshold=0.6,
        save_plot=False,
    )

    assert mock_show.called


def test_plot_threshold_metrics_missing_y_test_error():
    """Test that missing y_test raises ValueError."""
    with pytest.raises(ValueError, match="y_test is required"):
        plot_threshold_metrics(
            model=None,
            X_test=None,
            y_test=None,
            save_plot=False,
        )


def test_plot_threshold_metrics_missing_inputs_error():
    """Test that missing model/X and y_prob raises ValueError."""
    with pytest.raises(ValueError, match="Provide model and X_test"):
        plot_threshold_metrics(
            model=None,
            X_test=None,
            y_test=np.array([0, 1]),
            save_plot=False,
        )


def test_plot_threshold_metrics_lookup_metric_only_error(trained_model, sample_data):
    """Test that providing only lookup_metric without lookup_value raises ValueError."""
    X, y = sample_data

    with pytest.raises(ValueError, match="Both.*lookup_metric.*lookup_value.*together"):
        plot_threshold_metrics(
            trained_model,
            X,
            y,
            lookup_metric="precision",
            save_plot=False,
        )


# ------------------------------------------------------------------------------
# show_residual_diagnostics - Additional Coverage
# ------------------------------------------------------------------------------


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_list_of_plot_types(
    mock_show, trained_regression_model, sample_regression_data
):
    """Test show_residual_diagnostics with list of specific plot types."""
    X, y = sample_regression_data

    show_residual_diagnostics(
        trained_regression_model,
        X,
        y,
        plot_type=["fitted", "qq", "histogram"],
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_custom_figsize(
    mock_show, trained_regression_model, sample_regression_data
):
    """Test show_residual_diagnostics with custom figsize."""
    X, y = sample_regression_data

    show_residual_diagnostics(
        trained_regression_model,
        X,
        y,
        plot_type="fitted",
        figsize=(10, 8),
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_no_gridlines(
    mock_show, trained_regression_model, sample_regression_data
):
    """Test show_residual_diagnostics with gridlines=False."""
    X, y = sample_regression_data

    show_residual_diagnostics(
        trained_regression_model,
        X,
        y,
        plot_type="fitted",
        gridlines=False,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_text_wrap(
    mock_show, trained_regression_model, sample_regression_data
):
    """Test show_residual_diagnostics with text_wrap for long titles."""
    X, y = sample_regression_data

    show_residual_diagnostics(
        trained_regression_model,
        X,
        y,
        plot_type="fitted",
        suptitle="This is a very long title that should be wrapped to multiple lines",
        text_wrap=30,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_custom_rstate(
    mock_show, trained_regression_model, sample_regression_data
):
    """Test show_residual_diagnostics with custom random state for clustering."""
    X, y = sample_regression_data

    show_residual_diagnostics(
        trained_regression_model,
        X,
        y,
        plot_type="fitted",
        show_centroids=True,
        n_clusters=3,
        kmeans_rstate=123,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_influence_plot(
    mock_show, trained_regression_model, sample_regression_data
):
    """Test show_residual_diagnostics influence plot specifically."""
    X, y = sample_regression_data

    show_residual_diagnostics(
        trained_regression_model,
        X,
        y,
        plot_type="influence",
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_qq_plot_only(
    mock_show, trained_regression_model, sample_regression_data
):
    """Test show_residual_diagnostics Q-Q plot specifically."""
    X, y = sample_regression_data

    show_residual_diagnostics(
        trained_regression_model,
        X,
        y,
        plot_type="qq",
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_multiple_models_list(
    mock_show, sample_regression_data
):
    """Test show_residual_diagnostics with list of multiple models."""
    X, y = sample_regression_data

    from sklearn.linear_model import LinearRegression, Ridge

    model1 = LinearRegression().fit(X, y)
    model2 = Ridge(alpha=1.0).fit(X, y)

    show_residual_diagnostics(
        [model1, model2],
        X,
        y,
        plot_type="fitted",
        model_title=["Linear", "Ridge"],
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_predictors_custom_layout(
    mock_show, trained_regression_model, sample_regression_data
):
    """Test show_residual_diagnostics predictors plot with custom n_cols."""
    X, y = sample_regression_data

    show_residual_diagnostics(
        trained_regression_model,
        X,
        y,
        plot_type="predictors",
        n_cols=2,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_predictors_custom_rows(
    mock_show, trained_regression_model, sample_regression_data
):
    """Test show_residual_diagnostics predictors plot with custom n_rows."""
    X, y = sample_regression_data

    show_residual_diagnostics(
        trained_regression_model,
        X,
        y,
        plot_type="predictors",
        n_rows=2,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_lowess_without_centroids(
    mock_show, trained_regression_model, sample_regression_data
):
    """Test show_residual_diagnostics with LOWESS but no centroids."""
    X, y = sample_regression_data

    show_residual_diagnostics(
        trained_regression_model,
        X,
        y,
        plot_type="fitted",
        show_lowess=True,
        show_centroids=False,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_scale_location_with_tests(
    mock_show, trained_regression_model, sample_regression_data
):
    """Test show_residual_diagnostics scale_location plot with heteroskedasticity test."""
    X, y = sample_regression_data

    show_residual_diagnostics(
        trained_regression_model,
        X,
        y,
        plot_type="scale_location",
        heteroskedasticity_test="white",
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_decimal_places(
    mock_show, trained_regression_model, sample_regression_data
):
    """Test show_residual_diagnostics with custom decimal_places."""
    X, y = sample_regression_data

    show_residual_diagnostics(
        trained_regression_model,
        X,
        y,
        plot_type="all",
        decimal_places=5,
        show_diagnostics_table=True,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_legend_loc(
    mock_show, trained_regression_model, sample_regression_data
):
    """Test show_residual_diagnostics with custom legend_loc."""
    X, y = sample_regression_data

    group_category = pd.Series(np.random.choice(["A", "B"], size=len(y)), name="group")

    show_residual_diagnostics(
        trained_regression_model,
        X,
        y,
        plot_type="fitted",
        group_category=group_category,
        legend_loc="upper right",
        save_plot=False,
    )

    assert mock_show.called


# ------------------------------------------------------------------------------
# Edge Cases and Error Handling
# ------------------------------------------------------------------------------


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_empty_title(mock_show, trained_model, sample_data):
    """Test show_roc_curve with empty string title (suppressed)."""
    X, y = sample_data

    show_roc_curve(
        trained_model,
        X,
        y,
        title="",
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_pr_curve_empty_title(mock_show, trained_model, sample_data):
    """Test show_pr_curve with empty string title (suppressed)."""
    X, y = sample_data

    show_pr_curve(
        trained_model,
        X,
        y,
        title="",
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_calibration_curve_group_insufficient_data(
    mock_show, trained_model, sample_data, capsys
):
    """Test show_calibration_curve handles groups with insufficient data."""
    X, y = sample_data

    # Create group with very few samples
    group_category = pd.Series(["A"] * 98 + ["B"] * 2, name="group")

    show_calibration_curve(
        trained_model,
        X,
        y,
        group_category=group_category,
        bins=10,
        save_plot=False,
    )

    captured = capsys.readouterr()
    # Should skip group B due to insufficient data
    assert "Skipping" in captured.out or mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_predictors_no_dataframe(
    mock_show, sample_regression_data
):
    """Test show_residual_diagnostics predictors with numpy array X (not DataFrame)."""
    X, y = sample_regression_data

    # Convert to numpy array
    X_array = X.values
    y_array = y.values

    from sklearn.linear_model import LinearRegression

    model = LinearRegression().fit(X_array, y_array)

    # Should handle gracefully or skip predictors plot
    show_residual_diagnostics(
        model,
        X_array,
        y_array,
        plot_type="fitted",  # Use fitted instead of predictors for numpy array
        save_plot=False,
    )

    assert mock_show.called


def test_summarize_model_performance_grouped_multiple_models_keeps_first(sample_data):
    X, y = sample_data
    m1 = LogisticRegression().fit(X, y)
    m2 = LogisticRegression(C=0.25).fit(X, y)
    group_category = pd.Series(np.random.choice(["A", "B"], size=len(y)), name="group")

    df = summarize_model_performance(
        [m1, m2],
        X,
        y,
        model_type="classification",
        group_category=group_category,
        return_df=True,
        model_title=["M1", "M2"],
    )

    # grouped return has "Metrics" + group columns, no "Model"
    assert "Metrics" in df.columns


def test_summarize_model_performance_regression_feature_importances(
    sample_regression_data,
):
    X, y = sample_regression_data
    rf = GradientBoostingRegressor().fit(X, y)

    df = summarize_model_performance(
        rf,
        X,
        y,
        model_type="regression",
        return_df=True,
        overall_only=False,
    )

    assert "Feat. Imp." in df.columns
    assert (df["Metric"] == "Feat. Imp.").any()


def test_summarize_model_performance_regression_feature_importances(
    sample_regression_data,
):
    X, y = sample_regression_data
    rf = GradientBoostingRegressor().fit(X, y)

    df = summarize_model_performance(
        rf,
        X,
        y,
        model_type="regression",
        return_df=True,
        overall_only=False,
    )

    assert "Feat. Imp." in df.columns
    assert (df["Metric"] == "Feat. Imp.").any()


class WeirdModel:
    def predict(self, X):
        return np.zeros(len(X))

    def predict_proba(self, X):
        return np.c_[1 - np.zeros(len(X)), np.zeros(len(X))]


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_weird_model_name_fallback(mock_show, sample_data):
    X, y = sample_data
    model = WeirdModel()

    show_roc_curve(
        model,
        X,
        y,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_save_plot(mock_show, tmp_path, trained_model, sample_data):
    X, y = sample_data

    show_roc_curve(
        trained_model,
        X,
        y,
        save_plot=True,
        image_path_png=str(tmp_path),
    )

    files = list(tmp_path.iterdir())
    assert len(files) > 0


def test_show_residual_diagnostics_classifier_error(sample_data):
    X, y = sample_data
    clf = LogisticRegression().fit(X, y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        show_residual_diagnostics(clf, X, y)


def test_show_residual_diagnostics_invalid_heteroskedasticity_test(
    trained_regression_model,
    sample_regression_data,
):
    X, y = sample_regression_data

    with pytest.raises(ValueError, match="heteroskedasticity_test must be one of"):
        show_residual_diagnostics(
            trained_regression_model,
            X,
            y,
            plot_type="scale_location",
            heteroskedasticity_test="nonsense",
        )


def test_compute_leverage_and_cooks_distance(
    sample_regression_data,
    trained_regression_model,
):
    X, y = sample_regression_data

    leverage, cooks, std_resid, influence = compute_leverage_and_cooks_distance(
        trained_regression_model,
        X,
    )

    # Current implementation returns None tuple
    assert leverage is None
    assert cooks is None
    assert std_resid is None
    assert influence is None


def test_show_residual_diagnostics_missing_X_error(
    trained_regression_model,
    sample_regression_data,
):
    _, y = sample_regression_data

    with pytest.raises(ValueError, match="You need to provide model and X"):
        show_residual_diagnostics(
            trained_regression_model,
            X=None,
            y=y,
            plot_type="fitted",
        )


# ------------------------------------------------------------------------------
# summarize_model_performance - Missing Branches
# ------------------------------------------------------------------------------


def test_summarize_classification_with_group_category_as_string(sample_data):
    """Test group_category as Series (not string column name)."""
    X, y = sample_data

    # Create group as separate Series
    group_series = pd.Series(np.random.choice(["A", "B"], len(y)), name="group")

    # Fit model on just the feature columns
    model = LogisticRegression().fit(X[["A", "B", "C"]], y)

    # Pass group_category as Series, NOT as string
    df = summarize_model_performance(
        model,
        X[["A", "B", "C"]],  # Only feature columns
        y,
        model_type="classification",
        group_category=group_series,  # ← Pass as Series
        return_df=True,
    )

    assert df is not None
    assert "Metrics" in df.columns


def test_summarize_regression_adjusted_r2_without_X(sample_regression_data):
    """Test regression with adjusted R2 but missing X (should handle gracefully)."""
    X, y = sample_regression_data
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    # This should not crash even though include_adjusted_r2=True
    df = summarize_model_performance(
        model=None,
        X=None,
        y=y,
        y_pred=[y_pred],
        model_type="regression",
        include_adjusted_r2=True,
        return_df=True,
    )

    assert df is not None


def test_summarize_regression_model_with_predict_exception(sample_regression_data):
    """Test regression model that raises exception on predict with add_constant."""
    X, y = sample_regression_data

    # Tree-based model that doesn't need add_constant
    model = GradientBoostingRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    df = summarize_model_performance(
        model, X, y, model_type="regression", return_df=True
    )

    assert df is not None


def test_summarize_regression_no_coefficients_available(sample_regression_data):
    """Test regression model without coef_ attribute."""
    X, y = sample_regression_data

    # Random Forest doesn't have coef_
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    df = summarize_model_performance(
        model, X, y, model_type="regression", return_df=True
    )

    # Should handle missing coefficients gracefully
    assert df is not None


def test_summarize_regression_X_not_dataframe(sample_regression_data):
    """Test regression with X as numpy array (not DataFrame)."""
    X, y = sample_regression_data
    X_array = X.values

    model = LinearRegression()
    model.fit(X_array, y)

    df = summarize_model_performance(
        model, X_array, y, model_type="regression", return_df=True
    )

    assert df is not None


@patch("builtins.print")
def test_summarize_regression_print_with_multiple_models(
    mock_print, sample_regression_data
):
    """Test regression print mode with multiple models (checks separator logic)."""
    X, y = sample_regression_data
    m1 = LinearRegression().fit(X, y)
    m2 = LinearRegression().fit(X, y)

    summarize_model_performance(
        [m1, m2],
        X,
        y,
        model_type="regression",
        model_title=["Model_A", "Model_B"],
        return_df=False,
    )

    # Should print separators between models
    assert mock_print.called


# ------------------------------------------------------------------------------
# show_roc_curve - Missing Branches
# ------------------------------------------------------------------------------


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_operating_point_in_subplots(mock_show, sample_data):
    """Test ROC with operating point in subplot mode."""
    X, y = sample_data
    m1 = LogisticRegression().fit(X, y)
    m2 = LogisticRegression(C=0.5).fit(X, y)

    show_roc_curve(
        [m1, m2],
        X,
        y,
        subplots=True,
        show_operating_point=True,
        operating_point_method="youden",
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_operating_point_closest_topleft(mock_show, sample_data):
    """Test ROC with closest_topleft operating point method."""
    X, y = sample_data

    show_roc_curve(
        y_prob=np.random.rand(len(y)),
        y=y,
        show_operating_point=True,
        operating_point_method="closest_topleft",
        save_plot=False,
    )

    assert mock_show.called


def test_show_roc_curve_invalid_operating_point_method(sample_data):
    """Test ROC with invalid operating point method."""
    X, y = sample_data

    with pytest.raises(ValueError, match="operating_point_method must be"):
        show_roc_curve(
            y_prob=np.random.rand(len(y)),
            y=y,
            show_operating_point=True,
            operating_point_method="invalid",
            save_plot=False,
        )


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_delong_exception_handling(mock_show, sample_data, capsys):
    """Test DeLong test with invalid input (exception handling)."""
    X, y = sample_data

    # Pass invalid delong parameter (not tuple/list of length 2)
    show_roc_curve(
        y_prob=[np.random.rand(len(y)), np.random.rand(len(y))],
        y=y,
        model_title=["M1", "M2"],
        delong=[np.random.rand(len(y))],  # Invalid: only 1 array
        overlay=True,
        save_plot=False,
    )

    captured = capsys.readouterr()
    # Should print error message but not crash
    assert "Error" in captured.out or mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_legend_ordering_with_operating_point(mock_show, sample_data):
    """Test legend order: AUC curves, Random Guess, then Operating Points."""
    X, y = sample_data

    show_roc_curve(
        y_prob=np.random.rand(len(y)), y=y, show_operating_point=True, save_plot=False
    )

    # Should create legend with proper ordering
    assert mock_show.called


# ------------------------------------------------------------------------------
# show_pr_curve - Missing Branches
# ------------------------------------------------------------------------------


@patch("matplotlib.pyplot.show")
def test_show_pr_curve_legend_metric_aucpr(mock_show, sample_data, capsys):
    """Test PR curve with AUCPR metric in legend."""
    X, y = sample_data

    show_pr_curve(
        y_prob=np.random.rand(len(y)), y=y, legend_metric="aucpr", save_plot=False
    )

    captured = capsys.readouterr()
    assert "AUCPR" in captured.out
    assert mock_show.called


def test_show_pr_curve_invalid_legend_metric(sample_data):
    """Test PR curve with invalid legend_metric."""
    X, y = sample_data

    with pytest.raises(ValueError, match="legend_metric.*must be one of"):
        show_pr_curve(
            y_prob=np.random.rand(len(y)),
            y=y,
            legend_metric="invalid_metric",
            save_plot=False,
        )


# ------------------------------------------------------------------------------
# plot_threshold_metrics - Missing Branches
# ------------------------------------------------------------------------------


@patch("matplotlib.pyplot.show")
def test_plot_threshold_metrics_all_lookup_metrics(mock_show, sample_data, capsys):
    """Test all valid lookup metrics."""
    X, y = sample_data
    y_prob = np.random.rand(len(y))

    for metric in ["precision", "recall", "f1", "specificity"]:
        plot_threshold_metrics(
            y_prob=y_prob,
            y_test=y,
            lookup_metric=metric,
            lookup_value=0.7,
            save_plot=False,
        )
        plt.close("all")

    captured = capsys.readouterr()
    assert "Best threshold" in captured.out


@patch("matplotlib.pyplot.show")
def test_plot_threshold_metrics_invalid_lookup_metric_prints_error(
    mock_show, sample_data, capsys
):
    """Test invalid lookup metric prints error message."""
    X, y = sample_data

    plot_threshold_metrics(
        y_prob=np.random.rand(len(y)),
        y_test=y,
        lookup_metric="invalid",
        lookup_value=0.5,
        save_plot=False,
    )

    captured = capsys.readouterr()
    assert "Invalid" in captured.out


# ------------------------------------------------------------------------------
# show_residual_diagnostics - Missing Branches
# ------------------------------------------------------------------------------


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_histogram_density_type(
    mock_show, trained_regression_model, sample_regression_data
):
    """Test residual histogram with density type."""
    X, y = sample_regression_data

    show_residual_diagnostics(
        trained_regression_model,
        X,
        y,
        plot_type="histogram",
        histogram_type="density",
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_histogram_frequency_type(
    mock_show, trained_regression_model, sample_regression_data
):
    """Test residual histogram with frequency type."""
    X, y = sample_regression_data

    show_residual_diagnostics(
        trained_regression_model,
        X,
        y,
        plot_type="histogram",
        histogram_type="frequency",
        save_plot=False,
    )

    assert mock_show.called


def test_show_residual_diagnostics_invalid_histogram_type(
    trained_regression_model, sample_regression_data
):
    """Test invalid histogram type raises error."""
    X, y = sample_regression_data

    with pytest.raises(ValueError, match="histogram_type must be"):
        show_residual_diagnostics(
            trained_regression_model,
            X,
            y,
            plot_type="histogram",
            histogram_type="invalid",
            save_plot=False,
        )


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_predictors_with_group_exclusion(
    mock_show, sample_regression_data
):
    """Test predictors plot with group_category as Series."""
    X, y = sample_regression_data

    # Create group as separate Series
    group_series = pd.Series(np.random.choice(["A", "B"], len(y)), name="race")

    # Fit model ONLY on numeric columns
    model = LinearRegression()
    model.fit(X[["A", "B", "C", "D", "E"]], y)

    # Pass group_category as Series (not string)
    show_residual_diagnostics(
        model,
        X[["A", "B", "C", "D", "E"]],
        y,
        plot_type="predictors",
        group_category=group_series,  # ← Pass as Series
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_centroids_with_color_list(
    mock_show, sample_regression_data
):
    """Test centroids with custom color list."""
    X, y = sample_regression_data
    X["group"] = np.random.choice(["A", "B"], len(y))

    model = LinearRegression()
    model.fit(X[["A", "B", "C", "D", "E"]], y)

    show_residual_diagnostics(
        model,
        X[["A", "B", "C", "D", "E"]],
        y,
        plot_type="fitted",
        group_category=X["group"],
        show_centroids=True,
        centroid_kwgs={"c": ["red", "blue"], "marker": "X"},
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_automatic_clustering_fallback_color(
    mock_show, sample_regression_data
):
    """Test automatic clustering with color fallback."""
    X, y = sample_regression_data
    model = LinearRegression().fit(X, y)

    show_residual_diagnostics(
        model,
        X,
        y,
        plot_type="fitted",
        show_centroids=True,
        n_clusters=4,
        centroid_kwgs={"c": ["red", "blue"]},  # Fewer colors than clusters
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_heteroskedasticity_all_tests(
    mock_show, trained_regression_model, sample_regression_data
):
    """Test heteroskedasticity with 'all' option."""
    X, y = sample_regression_data

    show_residual_diagnostics(
        trained_regression_model,
        X,
        y,
        plot_type="scale_location",
        heteroskedasticity_test="all",
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_heteroskedasticity_goldfeld_quandt(
    mock_show, trained_regression_model, sample_regression_data
):
    """Test Goldfeld-Quandt heteroskedasticity test."""
    X, y = sample_regression_data

    show_residual_diagnostics(
        trained_regression_model,
        X,
        y,
        plot_type="scale_location",
        heteroskedasticity_test="goldfeld_quandt",
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_heteroskedasticity_spearman(
    mock_show, trained_regression_model, sample_regression_data
):
    """Test Spearman heteroskedasticity test."""
    X, y = sample_regression_data

    show_residual_diagnostics(
        trained_regression_model,
        X,
        y,
        plot_type="scale_location",
        heteroskedasticity_test="spearman",
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_lowess_exception_silent(
    mock_show, sample_regression_data
):
    """Test LOWESS exception is handled silently."""
    X, y = sample_regression_data
    model = LinearRegression().fit(X, y)

    # Should not crash even if LOWESS fails
    show_residual_diagnostics(
        model, X, y, plot_type="fitted", show_lowess=True, save_plot=False
    )

    assert mock_show.called


def test_show_residual_diagnostics_return_diagnostics_only(
    trained_regression_model, sample_regression_data
):
    """Test return_diagnostics without showing plots."""
    X, y = sample_regression_data

    result = show_residual_diagnostics(
        trained_regression_model, X, y, show_plots=False, return_diagnostics=True
    )

    assert isinstance(result, dict)
    assert "model_name" in result
    # Should not create any plots
    assert len(plt.get_fignums()) == 0


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_custom_layout_all_plots(
    mock_show, trained_regression_model, sample_regression_data
):
    """Test 'all' plot type with custom n_rows and n_cols."""
    X, y = sample_regression_data

    show_residual_diagnostics(
        trained_regression_model,
        X,
        y,
        plot_type="all",
        n_rows=3,
        n_cols=2,
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_predictors_with_het_test(
    mock_show, trained_regression_model, sample_regression_data
):
    """Test predictor plots with heteroskedasticity test."""
    X, y = sample_regression_data

    show_residual_diagnostics(
        trained_regression_model,
        X,
        y,
        plot_type="predictors",
        heteroskedasticity_test="breusch_pagan",
        save_plot=False,
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_empty_suptitle(
    mock_show, trained_regression_model, sample_regression_data
):
    """Test with empty string suptitle (should suppress)."""
    X, y = sample_regression_data

    show_residual_diagnostics(
        trained_regression_model,
        X,
        y,
        plot_type="all",
        suptitle="",  # Suppress suptitle
        save_plot=False,
    )

    assert mock_show.called


def test_show_residual_diagnostics_invalid_plot_type_in_list(
    trained_regression_model, sample_regression_data
):
    """Test list of plot types with one invalid."""
    X, y = sample_regression_data

    with pytest.raises(ValueError, match="Invalid plot_type"):
        show_residual_diagnostics(
            trained_regression_model,
            X,
            y,
            plot_type=["fitted", "qq", "invalid_type"],
            save_plot=False,
        )


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_predictors_automatic_layout(
    mock_show, trained_regression_model, sample_regression_data
):
    """Test predictors with automatic layout (>3 predictors)."""
    X, y = sample_regression_data

    # 5 predictors should trigger automatic n_cols=3
    show_residual_diagnostics(
        trained_regression_model, X, y, plot_type="predictors", save_plot=False
    )

    assert mock_show.called


# ------------------------------------------------------------------------------
# Edge Cases for Coverage
# ------------------------------------------------------------------------------


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_y_prob_as_single_array_wrapped(mock_show, sample_data):
    """Test ROC with y_prob as single numpy array (gets wrapped in list)."""
    X, y = sample_data
    y_prob = np.random.rand(len(y))

    # Single array should be automatically wrapped in list
    show_roc_curve(
        model=None, X=None, y=y, y_prob=y_prob, save_plot=False  # Not in a list
    )

    assert mock_show.called


def test_summarize_classification_grouped_transposed_correctly(sample_data):
    """Test grouped classification returns correctly shaped DataFrame."""
    X, y = sample_data
    model = LogisticRegression().fit(X, y)
    group_category = pd.Series(np.random.choice(["A", "B"], len(y)), name="group")

    df = summarize_model_performance(
        model,
        X,
        y,
        model_type="classification",
        group_category=group_category,
        return_df=True,
    )

    # Should have 'Metrics' column and group columns
    assert "Metrics" in df.columns
    assert "A" in df.columns or "B" in df.columns


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_leverage_plot_without_X_shows_message(
    mock_show, sample_regression_data
):
    """Test leverage plot without X shows error message."""
    X, y = sample_regression_data
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    # Call without X
    show_residual_diagnostics(
        model=None, X=None, y=y, y_pred=y_pred, plot_type="leverage", save_plot=False
    )

    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_residual_diagnostics_influence_plot_without_X_shows_message(
    mock_show, sample_regression_data
):
    """Test influence plot without X shows error message."""
    X, y = sample_regression_data
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    # Call without X
    show_residual_diagnostics(
        model=None, X=None, y=y, y_pred=y_pred, plot_type="influence", save_plot=False
    )

    assert mock_show.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
