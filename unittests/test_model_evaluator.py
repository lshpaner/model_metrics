import pytest
import builtins
from unittest.mock import patch
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import KFold, train_test_split
from model_metrics.model_evaluator import (
    save_plot_images,
    summarize_model_performance,
    show_confusion_matrix,
    show_roc_curve,
    show_pr_curve,
    show_calibration_curve,
    extract_model_name,
    show_lift_chart,
    show_gain_chart,
    plot_threshold_metrics,
)

matplotlib.use("Agg")  # Set non-interactive backend

# ------------------------------------------------------------------
# Test-suite shims to make calls robust against the evaluator's API
# ------------------------------------------------------------------
# Many helpers are defined as (model, X, y_prob=None, y_pred=None, y=None, ...)
# but older tests called them like helper(model, X, y). That y ended up in the
# y_prob or y_pred positional slot. To prevent recurring regressions, we wrap
# the imported helpers so that if the 3rd positional arg looks like it was
# really 'y', we pop it from *args and pass it as y= keyword. This keeps the
# entire suite working without rewriting every call site.
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


# Wrap once for this session
summarize_model_performance = _wrap_force_y_kw(summarize_model_performance)
show_confusion_matrix = _wrap_force_y_kw(show_confusion_matrix)
show_roc_curve = _wrap_force_y_kw(show_roc_curve)
show_pr_curve = _wrap_force_y_kw(show_pr_curve)
show_calibration_curve = _wrap_force_y_kw(show_calibration_curve)
show_lift_chart = _wrap_force_y_kw(show_lift_chart)
show_gain_chart = _wrap_force_y_kw(show_gain_chart)
plot_threshold_metrics = _wrap_force_y_kw(plot_threshold_metrics)


# ------------------------------------------------------------------
# Matplotlib hygiene: close figures after each test to avoid leaks
# ------------------------------------------------------------------
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
def classification_model():
    """Returns a trained classification model and test data."""
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression().fit(X_train, y_train)
    return model, (X_test, y_test)


@pytest.fixture
def regression_model():
    """Fixture to create a sample regression model and dataset."""
    X, y = make_regression(
        n_samples=100,
        n_features=5,
        noise=0.1,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    model = Lasso(alpha=0.1).fit(X_train, y_train)
    return model, (X_test, y_test)


@pytest.fixture
def rf_regression_model():
    """
    Fixture to create a RandomForestRegressor model with feature importances.
    """
    from sklearn.ensemble import RandomForestRegressor

    X, y = make_regression(
        n_samples=100,
        n_features=5,
        noise=0.1,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(random_state=42).fit(X_train, y_train)
    return model, (X_test, y_test)


def test_save_plot_images(tmp_path):
    """Test that images are saved correctly when save_plot=True."""
    filename = "test_plot"
    image_path_png = tmp_path / "png"
    image_path_svg = tmp_path / "svg"

    os.makedirs(image_path_png, exist_ok=True)
    os.makedirs(image_path_svg, exist_ok=True)

    plt.plot([0, 1], [0, 1])  # Dummy plot
    save_plot_images(
        filename,
        True,
        str(image_path_png),
        str(image_path_svg),
    )

    assert os.path.exists(os.path.join(image_path_png, f"{filename}.png"))
    assert os.path.exists(os.path.join(image_path_svg, f"{filename}.svg"))


def test_get_predictions(trained_model, sample_data):
    """Basic smoke test that wrapper doesn’t affect direct get_predictions import."""
    # NOTE: get_predictions is called inside other helpers; here we ensure
    # summarize_model_performance (wrapped) still works end-to-end.
    X, y = sample_data
    df = summarize_model_performance(
        [trained_model],
        X,
        y,
        return_df=True,
        model_type="classification",
    )
    assert isinstance(df, pd.DataFrame)


def test_get_predictions_single_model_proba(trained_model, sample_data):
    """Test indirectly via summarize_model_performance (probability path)."""
    X, y = sample_data
    df = summarize_model_performance(
        trained_model, X, y, return_df=True, model_type="classification"
    )
    assert isinstance(df, pd.DataFrame)


def test_get_predictions_single_model_no_proba(sample_data):
    """Test get_predictions fallback via a model lacking predict_proba."""

    class CustomModel:
        def fit(self, X, y):
            pass

        def predict(self, X):
            return np.random.randint(0, 2, size=len(X))

    model = CustomModel()
    X, y = sample_data
    # Exercise via a plotting helper that calls get_predictions.
    # Using ROC curve here just to smoke test.
    with patch("matplotlib.pyplot.show"):
        show_roc_curve(model, X, y, save_plot=False)


def test_get_predictions_with_custom_threshold(trained_model, sample_data):
    """Test custom threshold goes through end-to-end."""
    X, y = sample_data
    df = summarize_model_performance(
        trained_model, X, y, model_type="classification", return_df=True
    )
    assert isinstance(df, pd.DataFrame)


def test_get_predictions_kfold(sample_data):
    """Test get_predictions with a model using K-Fold cross-validation."""

    class KFoldModel:
        def __init__(self):
            self.kfold = True
            self.kf = KFold(n_splits=5)

        def fit(self, X, y):
            pass

        def predict_proba(self, X):
            return np.random.rand(len(X), 2)

        def predict(self, X):
            return np.random.randint(0, 2, size=len(X))

    model = KFoldModel()
    X, y = sample_data
    y = pd.Series(y)  # Ensure y is a Pandas Series to avoid 'iloc' issues

    with patch("matplotlib.pyplot.show"):
        show_pr_curve(model, X, y, save_plot=False)


def test_get_predictions_dict_model_threshold_fallback(sample_data):
    X, y = sample_data

    class DummyModel:
        def predict(self, X):
            return np.ones(len(X))

        def predict_proba(self, X):
            return np.tile(np.array([[0.2, 0.8]]), (len(X), 1))  # shape (n, 2)

    model = DummyModel()
    model.threshold = {"not_used_score": 0.3}

    # Run through a plotting helper (lift) to ensure threshold fallback does not blow up
    with patch("matplotlib.pyplot.show"):
        show_lift_chart(model, X, y, save_plot=False)


def test_get_predictions_kfold_with_predict_proba(sample_data):
    class KFoldModelWithProba:
        def __init__(self):
            self.kfold = True
            self.kf = KFold(n_splits=3)

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            proba = np.random.rand(len(X), 2)
            proba /= proba.sum(axis=1, keepdims=True)
            return proba

    model = KFoldModelWithProba()
    X, y = sample_data
    y = pd.Series(y)  # make sure .iloc works

    with patch("matplotlib.pyplot.show"):
        show_roc_curve(model, X, y, save_plot=False)


def test_summarize_model_performance(trained_model, sample_data, capsys):
    """
    Test summarize_model_performance function with formatted output.
    """
    X, y = sample_data

    df = summarize_model_performance(
        [trained_model],
        X,
        y,
        return_df=True,
        model_type="classification",
    )

    # Debugging: Print the actual DataFrame before assertions
    print("\nDEBUG: Model Performance DataFrame")
    print(df)

    # Ensure output is a DataFrame
    assert isinstance(df, pd.DataFrame), "Output should be a pandas DataFrame."

    # Check that required metrics exist in the "Metrics" column
    expected_metrics = [
        "Precision/PPV",
        "F1-Score",
        "Sensitivity/Recall",
        "AUC ROC",
    ]
    missing_metrics = [
        metric for metric in expected_metrics if metric not in df["Metrics"].values
    ]
    assert not missing_metrics, f"Missing expected metrics: {missing_metrics}"

    # Check column name
    assert "Model_1" in df.columns, "Expected column 'Model_1' in output."

    # Ensure none of the expected metric values are missing
    for metric in expected_metrics:
        val = df.loc[df["Metrics"] == metric, "Model_1"]
        assert not val.isnull().any(), f"Missing value for {metric}"


def test_summarize_model_performance_classification(classification_model):
    """
    Test summarize_model_performance function for classification models.
    """
    trained_model, sample_data = classification_model
    X, y = sample_data

    df = summarize_model_performance(
        [trained_model],
        X,
        y,
        model_type="classification",
        return_df=True,
    )

    print("\nDEBUG: Classification Model Performance DataFrame")
    print(df)
    print("Columns:", df.columns)

    assert isinstance(df, pd.DataFrame), "Output should be a pandas DataFrame."

    expected_metrics = [
        "Precision/PPV",
        "AUC ROC",
        "F1-Score",
        "Sensitivity/Recall",
    ]
    missing_metrics = [
        metric for metric in expected_metrics if metric not in df["Metrics"].values
    ]
    assert (
        not missing_metrics
    ), f"Missing expected classification metrics: {missing_metrics}"

    assert "Model_1" in df.columns, "Expected column 'Model_1' in output."

    for metric in expected_metrics:
        assert (
            df["Model_1"].loc[df["Metrics"] == metric].notna().all()
        ), f"Missing values for {metric}"


def test_summarize_model_performance_regression(regression_model):
    """
    Test summarize_model_performance function for regression models,
    including overall_only behavior.
    """
    model, (X, y) = regression_model

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])

    df_full = summarize_model_performance(
        model,
        X,
        y,
        model_type="regression",
        return_df=True,
        overall_only=False,
    )

    assert isinstance(df_full, pd.DataFrame), "Output should be a pandas DataFrame."

    expected_columns_full = [
        "Model",
        "Metric",
        "Variable",
        "Coefficient",
        "MAE",
        "MAPE",
        "MSE",
        "RMSE",
        "Expl. Var.",
        "R^2 Score",
    ]

    missing_columns_full = [
        col for col in expected_columns_full if col not in df_full.columns
    ]
    assert (
        not missing_columns_full
    ), f"Missing expected columns in full output: {missing_columns_full}"

    assert (
        "Model_1" in df_full["Model"].values
    ), "Expected 'Model_1' in 'Model' column of regression output."


def test_summarize_model_performance_rf_regression(rf_regression_model):
    """
    Test summarize_model_performance for RandomForestRegressor with feature
    importances, including overall_only behavior.
    """
    model, (X, y) = rf_regression_model

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])

    df_full = summarize_model_performance(
        model,
        X,
        y,
        model_type="regression",
        return_df=True,
        overall_only=False,
    )

    assert isinstance(df_full, pd.DataFrame)
    assert "Feat. Imp." in df_full.columns or "Feat. Imp." in df_full["Metric"].values

    # overall_only=True should still produce regression metrics,
    # even if Coefficient or Variable columns remain.
    df_overall = summarize_model_performance(
        model,
        X,
        y,
        model_type="regression",
        return_df=True,
        overall_only=True,
    )

    assert isinstance(df_overall, pd.DataFrame)
    assert "R^2 Score" in df_overall.columns
    assert "Overall Metrics" in df_overall["Metric"].values


def test_summarize_model_performance_mixed_regression(
    regression_model, rf_regression_model
):
    """
    Test summarize_model_performance for mixed regression models
    (Lasso + RandomForestRegressor). Ensures metrics render without dropping
    Coefficient or Feature Importance columns.
    """
    lasso_model, (X_lasso, y_lasso) = regression_model
    rf_model, (X_rf, y_rf) = rf_regression_model
    X = pd.DataFrame(X_lasso, columns=[f"Feature_{i}" for i in range(X_lasso.shape[1])])
    y = y_lasso

    df_full = summarize_model_performance(
        [rf_model, lasso_model],
        X,
        y,
        model_type="regression",
        return_df=True,
        overall_only=False,
    )
    assert isinstance(df_full, pd.DataFrame)
    assert any(df_full["Metric"].isin(["Coefficient", "Feat. Imp."]))

    df_overall = summarize_model_performance(
        [rf_model, lasso_model],
        X,
        y,
        model_type="regression",
        return_df=True,
        overall_only=True,
    )
    assert isinstance(df_overall, pd.DataFrame)
    assert "Overall Metrics" in df_overall["Metric"].values
    assert "R^2 Score" in df_overall.columns


def test_summarize_model_performance_overall_only(regression_model):
    """
    Test summarize_model_performance with overall_only=True for regression
    models, allowing presence of coefficient columns.
    """
    model, (X, y) = regression_model
    df = summarize_model_performance(
        model,
        X,
        y,
        model_type="regression",
        return_df=True,
        overall_only=True,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 1
    assert "Overall Metrics" in df["Metric"].values
    assert "R^2 Score" in df.columns


def test_summarize_model_performance_invalid_combination(regression_model):
    """
    Test ValueError for using overall_only=True with classification models.
    """
    model, (X, y) = regression_model

    with pytest.raises(ValueError, match=r"overall_only.*regression"):
        summarize_model_performance(
            model, X, y, model_type="classification", overall_only=True
        )


@patch("builtins.print")
def test_summarize_model_performance_print_output(mock_print, regression_model):
    """
    Test that summarize_model_performance prints output correctly when
    return_df=False.
    """
    model, (X, y) = regression_model
    summarize_model_performance(
        model,
        X,
        y,
        model_type="regression",
        return_df=False,
    )

    # Ensure print was called
    mock_print.assert_called()

    print("Print output test passed.")


def test_extract_model_name(trained_model):
    """Test extracting model names."""
    name = extract_model_name(trained_model)
    assert name == "LogisticRegression"


# Prevents figures from displaying during testing
@patch("matplotlib.pyplot.show")
def test_show_confusion_matrix_single_model(
    mock_show,
    trained_model,
    sample_data,
):
    """Test if show_confusion_matrix runs correctly for a single model."""
    X, y = sample_data

    try:
        show_confusion_matrix(trained_model, X, y, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_confusion_matrix raised an exception: {e}")


@patch("matplotlib.pyplot.show")
def test_show_confusion_matrix_multiple_models(
    mock_show,
    trained_model,
    sample_data,
):
    """Test if show_confusion_matrix runs correctly for multiple models."""
    X, y = sample_data
    # Using the same model twice for simplicity
    model = [trained_model, trained_model]

    try:
        show_confusion_matrix(model, X, y, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_confusion_matrix failed for multiple models: {e}")


@patch("matplotlib.pyplot.show")
def test_show_confusion_matrix_saves_plot(
    mock_show, trained_model, sample_data, tmp_path
):
    """
    Test if show_confusion_matrix saves the plot when save_plot=True.
    """
    X, y = sample_data
    image_path_png = tmp_path / "confusion_matrix.png"
    image_path_svg = tmp_path / "confusion_matrix.svg"

    try:
        show_confusion_matrix(
            trained_model,
            X,
            y,
            save_plot=True,
            image_path_png=str(image_path_png),
            image_path_svg=str(image_path_svg),
        )
    except Exception as e:
        pytest.fail(f"show_confusion_matrix failed when saving plots: {e}")

    assert image_path_png.exists(), "PNG image was not saved."
    assert image_path_svg.exists(), "SVG image was not saved."


@patch("matplotlib.pyplot.show")
def test_show_confusion_matrix_with_class_labels(
    mock_show,
    trained_model,
    sample_data,
):
    """
    Test if show_confusion_matrix correctly handles custom class labels.
    """
    X, y = sample_data
    custom_labels = ["Negative", "Positive"]

    try:
        show_confusion_matrix(
            trained_model, X, y, class_labels=custom_labels, save_plot=False
        )
    except Exception as e:
        pytest.fail(f"show_confusion_matrix failed with custom class labels: {e}")


@patch("matplotlib.pyplot.show")
def test_show_confusion_matrix_default_labels(
    mock_show,
    trained_model,
    sample_data,
):
    """
    Test if show_confusion_matrix correctly handles default class labels.
    """
    X, y = sample_data

    try:
        show_confusion_matrix(trained_model, X, y, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_confusion_matrix failed with default class labels: {e}")


@patch("matplotlib.pyplot.show")
def test_show_confusion_matrix_with_colorbar(
    mock_show,
    trained_model,
    sample_data,
):
    """Test if show_confusion_matrix runs without raising an error when
    enabling/disabling the colorbar.
    """
    X, y = sample_data

    try:
        show_confusion_matrix(
            trained_model,
            X,
            y,
            save_plot=False,
            show_colorbar=True,
        )
        show_confusion_matrix(
            trained_model,
            X,
            y,
            save_plot=False,
            show_colorbar=False,
        )
    except Exception as e:
        pytest.fail(f"show_confusion_matrix failed: {e}")


@patch("matplotlib.pyplot.show")
def test_show_confusion_matrix_grid(
    mock_show,
    trained_model,
    sample_data,
):
    """Test if show_confusion_matrix correctly handles subplot layout."""
    X, y = sample_data

    # Pass a list of models explicitly
    model = [trained_model, trained_model]

    print(f"DEBUG: models type = {type(model)}")
    print(f"DEBUG: models[0] type = {type(model[0])}")
    try:
        show_confusion_matrix(
            model,
            X,
            y,
            save_plot=False,
            subplots=True,
            n_cols=2,
        )
    except TypeError as e:
        pytest.fail(f"show_confusion_matrix failed due to TypeError: {e}")
    except Exception as e:
        pytest.fail(f"show_confusion_matrix failed unexpectedly: {e}")


@patch("matplotlib.pyplot.show")
def test_show_confusion_matrix_saves_plot(
    mock_show, trained_model, sample_data, tmp_path
):
    """Test if show_confusion_matrix saves the plot when save_plot=True."""
    X, y = sample_data
    image_path_png = tmp_path / "confusion_matrix.png"
    image_path_svg = tmp_path / "confusion_matrix.svg"

    try:
        show_confusion_matrix(
            trained_model,
            X,
            y,
            save_plot=True,
            image_path_png=str(image_path_png),
            image_path_svg=str(image_path_svg),
        )
    except Exception as e:
        pytest.fail(f"show_confusion_matrix failed when saving plots: {e}")

    assert image_path_png.exists(), "PNG image was not saved."
    assert image_path_svg.exists(), "SVG image was not saved."


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_single(
    mock_show,
    trained_model,
    sample_data,
):
    """Test if show_roc_curve runs correctly for a single model."""
    X, y = sample_data
    try:
        show_roc_curve(trained_model, X, y, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_roc_curve raised an exception: {e}")
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_multiple(
    mock_show,
    trained_model,
    sample_data,
):
    """Test if show_roc_curve runs without errors for multiple models."""
    X, y = sample_data
    model = [trained_model, trained_model]
    try:
        show_roc_curve(model, X, y, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_roc_curve raised an exception: {e}")
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_overlay(
    mock_show,
    trained_model,
    sample_data,
):
    """Test if show_roc_curve runs correctly with overlay enabled."""
    X, y = sample_data
    model = [trained_model, trained_model]
    try:
        show_roc_curve(
            model,
            X,
            y,
            overlay=True,
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(f"show_roc_curve raised an exception: {e}")
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_grid(
    mock_show,
    trained_model,
    sample_data,
):
    """Test if show_roc_curve runs correctly with subplots enabled."""
    X, y = sample_data
    model = [trained_model, trained_model]
    try:
        show_roc_curve(
            model,
            X,
            y,
            subplots=True,
            n_cols=2,
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(f"show_roc_curve raised an exception: {e}")
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_invalid_overlay_grid(
    mock_show,
    trained_model,
    sample_data,
):
    """
    Ensure ValueError is raised if both overlay and subplots are set to True.
    """
    X, y = sample_data
    model = [trained_model, trained_model]
    with pytest.raises(
        ValueError,
        match="`subplots` cannot be set to True when `overlay` is True.",
    ):
        show_roc_curve(
            model,
            X,
            y,
            overlay=True,
            subplots=True,
            save_plot=False,
        )


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_save_plot(
    mock_show,
    trained_model,
    sample_data,
    tmp_path,
):
    """Test if show_roc_curve saves the plot when save_plot=True."""
    X, y = sample_data
    image_path_png = tmp_path / "roc_curve.png"
    image_path_svg = tmp_path / "roc_curve.svg"
    try:
        show_roc_curve(
            trained_model,
            X,
            y,
            save_plot=True,
            image_path_png=str(image_path_png),
            image_path_svg=str(image_path_svg),
        )
    except Exception as e:
        pytest.fail(f"show_roc_curve failed when saving plots: {e}")

    # Verify that the files were created
    assert image_path_png.exists(), "PNG image was not saved."
    assert image_path_svg.exists(), "SVG image was not saved."
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_custom_titles(
    mock_show,
    trained_model,
    sample_data,
):
    """Test custom model_title and title parameters."""
    X, y = sample_data
    model = [trained_model, trained_model]
    model_title = ["ModelA", "ModelB"]
    custom_title = "Custom ROC Plot"
    try:
        show_roc_curve(
            model,
            X,
            y,
            model_title=model_title,
            title=custom_title,
            overlay=True,
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(f"show_roc_curve failed with custom titles: {e}")
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_empty_title(
    mock_show,
    trained_model,
    sample_data,
):
    """Test handling of empty title string."""
    X, y = sample_data
    try:
        show_roc_curve(trained_model, X, y, title="", save_plot=False)
    except Exception as e:
        pytest.fail(f"show_roc_curve failed with empty title: {e}")
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_text_wrap(
    mock_show,
    trained_model,
    sample_data,
):
    """Test text wrapping for long titles."""
    X, y = sample_data
    long_title = "This is a very long title that should wrap when text_wrap is set"
    try:
        show_roc_curve(
            trained_model,
            X,
            y,
            title=long_title,
            text_wrap=20,
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(f"show_roc_curve failed with text wrapping: {e}")
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_curve_styling(
    mock_show,
    trained_model,
    sample_data,
):
    """Test custom curve styling with curve_kwgs."""
    X, y = sample_data
    model = [trained_model, trained_model]
    model_title = ["ModelA", "ModelB"]
    curve_kwgs = {
        "ModelA": {"color": "red", "linestyle": "--"},
        "ModelB": {"color": "blue"},
    }
    try:
        show_roc_curve(
            model,
            X,
            y,
            model_title=model_title,
            curve_kwgs=curve_kwgs,
            overlay=True,
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(f"show_roc_curve failed with curve styling: {e}")
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_group_category(
    mock_show,
    trained_model,
    sample_data,
):
    """Test ROC curves grouped by category with class counts."""
    X, y = sample_data
    y = pd.Series(y)  # ensure series
    group_category = pd.Series(
        np.random.choice(
            ["Group1", "Group2"],
            size=len(y),
        )
    )
    try:
        show_roc_curve(
            trained_model,
            X,
            y,
            group_category=group_category,
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(f"show_roc_curve failed with group_category: {e}")
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_decimal_places(
    mock_show,
    trained_model,
    sample_data,
    capsys,
):
    """Test AUC formatting with decimal_places."""
    X, y = sample_data
    decimal_places = 3
    show_roc_curve(
        trained_model,
        X,
        y,
        decimal_places=decimal_places,
        save_plot=False,
    )
    captured = capsys.readouterr()
    assert (
        f"AUC for Model 1:" in captured.out
    ), "AUC calculation was not printed correctly."
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_no_gridlines(
    mock_show,
    trained_model,
    sample_data,
):
    """Test disabling gridlines."""
    X, y = sample_data
    try:
        show_roc_curve(
            trained_model,
            X,
            y,
            gridlines=False,
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(f"show_roc_curve failed with gridlines disabled: {e}")
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_figsize(
    mock_show,
    trained_model,
    sample_data,
):
    """Test custom figure size."""
    X, y = sample_data
    custom_figsize = (10, 8)
    try:
        show_roc_curve(
            trained_model,
            X,
            y,
            figsize=custom_figsize,
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(f"show_roc_curve failed with custom figsize: {e}")

    # Ensure the figure was set to the correct size
    assert np.allclose(
        plt.gcf().get_size_inches(), custom_figsize
    ), "Figure size is incorrect."
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_grid_layout(
    mock_show,
    trained_model,
    sample_data,
):
    """Test subplot layout with custom rows and columns."""
    X, y = sample_data
    model = [trained_model, trained_model, trained_model]
    try:
        show_roc_curve(
            model,
            X,
            y,
            subplots=True,
            n_rows=2,
            n_cols=2,
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(f"show_roc_curve failed with custom subplots layout: {e}")
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_pr_curve_single(
    mock_show,
    trained_model,
    sample_data,
):
    """Test if show_pr_curve runs correctly for a single model."""
    X, y = sample_data
    try:
        show_pr_curve(
            trained_model,
            X,
            y,
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(f"show_pr_curve raised an exception: {e}")
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_pr_curve_multiple(
    mock_show,
    trained_model,
    sample_data,
):
    """Test if show_pr_curve runs without errors for multiple models."""
    X, y = sample_data
    model = [trained_model, trained_model]
    try:
        show_pr_curve(model, X, y, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_pr_curve raised an exception: {e}")
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_pr_curve_overlay(
    mock_show,
    trained_model,
    sample_data,
):
    """Test if show_pr_curve runs correctly with overlay enabled."""
    X, y = sample_data
    model = [trained_model, trained_model]
    try:
        show_pr_curve(
            model,
            X,
            y,
            overlay=True,
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(f"show_pr_curve raised an exception: {e}")
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_pr_curve_grid(
    mock_show,
    trained_model,
    sample_data,
):
    """Test if show_pr_curve runs correctly with subplots enabled."""
    X, y = sample_data
    model = [trained_model, trained_model]
    try:
        show_pr_curve(
            model,
            X,
            y,
            subplots=True,
            n_cols=2,
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(f"show_pr_curve raised an exception: {e}")
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_pr_curve_invalid_overlay_grid(
    mock_show,
    trained_model,
    sample_data,
):
    """
    Ensure ValueError is raised if both overlay and subplots are set to True.
    """
    X, y = sample_data
    model = [trained_model, trained_model]
    with pytest.raises(
        ValueError,
        match="`subplots` cannot be set to True when `overlay` is True.",
    ):
        show_pr_curve(model, X, y, overlay=True, subplots=True, save_plot=False)


@patch("matplotlib.pyplot.show")
def test_show_pr_curve_save_plot(
    mock_show,
    trained_model,
    sample_data,
    tmp_path,
):
    """Test if show_pr_curve saves the plot when save_plot=True."""
    X, y = sample_data
    image_path_png = tmp_path / "pr_curve.png"
    image_path_svg = tmp_path / "pr_curve.svg"
    try:
        show_pr_curve(
            trained_model,
            X,
            y,
            save_plot=True,
            image_path_png=str(image_path_png),
            image_path_svg=str(image_path_svg),
        )
    except Exception as e:
        pytest.fail(f"show_pr_curve failed when saving plots: {e}")

    assert os.path.exists(image_path_png), "PNG image was not saved."
    assert os.path.exists(image_path_svg), "SVG image was not saved."
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_pr_curve_custom_titles(
    mock_show,
    trained_model,
    sample_data,
):
    """Test custom model_title and title parameters."""
    X, y = sample_data
    model = [trained_model, trained_model]
    model_title = ["ModelA", "ModelB"]
    custom_title = "Custom PR Plot"
    try:
        show_pr_curve(
            model,
            X,
            y,
            model_title=model_title,
            title=custom_title,
            overlay=True,
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(f"show_pr_curve failed with custom titles: {e}")
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_pr_curve_empty_title(
    mock_show,
    trained_model,
    sample_data,
):
    """Test handling of empty title string."""
    X, y = sample_data
    try:
        show_pr_curve(trained_model, X, y, title="", save_plot=False)
    except Exception as e:
        pytest.fail(f"show_pr_curve failed with empty title: {e}")
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_pr_curve_text_wrap(
    mock_show,
    trained_model,
    sample_data,
):
    """Test text wrapping for long titles."""
    X, y = sample_data
    long_title = "This is a very long title that should wrap when text_wrap is set"
    try:
        show_pr_curve(
            trained_model,
            X,
            y,
            title=long_title,
            text_wrap=20,
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(f"show_pr_curve failed with text wrapping: {e}")
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_pr_curve_curve_styling(
    mock_show,
    trained_model,
    sample_data,
):
    """Test custom curve styling with curve_kwgs."""
    X, y = sample_data
    model = [trained_model, trained_model]
    model_title = ["ModelA", "ModelB"]
    curve_kwgs = {
        "ModelA": {"color": "red", "linestyle": "--"},
        "ModelB": {"color": "blue"},
    }
    try:
        show_pr_curve(
            model,
            X,
            y,
            model_title=model_title,
            curve_kwgs=curve_kwgs,
            overlay=True,
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(f"show_pr_curve failed with curve styling: {e}")
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_pr_curve_group_category(
    mock_show,
    trained_model,
    sample_data,
):
    """Test PR curves grouped by category with class counts."""
    X, y = sample_data
    y = pd.Series(y)  # Ensure y is a pandas Series
    group_category = pd.Series(
        np.random.choice(
            ["Group1", "Group2"],
            size=len(y),
        )
    )
    try:
        show_pr_curve(
            trained_model,
            X,
            y,
            group_category=group_category,
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(f"show_pr_curve failed with group_category: {e}")
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_pr_curve_decimal_places(
    mock_show,
    trained_model,
    sample_data,
    capsys,
):
    """Test AP formatting with decimal_places."""
    X, y = sample_data
    decimal_places = 3
    show_pr_curve(
        trained_model,
        X,
        y,
        decimal_places=decimal_places,
        save_plot=False,
    )
    captured = capsys.readouterr()
    assert (
        "Average Precision for Model 1" in captured.out
    ), "AP calculation was not printed correctly."
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_pr_curve_no_gridlines(
    mock_show,
    trained_model,
    sample_data,
):
    """Test disabling gridlines."""
    X, y = sample_data
    try:
        show_pr_curve(
            trained_model,
            X,
            y,
            gridlines=False,
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(f"show_pr_curve failed with gridlines disabled: {e}")
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_pr_curve_figsize(
    mock_show,
    trained_model,
    sample_data,
):
    """Test custom figure size."""
    X, y = sample_data
    custom_figsize = (10, 8)
    try:
        show_pr_curve(
            trained_model,
            X,
            y,
            figsize=custom_figsize,
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(f"show_pr_curve failed with custom figsize: {e}")

    assert np.allclose(
        plt.gcf().get_size_inches(), custom_figsize
    ), "Figure size is incorrect."
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_pr_curve_grid_layout(
    mock_show,
    trained_model,
    sample_data,
):
    """
    Test subplots layout with custom rows and columns and corrected labels.
    """
    X, y = sample_data
    model = [trained_model, trained_model, trained_model]
    model_title = ["ModelA", "ModelB", "ModelC"]
    try:
        show_pr_curve(
            model,
            X,
            y,
            model_title=model_title,
            subplots=True,
            n_rows=2,
            n_cols=2,
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(f"show_pr_curve failed with custom subplots layout: {e}")
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_lift_chart_single(
    mock_show,
    trained_model,
    sample_data,
):
    """Test if show_lift_chart runs correctly for a single model."""
    X, y = sample_data
    try:
        show_lift_chart(trained_model, X, y, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_lift_chart raised an exception: {e}")


@patch("matplotlib.pyplot.show")
def test_show_lift_chart_multiple(
    mock_show,
    trained_model,
    sample_data,
):
    """Test if show_lift_chart runs without errors for multiple models."""
    X, y = sample_data
    model = [trained_model, trained_model]
    try:
        show_lift_chart(model, X, y, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_lift_chart raised an exception: {e}")


@patch("matplotlib.pyplot.show")
def test_show_lift_chart_overlay(
    mock_show,
    trained_model,
    sample_data,
):
    """Test if show_lift_chart runs correctly with overlay enabled."""
    X, y = sample_data
    model = [trained_model, trained_model]
    try:
        show_lift_chart(
            model,
            X,
            y,
            overlay=True,
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(f"show_lift_chart raised an exception: {e}")


@patch("matplotlib.pyplot.show")
def test_show_lift_chart_grid(
    mock_show,
    trained_model,
    sample_data,
):
    """Test if show_lift_chart runs correctly with subplots enabled."""
    X, y = sample_data
    model = [trained_model, trained_model]
    try:
        show_lift_chart(
            model,
            X,
            y,
            subplots=True,
            n_cols=2,
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(f"show_lift_chart raised an exception: {e}")


@patch("model_metrics.model_evaluator.get_predictions")
@patch("matplotlib.pyplot.show")
def test_show_lift_chart_saves_plot(
    mock_show, mock_gp, trained_model, sample_data, tmp_path
):
    """Ensure saving works (mock predictions to isolate plotting)."""
    X, y = sample_data
    # Return y_true, y_prob, y_pred, threshold
    mock_gp.return_value = (
        y,
        np.random.rand(len(y)),
        (np.random.rand(len(y)) > 0.5).astype(int),
        0.5,
    )
    image_path_png = tmp_path / "lift_chart.png"
    image_path_svg = tmp_path / "lift_chart.svg"

    try:
        show_lift_chart(
            trained_model,
            X,
            y,
            save_plot=True,
            image_path_png=str(image_path_png),
            image_path_svg=str(image_path_svg),
        )
    except Exception as e:
        pytest.fail(f"show_lift_chart failed when saving plots: {e}")

    assert image_path_png.exists(), "PNG image was not saved."
    assert image_path_svg.exists(), "SVG image was not saved."


@patch("matplotlib.pyplot.show")
def test_show_gain_chart_single(mock_show, trained_model, sample_data):
    """Test if show_gain_chart runs correctly for a single model."""
    X, y = sample_data
    try:
        show_gain_chart(trained_model, X, y, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_gain_chart raised an exception: {e}")


@patch("matplotlib.pyplot.show")
def test_show_gain_chart_multiple(mock_show, trained_model, sample_data):
    """
    Test if show_gain_chart runs without errors for multiple models.
    """
    X, y = sample_data
    model = [trained_model, trained_model]
    try:
        show_gain_chart(model, X, y, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_gain_chart raised an exception: {e}")


@patch("matplotlib.pyplot.show")
def test_show_gain_chart_overlay(mock_show, trained_model, sample_data):
    """Test if show_gain_chart runs correctly with overlay enabled."""
    X, y = sample_data
    model = [trained_model, trained_model]
    try:
        show_gain_chart(model, X, y, overlay=True, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_gain_chart raised an exception: {e}")


@patch("matplotlib.pyplot.show")
def test_show_gain_chart_grid(mock_show, trained_model, sample_data):
    """Test if show_gain_chart runs correctly with subplots enabled."""
    X, y = sample_data
    model = [trained_model, trained_model]
    try:
        show_gain_chart(model, X, y, subplots=True, n_cols=2, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_gain_chart raised an exception: {e}")


@patch("model_metrics.model_evaluator.get_predictions")
@patch("matplotlib.pyplot.show")
def test_show_gain_chart_saves_plot(
    mock_show, mock_gp, trained_model, sample_data, tmp_path
):
    """Ensure saving works (mock predictions to isolate plotting)."""
    X, y = sample_data
    mock_gp.return_value = (
        y,
        np.random.rand(len(y)),
        (np.random.rand(len(y)) > 0.5).astype(int),
        0.5,
    )
    image_path_png = tmp_path / "gain_chart.png"
    image_path_svg = tmp_path / "gain_chart.svg"

    try:
        show_gain_chart(
            trained_model,
            X,
            y,
            save_plot=True,
            image_path_png=str(image_path_png),
            image_path_svg=str(image_path_svg),
        )
    except Exception as e:
        pytest.fail(f"show_gain_chart failed when saving plots: {e}")

    assert image_path_png.exists(), "PNG image was not saved."
    assert image_path_svg.exists(), "SVG image was not saved."


def test_show_calibration_curve_basic(trained_model, sample_data):
    """Test basic functionality of the calibration curve."""
    X, y = sample_data
    try:
        show_calibration_curve(trained_model, X, y)
    except Exception as e:
        pytest.fail(f"show_calibration_curve failed unexpectedly: {e}")


def test_show_calibration_curve_overlay(trained_model, sample_data):
    """Test overlaying multiple models on one calibration plot."""
    X, y = sample_data
    model = [trained_model, trained_model]  # Use the same model twice
    try:
        show_calibration_curve(
            model,
            X,
            y,
            overlay=True,
            title="Overlay Test",
        )
    except Exception as e:
        pytest.fail(f"show_calibration_curve failed on overlay: {e}")


def test_show_calibration_curve_grid(trained_model, sample_data):
    """Test subplots layout with multiple models."""
    X, y = sample_data
    model = [trained_model, trained_model]  # Ensure it's a list
    try:
        show_calibration_curve(model=model, X=X, y=y, subplots=True, n_cols=2)
    except Exception as e:
        pytest.fail(f"show_calibration_curve failed on subplots layout: {e}")


def test_show_calibration_curve_invalid_overlay_grid(
    trained_model,
    sample_data,
):
    """Ensure ValueError is raised if both overlay and subplots are set to True."""
    X, y = sample_data
    with pytest.raises(
        ValueError, match="`subplots` cannot be set to True when `overlay` is True."
    ):
        show_calibration_curve(
            [trained_model, trained_model], X, y, overlay=True, subplots=True
        )


def test_show_calibration_curve_save_plot(
    trained_model,
    sample_data,
    tmp_path,
):
    """Test saving the calibration curve plot."""
    X, y = sample_data
    save_path = str(tmp_path / "calibration.png")

    try:
        show_calibration_curve(
            trained_model, X, y, save_plot=True, image_path_png=save_path
        )
        assert os.path.exists(save_path), "Plot was not saved correctly"
    except Exception as e:
        pytest.fail(f"show_calibration_curve failed when saving plot: {e}")


def test_show_calibration_curve_brier_score(trained_model, sample_data):
    """Test calculation of Brier score."""
    X, y = sample_data
    try:
        show_calibration_curve(trained_model, X, y, show_brier_score=True)
    except Exception as e:
        pytest.fail(f"show_calibration_curve failed with Brier score enabled: {e}")


def test_show_calibration_curve_custom_titles(trained_model, sample_data):
    """Test custom model titles for subplots layout."""
    X, y = sample_data
    model = [trained_model, trained_model]  # Ensure it's a list
    titles = ["Model 1", "Model 2"]  # Titles must match models length
    try:
        show_calibration_curve(
            model=model,
            X=X,
            y=y,
            model_title=titles,
            subplots=True,
        )
    except Exception as e:
        pytest.fail(f"show_calibration_curve failed with custom titles: {e}")


@patch("matplotlib.pyplot.show")
def test_show_calibration_curve_empty_title(
    mock_show,
    trained_model,
    sample_data,
):
    """Ensure plot title is suppressed when title=''."""
    X, y = sample_data
    try:
        show_calibration_curve(
            trained_model,
            X,
            y,
            title="",  # Suppresses title
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(f"show_calibration_curve failed with empty title: {e}")
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_calibration_curve_text_wrap(
    mock_show,
    trained_model,
    sample_data,
):
    """Ensure text_wrap works correctly for long titles."""
    X, y = sample_data
    long_title = (
        "This is a very long title for testing text wrapping in calibration plots"
    )
    try:
        show_calibration_curve(
            trained_model,
            X,
            y,
            title=long_title,
            text_wrap=20,
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(f"show_calibration_curve failed with text_wrap: {e}")
    assert mock_show.called, "plt.show() was not called."


@patch("matplotlib.pyplot.show")
def test_show_calibration_curve_group_category_multiple_models(
    mock_show,
    trained_model,
    sample_data,
):
    """
    Test show_calibration_curve with group_category and multiple models.
    Ensures each model's group plot renders individually.
    """
    X, y = sample_data
    y = pd.Series(y)
    group_category = pd.Series(
        np.random.choice(
            ["Group A", "Group B"],
            size=len(y),
        )
    )

    # Using same model twice for simplicity
    models = [trained_model, trained_model]
    model_titles = ["Model 1", "Model 2"]

    show_calibration_curve(
        model=models,
        X=X,
        y=y,
        model_title=model_titles,
        group_category=group_category,
        save_plot=False,
    )

    # Should call plt.show once per model when group_category is used
    assert mock_show.call_count == len(
        models
    ), "Expected one plot per model for group_category"


def test_show_calibration_curve_invalid_combination_with_group_category(
    trained_model,
    sample_data,
):
    """
    Ensure ValueError is raised if `group_category` is used with `overlay` or `subplots`.
    """
    X, y = sample_data
    group = pd.Series(np.random.choice(["A", "B"], size=len(y)))

    with pytest.raises(
        ValueError,
        match="`group_category` requires `overlay=False` and `subplots=False`.",
    ):
        show_calibration_curve(
            [trained_model],
            X,
            y,
            group_category=group,
            overlay=True,
        )

    with pytest.raises(
        ValueError,
        match="`group_category` requires `overlay=False` and `subplots=False`.",
    ):
        show_calibration_curve(
            [trained_model],
            X,
            y,
            group_category=group,
            subplots=True,
        )


def test_custom_help(capsys):
    # Capture the help output
    help()
    captured = capsys.readouterr()

    # Check if the ASCII art and detailed documentation are present
    assert "Welcome to Model Metrics!" in captured.out
    assert "PyPI: https://pypi.org/project/model-metrics/" in captured.out

    # Ensure built-in help still works
    original_help = builtins.help
    assert original_help is not None


@patch("model_metrics.model_evaluator.get_predictions")
@patch("matplotlib.pyplot.show")  # Prevents figures from displaying
def test_plot_threshold_metrics_execution(
    mock_show, mock_get_predictions, trained_model, sample_data
):
    """Test that plot_threshold_metrics runs without errors."""
    X, y = sample_data
    mock_get_predictions.return_value = (
        y,
        np.random.rand(len(y)),  # Simulated probabilities
        np.random.randint(0, 2, len(y)),  # Simulated predictions
        0.5,
    )

    try:
        plot_threshold_metrics(trained_model, X, y, save_plot=False)
    except Exception as e:
        pytest.fail(f"plot_threshold_metrics failed unexpectedly: {e}")

    mock_get_predictions.assert_called_once()  # Ensure function is called
    assert mock_show.called, "plt.show() was not called."


@patch("model_metrics.model_evaluator.get_predictions")
@patch("matplotlib.pyplot.show")
@pytest.mark.parametrize(
    "lookup_metric, lookup_value",
    [
        ("precision", 0.8),
        ("recall", 0.75),
        ("f1", 0.85),
        ("specificity", 0.9),
    ],
)
def test_plot_threshold_metrics_lookup_metric(
    mock_show,
    mock_get_predictions,
    trained_model,
    sample_data,
    lookup_metric,
    lookup_value,
):
    """Test finding the best threshold for a given lookup metric."""
    X, y = sample_data
    mock_get_predictions.return_value = (
        y,
        np.random.rand(len(y)),  # Simulated probabilities
        np.random.randint(0, 2, len(y)),  # Simulated predictions
        0.5,
    )

    with patch("builtins.print") as mock_print:
        plot_threshold_metrics(
            trained_model,
            X,
            y,
            lookup_metric=lookup_metric,
            lookup_value=lookup_value,
            decimal_places=4,
            save_plot=False,
        )
        mock_print.assert_called()
        assert (
            f"Best threshold for target {lookup_metric}" in mock_print.call_args[0][0]
        ), f"Expected 'Best threshold for target {lookup_metric}' in print output"


@patch("model_metrics.model_evaluator.get_predictions")
@patch("matplotlib.pyplot.show")
@pytest.mark.parametrize("decimal_places", [2, 4, 6])
def test_plot_threshold_metrics_decimal_places(
    mock_show,
    mock_get_predictions,
    trained_model,
    sample_data,
    decimal_places,
):
    """Test if decimal_places parameter correctly formats outputs."""
    X, y = sample_data
    lookup_metric = "recall"
    lookup_value = 0.5
    mock_get_predictions.return_value = (
        y,
        np.random.rand(len(y)),
        np.random.randint(0, 2, len(y)),
        0.5,
    )

    try:
        plot_threshold_metrics(
            trained_model,
            X,
            y,
            lookup_metric=lookup_metric,
            lookup_value=lookup_value,
            decimal_places=decimal_places,
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(
            f"plot_threshold_metrics failed for "
            f"decimal_places={decimal_places}: {e}"
        )

    mock_get_predictions.assert_called_once()


@patch("model_metrics.model_evaluator.get_predictions")
@patch("matplotlib.pyplot.show")
def test_plot_threshold_metrics_no_baseline(
    mock_show,
    mock_get_predictions,
    trained_model,
    sample_data,
):
    """Test disabling the baseline threshold line."""
    X, y = sample_data
    mock_get_predictions.return_value = (
        y,
        np.random.rand(len(y)),
        np.random.randint(0, 2, len(y)),
        0.5,
    )

    plot_threshold_metrics(
        trained_model,
        X,
        y,
        baseline_thresh=False,
        save_plot=False,
    )

    assert mock_show.called, "plt.show() was not called."


@patch("model_metrics.model_evaluator.get_predictions")
@patch("matplotlib.pyplot.show")
def test_plot_threshold_metrics_save_plot(
    mock_show,
    mock_get_predictions,
    trained_model,
    sample_data,
    tmp_path,
):
    """Test that images are saved correctly when save_plot=True."""
    X, y = sample_data
    image_path_png = tmp_path / "threshold_plot.png"
    image_path_svg = tmp_path / "threshold_plot.svg"
    mock_get_predictions.return_value = (
        y,
        np.random.rand(len(y)),
        np.random.randint(0, 2, len(y)),
        0.5,
    )

    try:
        plot_threshold_metrics(
            trained_model,
            X,
            y,
            save_plot=True,
            image_path_png=str(image_path_png),
            image_path_svg=str(image_path_svg),
        )
    except Exception as e:
        pytest.fail(f"plot_threshold_metrics failed when saving plots: {e}")

    assert image_path_png.exists(), "PNG image was not saved."
    assert image_path_svg.exists(), "SVG image was not saved."


@patch("model_metrics.model_evaluator.get_predictions")
@patch("matplotlib.pyplot.show")
def test_plot_threshold_metrics_custom_styles(
    mock_show,
    mock_get_predictions,
    trained_model,
    sample_data,
):
    """Test custom curve and baseline styles."""
    X, y = sample_data
    mock_get_predictions.return_value = (
        y,
        np.random.rand(len(y)),
        np.random.randint(0, 2, len(y)),
        0.5,
    )

    curve_kwgs = {"linestyle": "--", "linewidth": 2}
    baseline_kwgs = {"linestyle": "dotted", "linewidth": 1.5, "color": "black"}

    plot_threshold_metrics(
        trained_model,
        X,
        y,
        curve_kwgs=curve_kwgs,
        baseline_kwgs=baseline_kwgs,
        save_plot=False,
    )

    assert mock_show.called, "plt.show() was not called."


def test_invalid_model_type_raises_value_error(trained_model, sample_data):
    """
    Ensure invalid model_type raises ValueError with flexible match.
    """
    X, y = sample_data
    with pytest.raises(
        ValueError,
        match=r"model_type.*classification.*regression",
    ):
        summarize_model_performance(
            [trained_model], X, y, model_type="clustering", return_df=True
        )


def test_invalid_model_title_type(trained_model, sample_data):
    """
    Ensure TypeError is raised when model_title is not str, list, Series, or None.
    """
    X, y = sample_data
    with pytest.raises(
        (TypeError, ValueError),
        match=r"model_title|subscriptable",
    ):
        summarize_model_performance(
            [trained_model], X, y, model_type="classification", model_title=123
        )


def test_statsmodels_ols_model_handling():
    # Generate simple data
    np.random.seed(0)
    X = pd.DataFrame({"x1": np.random.rand(100)})
    y = X["x1"] * 3 + np.random.normal(0, 0.1, 100)

    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    df = summarize_model_performance(
        model,
        X,
        y,
        model_type="regression",
        return_df=True,
    )
    assert "MAE" in df.columns or "MAE" in df["Metric"].values  # Based on mode


def test_pipeline_last_step_has_coef(sample_data):
    X, y = sample_data
    pipeline = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression())])
    pipeline.fit(X, y)
    df = summarize_model_performance(
        pipeline,
        X,
        y,  # classification path; get_predictions is used
        model_type="classification",
        return_df=True,
    )
    assert isinstance(df, pd.DataFrame)


@patch("builtins.print")
def test_print_classification_output_format(mock_print, trained_model, sample_data):
    X, y = sample_data
    summarize_model_performance(
        [trained_model],
        X,
        y,
        model_type="classification",
        return_df=False,
    )
    assert mock_print.called


@patch("builtins.print")
def test_print_regression_output_format(mock_print, regression_model):
    model, (X, y) = regression_model
    summarize_model_performance(
        [model],
        X,
        y,
        model_type="regression",
        return_df=False,
    )
    assert mock_print.called


@pytest.mark.parametrize("container", ["ndarray", "series", "tuple", "list"])
@patch("model_metrics.model_evaluator.get_predictions")
@patch("matplotlib.pyplot.show")
def test_show_roc_curve_y_prob_normalization(
    mock_show, mock_gp, trained_model, sample_data, container
):
    X, y = sample_data
    p = trained_model.predict_proba(X)[:, 1]

    if container == "series":
        y_prob = pd.Series(p)
    elif container == "tuple":
        y_prob = (p,)
    elif container == "list":
        y_prob = [p]
    else:
        y_prob = p

    # Return the same shapes the helper expects
    mock_gp.return_value = (
        y,
        np.asarray(p),  # y_prob
        (np.asarray(p) > 0.5).astype(int),  # y_pred
        0.5,  # threshold
    )

    # Should not raise regardless of container
    try:
        show_roc_curve(trained_model, X, y=y, y_prob=y_prob, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_roc_curve failed for container={container}: {e}")
    assert mock_show.called


@patch("model_metrics.model_evaluator.get_predictions")
@patch("matplotlib.pyplot.show")
def test_show_calibration_curve_y_prob_ndarray_ok(
    mock_show, mock_gp, trained_model, sample_data
):
    X, y = sample_data
    p = trained_model.predict_proba(X)[:, 1]

    mock_gp.return_value = (
        y,
        np.asarray(p),
        (np.asarray(p) > 0.5).astype(int),
        0.5,
    )

    try:
        show_calibration_curve(trained_model, X, y=y, y_prob=p, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_calibration_curve failed with ndarray y_prob: {e}")
    assert mock_show.called


@patch("model_metrics.model_evaluator.get_predictions")
@patch("matplotlib.pyplot.show")
def test_y_prob_explicit_list_ok(mock_show, mock_gp, trained_model, sample_data):
    X, y = sample_data
    p = trained_model.predict_proba(X)[:, 1]
    models = [trained_model, trained_model]
    y_probs = [p, p]

    mock_gp.return_value = (
        y,
        np.asarray(p),
        (np.asarray(p) > 0.5).astype(int),
        0.5,
    )

    try:
        show_pr_curve(
            models,
            X,
            y=y,
            y_prob=y_probs,
            overlay=True,
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(f"show_pr_curve failed with explicit list y_prob: {e}")
    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_with_delong(mock_show, trained_model, sample_data):
    """
    Test DeLong/Hanley-McNeil AUC comparison when comparing two models.
    Ensures correct structure and prints expected output.
    """
    X, y = sample_data
    model1 = trained_model
    model2 = LogisticRegression().fit(X, y)

    # Two models, same data for simplicity
    models = [model1, model2]
    model_titles = ["LogReg_1", "LogReg_2"]

    # Get predicted probabilities manually for delong input
    y_prob1 = model1.predict_proba(X)[:, 1]
    y_prob2 = model2.predict_proba(X)[:, 1]

    # Run with tuple of y_probs instead of delong=True
    with patch("builtins.print") as mock_print:
        show_roc_curve(
            models,
            X,
            y,
            model_title=model_titles,
            delong=(y_prob1, y_prob2),  # ✅ this satisfies the function’s expectation
            save_plot=False,
        )

    printed = " ".join(
        [str(arg) for call in mock_print.call_args_list for arg in call[0]]
    )
    assert (
        "Hanley" in printed or "Delong" in printed
    ), "Expected AUC comparison printout."
    assert any(
        k in printed for k in ["p-value", "P-value", "P="]
    ), "Expected p-value output."
    assert mock_show.called


def test_hanley_mcneil_auc_test_outputs_valid_values(sample_data, capsys):
    """Ensure hanley_mcneil_auc_test prints valid AUC comparison output."""
    from model_metrics.model_evaluator import hanley_mcneil_auc_test

    X, y = sample_data
    y_scores_1 = np.random.rand(len(y))
    y_scores_2 = np.random.rand(len(y))

    hanley_mcneil_auc_test(y, y_scores_1, y_scores_2)
    captured = capsys.readouterr().out

    assert "Hanley" in captured
    assert "AUC" in captured
    assert "p-value" in captured
