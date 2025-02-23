import pytest
import builtins
from unittest.mock import patch
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from model_metrics.model_evaluator import (
    save_plot_images,
    get_predictions,
    summarize_model_performance,
    show_confusion_matrix,
    show_roc_curve,
    show_pr_curve,
    show_calibration_curve,
    get_model_probabilities,
    extract_model_titles,
    extract_model_name,
    show_lift_chart,
    show_gain_chart,
    show_ks_curve,
    plot_threshold_metrics,
)

matplotlib.use("Agg")  # Set non-interactive backend


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
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
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
        X, y, test_size=0.2, random_state=42
    )
    model = Lasso(alpha=0.1).fit(X_train, y_train)
    return model, (X_test, y_test)


def test_save_plot_images(tmp_path):
    """Test that images are saved correctly when save_plot=True."""
    filename = "test_plot"
    image_path_png = tmp_path / "png"
    image_path_svg = tmp_path / "svg"

    os.makedirs(image_path_png, exist_ok=True)
    os.makedirs(image_path_svg, exist_ok=True)

    plt.plot([0, 1], [0, 1])  # Dummy plot
    save_plot_images(filename, True, str(image_path_png), str(image_path_svg))

    assert os.path.exists(os.path.join(image_path_png, f"{filename}.png"))
    assert os.path.exists(os.path.join(image_path_svg, f"{filename}.svg"))


def test_get_predictions(trained_model, sample_data):
    """Test get_predictions function."""
    X, y = sample_data
    y_true, y_prob, y_pred, threshold = get_predictions(
        trained_model, X, y, None, None, None
    )

    assert len(y_true) == len(y_prob) == len(y_pred) == len(y)
    assert 0 <= threshold <= 1


def test_get_predictions_single_model_proba(trained_model, sample_data):
    """Test get_predictions using a single model with predict_proba."""
    X, y = sample_data
    y_true, y_prob, y_pred, threshold = get_predictions(
        trained_model, X, y, None, None, None
    )

    assert len(y_true) == len(y_prob) == len(y_pred)
    assert threshold == 0.5  # Default threshold


def test_get_predictions_single_model_no_proba(sample_data):
    """Test get_predictions with a model lacking predict_proba."""

    class CustomModel:
        def fit(self, X, y):
            pass

        def predict(self, X):
            return np.random.randint(0, 2, size=len(X))

    model = CustomModel()
    X, y = sample_data
    y_true, y_prob, y_pred, threshold = get_predictions(
        model,
        X,
        y,
        None,
        None,
        None,
    )

    assert len(y_true) == len(y_prob) == len(y_pred)


def test_get_predictions_with_custom_threshold(trained_model, sample_data):
    """Test get_predictions with a custom threshold."""
    X, y = sample_data
    custom_threshold = 0.7
    _, _, y_pred, threshold = get_predictions(
        trained_model, X, y, None, custom_threshold, None
    )

    assert threshold == custom_threshold


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

    y_true, y_prob, y_pred, threshold = get_predictions(
        model,
        X,
        y,
        None,
        None,
        None,
    )

    assert len(y_true) > 0
    assert len(y_prob) > 0
    assert len(y_pred) > 0
    assert isinstance(threshold, float)


def test_summarize_model_performance(trained_model, sample_data, capsys):
    """Test summarize_model_performance function with formatted output."""
    X, y = sample_data
    df = summarize_model_performance([trained_model], X, y, return_df=True)

    # Debugging: Print the actual DataFrame before assertions
    print("\nDEBUG: Model Performance DataFrame")
    print(df)

    print("DEBUG: DataFrame Index Values ->", df.index.tolist())
    print("DEBUG: DataFrame Column Names ->", df.columns.tolist())

    # Ensure output is a DataFrame
    assert isinstance(df, pd.DataFrame), "Output should be a pandas DataFrame."

    # FIXED: Check inside the "Metrics" column instead of column names
    expected_metrics = ["Precision/PPV", "AUC ROC"]
    missing_metrics = [
        metric for metric in expected_metrics if metric not in df["Metrics"].values
    ]

    # Debugging: Print missing metrics if assertion fails
    if missing_metrics:
        print(f"DEBUG: Missing metrics -> {missing_metrics}")

    assert not missing_metrics, f"Missing expected metrics: {missing_metrics}"

    # Capture printed output
    summarize_model_performance([trained_model], X, y, return_df=False)
    captured = capsys.readouterr().out

    # Debugging: Print captured output to check format
    print("\nDEBUG: Captured Output")
    print(captured)

    # Find actual header line by locating the first separator line ("----")
    captured_lines = captured.split("\n")
    separator_idx = next(
        (i for i, line in enumerate(captured_lines) if set(line) == {"-"}), None
    )

    if separator_idx is None or separator_idx + 1 >= len(captured_lines):
        raise AssertionError("Could not locate separator line in captured output.")

    header_line = captured_lines[
        separator_idx + 1
    ]  # The line immediately after separator

    # Validate column headers formatting
    expected_header_keywords = ["Model", "Precision/PPV", "AUC ROC"]
    for keyword in expected_header_keywords:
        assert (
            keyword in header_line
        ), f"Header misalignment detected. Expected '{keyword}', but found: {header_line}"

    print("All tests passed for formatted output.")


def test_summarize_model_performance_classification(classification_model, capsys):
    """Test summarize_model_performance function for classification models."""
    trained_model, sample_data = classification_model
    X, y = sample_data

    df = summarize_model_performance(
        [trained_model], X, y, model_type="classification", return_df=True
    )

    # Print actual DataFrame for debugging if test fails
    print("\nDEBUG: Classification Model Performance DataFrame")
    print(df)

    # Print actual column names to check expected format
    print("Columns:", df.columns)

    # Ensure output is a DataFrame
    assert isinstance(df, pd.DataFrame), "Output should be a pandas DataFrame."

    # Ensure key classification metrics exist in the DataFrame
    expected_metrics = ["Precision/PPV", "AUC ROC", "F1-Score", "Sensitivity/Recall"]
    missing_metrics = [
        metric for metric in expected_metrics if metric not in df["Metrics"].values
    ]

    # Print missing metrics if assertion fails
    assert (
        not missing_metrics
    ), f"Missing expected classification metrics: {missing_metrics}"

    # Ensure model name is included in columns
    model_name = extract_model_name(trained_model)
    assert model_name in df.columns, f"Expected model name '{model_name}' in columns."

    # Ensure numeric values are not empty
    for metric in expected_metrics:
        assert (
            df[model_name].loc[df["Metrics"] == metric].notna().all()
        ), f"Missing values for {metric}"

    print("All classification metrics present and correctly formatted.")


def test_summarize_model_performance_regression(regression_model):
    """Test summarize_model_performance function for regression models."""
    model, (X, y) = regression_model
    df = summarize_model_performance(
        model, X, y, model_type="regression", return_df=True
    )

    # Ensure output is a DataFrame
    assert isinstance(df, pd.DataFrame), "Output should be a pandas DataFrame."

    # Expected columns in full regression output
    expected_columns = [
        "Model",
        "Metric",
        "Variable",
        "Coefficient",
        "P-value",
        "MAE",
        "MAPE (%)",
        "MSE",
        "RMSE",
        "Expl. Var.",
        "R^2 Score",
    ]

    missing_columns = [col for col in expected_columns if col not in df.columns]
    assert not missing_columns, f"Missing expected columns: {missing_columns}"

    # Ensure the model name is captured
    model_name = extract_model_name(model)
    assert (
        model_name in df["Model"].values
    ), f"Model name '{model_name}' is missing from DataFrame."

    print("Regression performance summary test passed.")


def test_summarize_model_performance_overall_only(regression_model):
    """
    Test summarize_model_performance with overall_only=True for regression models.
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

    # Ensure DataFrame is returned
    assert isinstance(df, pd.DataFrame), "Expected a DataFrame output."

    # Ensure "Overall Metrics" row is the only row
    assert len(df) == 1, "Expected only one row for 'Overall Metrics'."
    assert (
        df.iloc[0]["Metric"] == "Overall Metrics"
    ), "First row should be 'Overall Metrics'."

    # Ensure unnecessary columns are removed
    assert (
        "Variable" not in df.columns
    ), "Column 'Variable' should be removed in 'overall_only' mode."
    assert (
        "Coefficient" not in df.columns
    ), "Column 'Coefficient' should be removed in 'overall_only' mode."
    assert (
        "P-value" not in df.columns
    ), "Column 'P-value' should be removed in 'overall_only' mode."

    print("Overall metrics filtering test passed.")


def test_summarize_model_performance_invalid_combination(regression_model):
    """
    Test that summarize_model_performance raises an error when overall_only=True with classification.
    """
    model, (X, y) = regression_model

    with pytest.raises(
        ValueError,
        match="The 'overall_only' option is only valid for regression models",
    ):
        summarize_model_performance(
            model, X, y, model_type="classification", overall_only=True
        )

    print("Invalid overall_only check passed.")


@patch("builtins.print")
def test_summarize_model_performance_print_output(mock_print, regression_model):
    """
    Test that summarize_model_performance prints output correctly when return_df=False.
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


def test_get_model_probabilities(trained_model, sample_data):
    """Test extracting model probabilities."""
    X, _ = sample_data
    probs = get_model_probabilities(trained_model, X, "TestModel")

    assert len(probs) == len(X)
    assert np.all(0 <= probs) and np.all(
        probs <= 1
    )  # Probabilities should be between 0 and 1


def test_extract_model_titles(trained_model):
    """Test extracting model titles."""
    titles = extract_model_titles(
        [trained_model],
        model_titles=["LogisticRegression"],
    )
    assert titles == ["LogisticRegression"]


def test_extract_model_name(trained_model):
    """Test extracting model names."""
    name = extract_model_name(trained_model)
    assert name == "LogisticRegression"


@patch("matplotlib.pyplot.show")  # Prevents figures from displaying during testing
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
    models = [trained_model, trained_model]

    try:
        show_confusion_matrix(models, X, y, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_confusion_matrix failed for multiple models: {e}")


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
def test_show_confusion_matrix_with_class_labels(
    mock_show,
    trained_model,
    sample_data,
):
    """Test if show_confusion_matrix correctly handles custom class labels."""
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
    """Test if show_confusion_matrix correctly handles default class labels."""
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
def test_show_confusion_matrix_grid(mock_show, trained_model, sample_data):
    """Test if show_confusion_matrix correctly handles grid layout."""
    X, y = sample_data

    # Pass a list of models explicitly
    models = [trained_model, trained_model]

    print(f"DEBUG: models type = {type(models)}")
    print(f"DEBUG: models[0] type = {type(models[0])}")
    try:
        show_confusion_matrix(
            models,
            X,
            y,
            save_plot=False,
            grid=True,
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
def test_show_roc_curve_single(mock_show, trained_model, sample_data):
    """Test if show_roc_curve runs correctly for a single model."""
    X, y = sample_data
    try:
        show_roc_curve(trained_model, X, y, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_roc_curve raised an exception: {e}")


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_multiple(mock_show, trained_model, sample_data):
    """Test if show_roc_curve runs without errors for multiple models."""
    X, y = sample_data
    models = [trained_model, trained_model]
    try:
        show_roc_curve(models, X, y, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_roc_curve raised an exception: {e}")


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_overlay(mock_show, trained_model, sample_data):
    """Test if show_roc_curve runs correctly with overlay enabled."""
    X, y = sample_data
    models = [trained_model, trained_model]
    try:
        show_roc_curve(models, X, y, overlay=True, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_roc_curve raised an exception: {e}")


@patch("matplotlib.pyplot.show")
def test_show_roc_curve_grid(mock_show, trained_model, sample_data):
    """Test if show_roc_curve runs correctly with grid enabled."""
    X, y = sample_data
    models = [trained_model, trained_model]
    try:
        show_roc_curve(models, X, y, grid=True, n_cols=2, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_roc_curve raised an exception: {e}")


@patch("matplotlib.pyplot.show")
def test_show_pr_curve_single(mock_show, trained_model, sample_data):
    """Test if show_pr_curve runs correctly for a single model."""
    X, y = sample_data
    try:
        show_pr_curve(trained_model, X, y, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_pr_curve raised an exception: {e}")


@patch("matplotlib.pyplot.show")
def test_show_pr_curve_multiple(mock_show, trained_model, sample_data):
    """Test if show_pr_curve runs without errors for multiple models."""
    X, y = sample_data
    models = [trained_model, trained_model]
    try:
        show_pr_curve(models, X, y, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_pr_curve raised an exception: {e}")


@patch("matplotlib.pyplot.show")
def test_show_pr_curve_overlay(mock_show, trained_model, sample_data):
    """Test if show_pr_curve runs correctly with overlay enabled."""
    X, y = sample_data
    models = [trained_model, trained_model]
    try:
        show_pr_curve(models, X, y, overlay=True, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_pr_curve raised an exception: {e}")


@patch("matplotlib.pyplot.show")
def test_show_pr_curve_grid(mock_show, trained_model, sample_data):
    """Test if show_pr_curve runs correctly with grid enabled."""
    X, y = sample_data
    models = [trained_model, trained_model]
    try:
        show_pr_curve(models, X, y, grid=True, n_cols=2, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_pr_curve raised an exception: {e}")


@patch("matplotlib.pyplot.show")
def test_show_lift_chart_single(mock_show, trained_model, sample_data):
    """Test if show_lift_chart runs correctly for a single model."""
    X, y = sample_data
    try:
        show_lift_chart(trained_model, X, y, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_lift_chart raised an exception: {e}")


@patch("matplotlib.pyplot.show")
def test_show_lift_chart_multiple(mock_show, trained_model, sample_data):
    """Test if show_lift_chart runs without errors for multiple models."""
    X, y = sample_data
    models = [trained_model, trained_model]
    try:
        show_lift_chart(models, X, y, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_lift_chart raised an exception: {e}")


@patch("matplotlib.pyplot.show")
def test_show_lift_chart_overlay(mock_show, trained_model, sample_data):
    """Test if show_lift_chart runs correctly with overlay enabled."""
    X, y = sample_data
    models = [trained_model, trained_model]
    try:
        show_lift_chart(models, X, y, overlay=True, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_lift_chart raised an exception: {e}")


@patch("matplotlib.pyplot.show")
def test_show_lift_chart_grid(mock_show, trained_model, sample_data):
    """Test if show_lift_chart runs correctly with grid enabled."""
    X, y = sample_data
    models = [trained_model, trained_model]
    try:
        show_lift_chart(models, X, y, grid=True, n_cols=2, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_lift_chart raised an exception: {e}")


@patch("matplotlib.pyplot.show")
def test_show_lift_chart_invalid_overlay_grid(
    mock_show,
    trained_model,
    sample_data,
):
    """Ensure ValueError is raised if both overlay and grid are set to True."""
    X, y = sample_data
    with pytest.raises(
        ValueError, match="`grid` cannot be set to True when `overlay` is True."
    ):
        show_lift_chart(
            [trained_model, trained_model],
            X,
            y,
            overlay=True,
            grid=True,
        )


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
    """Test if show_gain_chart runs without errors for multiple models."""
    X, y = sample_data
    models = [trained_model, trained_model]
    try:
        show_gain_chart(models, X, y, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_gain_chart raised an exception: {e}")


@patch("matplotlib.pyplot.show")
def test_show_gain_chart_overlay(mock_show, trained_model, sample_data):
    """Test if show_gain_chart runs correctly with overlay enabled."""
    X, y = sample_data
    models = [trained_model, trained_model]
    try:
        show_gain_chart(models, X, y, overlay=True, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_gain_chart raised an exception: {e}")


@patch("matplotlib.pyplot.show")
def test_show_gain_chart_grid(mock_show, trained_model, sample_data):
    """Test if show_gain_chart runs correctly with grid enabled."""
    X, y = sample_data
    models = [trained_model, trained_model]
    try:
        show_gain_chart(models, X, y, grid=True, n_cols=2, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_gain_chart raised an exception: {e}")


@patch("matplotlib.pyplot.show")
def test_show_gain_chart_invalid_overlay_grid(
    mock_show,
    trained_model,
    sample_data,
):
    """Ensure ValueError is raised if both overlay and grid are set to True."""
    X, y = sample_data
    with pytest.raises(
        ValueError, match="`grid` cannot be set to True when `overlay` is True."
    ):
        show_gain_chart(
            [trained_model, trained_model],
            X,
            y,
            overlay=True,
            grid=True,
        )


@patch("matplotlib.pyplot.show")
def test_show_lift_chart_saves_plot(
    mock_show,
    trained_model,
    sample_data,
    tmp_path,
):
    """Test if show_lift_chart saves the plot when save_plot=True."""
    X, y = sample_data
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
def test_show_gain_chart_saves_plot(
    mock_show,
    trained_model,
    sample_data,
    tmp_path,
):
    """Test if show_gain_chart saves the plot when save_plot=True."""
    X, y = sample_data
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
    models = [trained_model, trained_model]  # Use the same model twice
    try:
        show_calibration_curve(models, X, y, overlay=True, title="Overlay Test")
    except Exception as e:
        pytest.fail(f"show_calibration_curve failed on overlay: {e}")


def test_show_calibration_curve_grid(trained_model, sample_data):
    """Test grid layout with multiple models."""
    X, y = sample_data
    models = [trained_model, trained_model]  # Ensure it's a list
    try:
        show_calibration_curve(model=models, X=X, y=y, grid=True, n_cols=2)
    except Exception as e:
        pytest.fail(f"show_calibration_curve failed on grid layout: {e}")


def test_show_calibration_curve_invalid_overlay_grid(trained_model, sample_data):
    """Ensure ValueError is raised if both overlay and grid are set to True."""
    X, y = sample_data
    with pytest.raises(
        ValueError, match="`grid` cannot be set to True when `overlay` is True."
    ):
        show_calibration_curve(
            [trained_model, trained_model], X, y, overlay=True, grid=True
        )


def test_show_calibration_curve_save_plot(trained_model, sample_data, tmp_path):
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
    """Test custom model titles for grid layout."""
    X, y = sample_data
    models = [trained_model, trained_model]  # Ensure it's a list
    titles = ["Model 1", "Model 2"]  # Titles must match models length
    try:
        show_calibration_curve(
            model=models,
            X=X,
            y=y,
            model_titles=titles,
            grid=True,
        )
    except Exception as e:
        pytest.fail(f"show_calibration_curve failed with custom titles: {e}")


def test_get_model_probabilities_predict_proba(trained_model, sample_data):
    """Test direct model with `predict_proba()`."""
    X, _ = sample_data
    probabilities = get_model_probabilities(trained_model, X, "Logistic Model")
    assert probabilities.shape == (X.shape[0],)  # Should return 1D array
    assert np.all(
        (probabilities >= 0) & (probabilities <= 1)
    )  # Valid probability range


def test_get_model_probabilities_pipeline(sample_data):
    """Test pipeline where final step has `predict_proba()`."""
    X, y = sample_data
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression()),  # Has predict_proba()
        ]
    )
    pipeline.fit(X, y)

    probabilities = get_model_probabilities(pipeline, X, "Pipeline Model")
    assert probabilities.shape == (X.shape[0],)
    assert np.all(
        (probabilities >= 0) & (probabilities <= 1)
    )  # Valid probability range


def test_get_model_probabilities_decision_function(sample_data):
    """Test standalone model that has only `decision_function()`."""
    X, y = sample_data
    model = SVC(probability=False)  # No predict_proba(), only decision_function()
    model.fit(X, y)

    probabilities = get_model_probabilities(model, X, "SVM Model")
    assert probabilities.shape == (X.shape[0],)
    assert np.all(
        (probabilities >= 0) & (probabilities <= 1)
    )  # Probability conversion check


def test_get_model_probabilities_invalid_model():
    """Test model that does not support probability-based predictions."""

    class DummyModel:
        def fit(self, X, y):
            pass

        def predict(self, X):
            return np.ones(len(X))

    model = DummyModel()
    X = np.random.rand(5, 3)

    with pytest.raises(
        ValueError, match="does not support probability-based prediction"
    ):
        get_model_probabilities(model, X, "Dummy Model")


@patch("matplotlib.pyplot.show")
def test_show_ks_curve_single_model(mock_show, trained_model, sample_data):
    """Test show_ks_curve with a single model."""
    X, y = sample_data  # Get sample test data
    try:
        show_ks_curve(trained_model, X, y, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_ks_curve failed with a single model: {e}")


@patch("matplotlib.pyplot.show")
def test_show_ks_curve_multiple_models(mock_show, trained_model, sample_data):
    """Test show_ks_curve with multiple models."""
    X, y = sample_data  # Get sample test data
    models = [trained_model, trained_model]  # Using the same model twice for simplicity
    try:
        show_ks_curve(models, X, y, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_ks_curve failed with multiple models: {e}")


@patch("matplotlib.pyplot.show")
def test_show_ks_curve_saves_plot(mock_show, trained_model, sample_data, tmp_path):
    """Test if show_ks_curve saves the plot correctly."""
    X, y = sample_data  # Get sample test data
    image_path_png = tmp_path / "ks_curve.png"
    image_path_svg = tmp_path / "ks_curve.svg"

    try:
        show_ks_curve(
            trained_model,
            X,
            y,
            save_plot=True,
            image_path_png=str(image_path_png),
            image_path_svg=str(image_path_svg),
        )
    except Exception as e:
        pytest.fail(f"show_ks_curve failed when saving plot: {e}")

    assert image_path_png.exists(), "PNG image was not saved."
    assert image_path_svg.exists(), "SVG image was not saved."


@patch("matplotlib.pyplot.show")
def test_show_ks_curve_empty_groups(mock_show, trained_model, sample_data):
    """Test show_ks_curve when one group is empty."""
    X, _ = sample_data  # Extract feature data
    y = np.zeros(len(X))  # Set all labels to zero (no positives)

    try:
        show_ks_curve(trained_model, X, y, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_ks_curve failed when handling empty groups: {e}")


@patch("matplotlib.pyplot.show")
def test_show_ks_curve_custom_threshold(mock_show, trained_model, sample_data):
    """Test show_ks_curve with a custom threshold."""
    X, y = sample_data  # Get sample test data
    try:
        show_ks_curve(trained_model, X, y, model_threshold=0.7, save_plot=False)
    except Exception as e:
        pytest.fail(f"show_ks_curve failed with custom threshold: {e}")


@patch("matplotlib.pyplot.show")
def test_show_ks_curve_custom_labels(mock_show, trained_model, sample_data):
    """Test show_ks_curve with custom axis labels and title."""
    X, y = sample_data  # Get sample test data
    try:
        show_ks_curve(
            trained_model,
            X,
            y,
            xlabel="Custom X",
            ylabel="Custom Y",
            title="Custom Title",
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(f"show_ks_curve failed with custom labels: {e}")


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


########


def test_plot_threshold_metrics_execution(trained_model, sample_data):
    """Test if plot_threshold_metrics runs without errors."""
    X, y = sample_data
    try:
        plot_threshold_metrics(trained_model, X, y, save_plot=False)
    except Exception as e:
        pytest.fail(f"plot_threshold_metrics raised an exception: {e}")


def test_plot_threshold_metrics_lookup_metric(trained_model, sample_data):
    """Test lookup metric functionality."""
    X, y = sample_data
    lookup_metric = "precision"
    lookup_value = 0.7

    try:
        plot_threshold_metrics(
            trained_model,
            X,
            y,
            lookup_metric=lookup_metric,
            lookup_value=lookup_value,
            save_plot=False,
        )
    except Exception as e:
        pytest.fail(f"plot_threshold_metrics failed with lookup metric: {e}")


@pytest.mark.parametrize("decimal_places", [2, 4, 6])
def test_plot_threshold_metrics_decimal_places(
    trained_model, sample_data, decimal_places
):
    """Test if decimal_places parameter correctly formats outputs."""
    X, y = sample_data
    lookup_metric = "recall"
    lookup_value = 0.5

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
            f"plot_threshold_metrics failed for decimal_places={decimal_places}: {e}"
        )


def test_plot_threshold_metrics_save_plot(trained_model, sample_data, tmp_path):
    """Test if plot_threshold_metrics saves the plot correctly."""
    X, y = sample_data
    image_path_png = tmp_path / "threshold_plot.png"
    image_path_svg = tmp_path / "threshold_plot.svg"

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
