import os
import pytest
from unittest import mock
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")  # hard-stop any interactive GUI use
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from model_metrics.partial_dependence import plot_2d_pdp, plot_3d_pdp


# --- Global hygiene to prevent freezes / GUI calls ---------------------------------
@pytest.fixture(autouse=True)
def _matplotlib_hygiene():
    """Prevent interactive windows and clean up figures after every test."""
    # make sure plt.show() is a noop during tests
    orig_show = plt.show
    plt.show = lambda *a, **k: None
    try:
        yield
    finally:
        plt.close("all")
        plt.show = orig_show


@pytest.fixture(autouse=True)
def _plotly_hygiene(monkeypatch, tmp_path):
    """
    Disable interactive Plotly outputs and ensure html files are created
    deterministically when requested, without opening a browser.
    """
    try:
        import plotly.offline as po
    except Exception:
        # If plotly isn't installed in CI, tests that rely on it will be skipped anyway.
        yield
        return

    # iplot should never try to display in tests
    monkeypatch.setattr(po, "iplot", lambda *a, **k: None, raising=False)

    def _safe_plot(fig=None, filename=None, auto_open=False, *a, **k):
        # Respect provided filename, create minimal html to satisfy existence checks
        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w", encoding="utf-8") as f:
                f.write("<html><head></head><body>test</body></html>")
        return filename

    # Make plotly.offline.plot safe and non-blocking; never open a browser
    monkeypatch.setattr(po, "plot", _safe_plot, raising=False)
    yield


# -----------------------------------------------------------------------------------


@pytest.fixture
def trained_model_and_data():
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        random_state=42,
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    model = RandomForestClassifier(random_state=42).fit(X_df, y)
    return model, X_df, feature_names


def test_plot_2d_pdp_grid(trained_model_and_data):
    model, X_df, feature_names = trained_model_and_data
    plot_2d_pdp(
        model=model,
        X_train=X_df,
        feature_names=feature_names,
        features=[0, 1],
        plot_type="grid",
        title="Grid Plot",
    )


def test_plot_2d_pdp_individual(trained_model_and_data):
    model, X_df, feature_names = trained_model_and_data
    plot_2d_pdp(
        model=model,
        X_train=X_df,
        feature_names=feature_names,
        features=[0, 1],
        plot_type="individual",
        title="Individual Plot",
    )


def test_plot_2d_invalid_plot_type(trained_model_and_data):
    model, X_df, feature_names = trained_model_and_data
    with pytest.raises(ValueError):
        plot_2d_pdp(
            model=model,
            X_train=X_df,
            feature_names=feature_names,
            features=[0],
            plot_type="bogus",
        )


def test_plot_2d_invalid_save_option(trained_model_and_data):
    model, X_df, feature_names = trained_model_and_data
    with pytest.raises(ValueError):
        plot_2d_pdp(
            model=model,
            X_train=X_df,
            feature_names=feature_names,
            features=[0],
            save_plots="invalid",
        )


def test_plot_2d_pdp_save_grid_png(tmp_path, trained_model_and_data):
    model, X_df, feature_names = trained_model_and_data
    plot_2d_pdp(
        model=model,
        X_train=X_df,
        feature_names=feature_names,
        features=[0, 1],
        plot_type="grid",
        save_plots="grid",
        image_path_png=str(tmp_path),
        file_prefix="test_plot",
    )
    expected = tmp_path / "test_plot_2d_pdp_grid.png"
    assert expected.exists()


def test_2d_pdp_save_plots_missing_paths(trained_model_and_data):
    model, X_df, feature_names = trained_model_and_data
    with pytest.raises(ValueError, match="To save plots"):
        plot_2d_pdp(
            model=model,
            X_train=X_df,
            feature_names=feature_names,
            features=[0, 1],
            plot_type="grid",
            save_plots="grid",  # save_plots is set
            # BUT no image_path_png or image_path_svg
        )


def test_2d_pdp_save_individual_svg(tmp_path, trained_model_and_data):
    model, X_df, feature_names = trained_model_and_data
    plot_2d_pdp(
        model=model,
        X_train=X_df,
        feature_names=feature_names,
        features=[0, 1],
        plot_type="individual",
        save_plots="individual",
        image_path_svg=str(tmp_path),
    )

    assert any(str(f).endswith(".svg") for f in tmp_path.iterdir())


def test_2d_pdp_save_grid_svg(tmp_path, trained_model_and_data):
    model, X_df, feature_names = trained_model_and_data
    plot_2d_pdp(
        model=model,
        X_train=X_df,
        feature_names=feature_names,
        features=[0, 1],
        plot_type="grid",
        save_plots="grid",
        image_path_svg=str(tmp_path),
        file_prefix="test_prefix",
    )
    expected = tmp_path / "test_prefix_2d_pdp_grid.svg"
    assert expected.exists()


def test_plot_3d_pdp_static(trained_model_and_data):
    model, X_df, feature_names = trained_model_and_data
    plot_3d_pdp(
        model=model,
        dataframe=X_df,
        feature_names=[feature_names[0], feature_names[1]],
        title="Static PDP",
        plot_type="static",
    )


def test_plot_3d_pdp_invalid_type(trained_model_and_data):
    model, X_df, feature_names = trained_model_and_data
    with pytest.raises(ValueError):
        plot_3d_pdp(
            model=model,
            dataframe=X_df,
            feature_names=[feature_names[0], feature_names[1]],
            title="Invalid",
            plot_type="bad_type",
        )


def test_plot_3d_pdp_interactive_missing_path(trained_model_and_data):
    model, X_df, feature_names = trained_model_and_data
    with pytest.raises(ValueError):
        plot_3d_pdp(
            model=model,
            dataframe=X_df,
            feature_names=[feature_names[0], feature_names[1]],
            plot_type="interactive",
        )


def test_plot_3d_pdp_save_html(tmp_path, trained_model_and_data):
    model, X_df, feature_names = trained_model_and_data
    expected_file = tmp_path / "plot_3d_pdp.html"  # function hardcodes this

    plot_3d_pdp(
        model=model,
        dataframe=X_df,
        feature_names=[feature_names[0], feature_names[1]],
        plot_type="both",  # ensure saving logic executes
        save_plots="html",
        html_file_path=str(tmp_path),
        html_file_name="ignored.html",  # ignored by implementation
        title="Test Save HTML",
    )

    assert expected_file.exists()


def test_3d_pdp_invalid_save_plots(trained_model_and_data):
    model, X_df, feature_names = trained_model_and_data
    with pytest.raises(ValueError, match="Invalid `save_plots` value"):
        plot_3d_pdp(
            model=model,
            dataframe=X_df,
            feature_names=[feature_names[0], feature_names[1]],
            save_plots="nonsense",
            plot_type="both",
        )


def test_3d_pdp_save_static_missing_paths(trained_model_and_data):
    model, X_df, feature_names = trained_model_and_data
    with pytest.raises(ValueError, match="To save static plots"):
        plot_3d_pdp(
            model=model,
            dataframe=X_df,
            feature_names=[feature_names[0], feature_names[1]],
            save_plots="static",
            plot_type="both",
        )


def test_3d_pdp_html_missing_path(trained_model_and_data):
    model, X_df, feature_names = trained_model_and_data
    with pytest.raises(ValueError, match="provide `html_file_path`"):
        plot_3d_pdp(
            model=model,
            dataframe=X_df,
            feature_names=[feature_names[0], feature_names[1]],
            save_plots="html",
            plot_type="both",
            html_file_name="plot.html",
        )


def test_3d_pdp_interactive_missing_html_filename(trained_model_and_data):
    model, X_df, feature_names = trained_model_and_data
    with pytest.raises(
        ValueError, match="must be provided for 'interactive' or 'both'"
    ):
        plot_3d_pdp(
            model=model,
            dataframe=X_df,
            feature_names=[feature_names[0], feature_names[1]],
            plot_type="interactive",
            html_file_path="some/path",
            # missing html_file_name
        )


def test_plot_3d_pdp_fallback_to_plot(tmp_path, trained_model_and_data):
    model, X_df, feature_names = trained_model_and_data

    with mock.patch(
        "plotly.offline.iplot",
        side_effect=ImportError,
    ), mock.patch("plotly.offline.plot") as mock_plot:
        plot_3d_pdp(
            model=model,
            dataframe=X_df,
            feature_names=[feature_names[0], feature_names[1]],
            plot_type="interactive",
            save_plots="html",
            html_file_path=str(tmp_path),
            html_file_name="interactive_fallback.html",
            title="Fallback Plot",
        )
        mock_plot.assert_called_once()
        # Do NOT check for .exists() — file wasn't actually written


def test_3d_pdp_save_static_png_svg(tmp_path, trained_model_and_data):
    model, X_df, feature_names = trained_model_and_data

    plot_3d_pdp(
        model=model,
        dataframe=X_df,
        feature_names=[feature_names[0], feature_names[1]],
        plot_type="static",
        save_plots="static",
        image_path_png=str(tmp_path),
        image_path_svg=str(tmp_path),
        title="Save Static",
    )

    assert (tmp_path / "plot_3d_pdp.png").exists()
    assert (tmp_path / "plot_3d_pdp.svg").exists()


def test_3d_pdp_static_plot_and_html(tmp_path, trained_model_and_data):
    model, X_df, feature_names = trained_model_and_data
    plot_3d_pdp(
        model=model,
        dataframe=X_df,
        feature_names=[feature_names[0], feature_names[1]],
        plot_type="both",
        save_plots="both",
        image_path_png=str(tmp_path),
        html_file_path=str(tmp_path),
        html_file_name="combo_plot.html",
        title="Combo Save",
    )

    assert (tmp_path / "plot_3d_pdp.png").exists()
    assert (tmp_path / "combo_plot.html").exists()
