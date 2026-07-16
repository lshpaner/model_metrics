import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# save_plot_images may live in metrics_utils (current) or plot_utils (after the
# recommended move); import from wherever it resolves.
try:
    from model_metrics.plot_utils import save_plot_images
except ImportError:
    from model_metrics.metrics_utils import save_plot_images


@pytest.fixture(autouse=True)
def _mpl_hygiene():
    orig_show = plt.show
    plt.show = lambda *a, **k: None
    try:
        yield
    finally:
        plt.close("all")
        plt.show = orig_show


@pytest.fixture
def fig():
    f, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])
    return f


@pytest.mark.parametrize("ext", ["png", "jpg", "pdf", "svg", "tiff", "eps"])
def test_arbitrary_extension_saved_verbatim(tmp_path, fig, ext):
    target = tmp_path / f"figure.{ext}"
    save_plot_images("auto", False, None, None, image_filename=str(target), fig=fig)
    assert target.exists()


def test_legacy_png_and_svg_dirs(tmp_path, fig):
    png_dir = tmp_path / "png"
    svg_dir = tmp_path / "svg"
    save_plot_images("myplot", True, str(png_dir), str(svg_dir), fig=fig)
    assert (png_dir / "myplot.png").exists()
    assert (svg_dir / "myplot.svg").exists()


def test_verbatim_and_legacy_combined(tmp_path, fig):
    png_dir = tmp_path / "png"
    save_plot_images(
        "auto", False, str(png_dir), None,
        image_filename=str(tmp_path / "combo.pdf"), fig=fig,
    )
    assert (tmp_path / "combo.pdf").exists()
    assert (png_dir / "combo.png").exists()  # stem taken from image_filename


def test_image_filename_triggers_save_even_when_save_plot_false(tmp_path, fig):
    target = tmp_path / "triggered.png"
    save_plot_images("auto", False, None, None, image_filename=str(target), fig=fig)
    assert target.exists()


def test_bare_stem_no_dir_raises(fig):
    with pytest.raises(ValueError):
        save_plot_images("auto", False, None, None, image_filename="noext", fig=fig)


def test_noop_when_nothing_requested(fig):
    # save_plot False and no image_filename -> returns without writing anything
    assert save_plot_images("auto", False, None, None, fig=fig) is None
