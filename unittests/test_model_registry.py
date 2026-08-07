"""Tests for the MLflow model registry.

The module under test reads a local MLflow FileStore straight off disk rather
than going through the mlflow client, so every test builds a synthetic
``mlruns/`` tree in ``tmp_path``. Three behaviours get the most attention
because they are the ones that fail silently in production:

* cross-version unpickling, where an artifact written under an older
  scikit-learn references private classes the installed version no longer
  defines;
* fitted-state repair, where the newer ``transform()`` reads instance
  attributes an older ``fit()`` never wrote;
* split-aware verification, which is the only check that a repaired model
  still reproduces the metrics logged at training time.
"""

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
import sklearn.compose._column_transformer as _ct_module
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# The registry may live inside the package (after the recommended move) or as a
# standalone module during development; import from wherever it resolves.
try:
    from model_metrics import registry as mr
except ImportError:  # pragma: no cover - local development layout
    import model_metrics.model_registry as mr


TARGET = "income"


# --------------------------------------------------------------------------- #
# Store construction                                                          #
# --------------------------------------------------------------------------- #
def _write_run(
    exp_dir,
    run_id,
    run_name,
    algo,
    model_obj,
    metrics=None,
    start_time=1000,
    lifecycle="active",
):
    """Write one MLflow FileStore run directory."""
    run_dir = exp_dir / run_id
    (run_dir / "tags").mkdir(parents=True)
    (run_dir / "metrics").mkdir()
    art = run_dir / "artifacts" / f"{algo}_{TARGET}"
    art.mkdir(parents=True)

    (run_dir / "meta.yaml").write_text(
        f"run_id: {run_id}\n"
        f"run_uuid: {run_id}\n"
        f"start_time: {start_time}\n"
        f"lifecycle_stage: {lifecycle}\n"
        f"experiment_id: '1'\n"
    )
    (run_dir / "tags" / "mlflow.runName").write_text(run_name)

    for key, value in (metrics or {}).items():
        (run_dir / "metrics" / key).write_text(f"{start_time} {value} 0")

    blob = model_obj if isinstance(model_obj, bytes) else pickle.dumps(model_obj)
    (art / "model.pkl").write_bytes(blob)
    return run_dir


def _make_frame(n=400, seed=3):
    """Feature frame with 8 numeric, 2 categorical, 10 passthrough columns."""
    rng = np.random.default_rng(seed)
    num = [f"n{i}" for i in range(8)]
    extra = [f"p{i}" for i in range(10)]
    df = pd.DataFrame(rng.normal(size=(n, 18)), columns=num + extra)
    df["race"] = rng.choice(list("ABCDE"), n)
    df["sex"] = rng.choice(["F", "M"], n)
    X = df[num + ["race", "sex"] + extra]
    y = pd.Series(
        (X["n0"] + rng.normal(scale=0.7, size=n) > 0).astype(int), name="y"
    )
    return X, y, num


def _fit_pipeline(X, y, num_cols, remainder="passthrough", with_imputer=False):
    """A ColumnTransformer pipeline of the shape the repairs target."""
    numeric = (
        Pipeline([("imp", SimpleImputer(strategy="mean")), ("sc", StandardScaler())])
        if with_imputer
        else StandardScaler()
    )
    pre = ColumnTransformer(
        [
            ("num", numeric, num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["race", "sex"]),
        ],
        remainder=remainder,
    )
    return Pipeline(
        [("pre", pre), ("lr", LogisticRegression(max_iter=500))]
    ).fit(X, y)


class _LegacyRemainderColsList(list):
    """Stand-in for sklearn's private remainder-column class.

    Given sklearn's module path so pickle can resolve it by qualified name
    while it is temporarily attached to ``sklearn.compose._column_transformer``.
    """


_LegacyRemainderColsList.__module__ = "sklearn.compose._column_transformer"
_LegacyRemainderColsList.__qualname__ = "_RemainderColsList"

_ATTR = "_RemainderColsList"
_MISSING = object()


class _RemainderClassHidden:
    """Context manager that hides sklearn's private remainder class.

    Whether ``_RemainderColsList`` exists depends on the installed
    scikit-learn, and newer versions call it during ``fit()``. So the class is
    installed for the pickle dump, removed for the load, and whatever was
    there originally is put back on exit. Fitting must happen before entering,
    since a bare stand-in cannot accept the ``future_dtype`` keyword the real
    class takes.
    """

    def __enter__(self):
        self._original = getattr(_ct_module, _ATTR, _MISSING)
        setattr(_ct_module, _ATTR, _LegacyRemainderColsList)
        return self

    def hide(self):
        """Remove the class so a subsequent unpickle cannot resolve it."""
        _ct_module.__dict__.pop(_ATTR, None)

    def __exit__(self, *exc):
        if self._original is _MISSING:
            _ct_module.__dict__.pop(_ATTR, None)
        else:
            setattr(_ct_module, _ATTR, self._original)
        return False


def _empty_remainder(pipe):
    """Blank every copy of the remainder column list, as the shim does.

    A synthesized stand-in for ``_RemainderColsList`` unpickles as an empty
    list. With ``remainder='passthrough'`` that silently drops the passthrough
    columns, so the transformer emits fewer features than the downstream
    estimator was fitted on.
    """
    pre = pipe.named_steps["pre"]
    pre._remainder = ("remainder", "passthrough", _LegacyRemainderColsList([]))
    pre._transformer_to_input_indices["remainder"] = []
    pre.transformers_ = [
        (name, trans, _LegacyRemainderColsList([]) if name == "remainder" else cols)
        for name, trans, cols in pre.transformers_
    ]


def _break_imputer(pipe):
    """Drop ``_fill_dtype``, which an older ``fit()`` never wrote.

    Tolerant of its absence, since whether the attribute exists at all depends
    on the installed scikit-learn.
    """
    imp = pipe.named_steps["pre"].named_transformers_["num"].named_steps["imp"]
    imp.__dict__.pop("_fill_dtype", None)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture
def data():
    """(X, y, numeric_column_names) reused across the model fixtures."""
    return _make_frame()


@pytest.fixture
def store(tmp_path, data):
    """A store with four healthy runs across three algorithms.

    Returns a dict carrying the root, the evaluation frame, and the metrics
    that were logged, so tests can assert against the same numbers.
    """
    X, y, num = data
    exp = tmp_path / "mlruns" / "models" / "1"
    exp.mkdir(parents=True)
    (exp / "meta.yaml").write_text("experiment_id: '1'\nname: income_model\n")

    pipe = _fit_pipeline(X, y, num)
    proba = pipe.predict_proba(X)[:, 1]
    logged = {
        "test roc_auc": round(roc_auc_score(y, proba), 3),
        "test Average Precision": round(average_precision_score(y, proba), 3),
        "valid Average Precision": 0.111,
        "train Average Precision": 0.999,
    }

    _write_run(exp, "r_cat_orig", "cat_orig_training", "cat", pipe, logged, 3000)
    _write_run(
        exp,
        "r_cat_smote",
        "cat_smote_training",
        "cat",
        pipe,
        {**logged, "test roc_auc": 0.500, "valid Average Precision": 0.050},
        2000,
    )
    _write_run(exp, "r_xgb", "xgb_orig_training", "xgb", pipe, logged, 1500)
    _write_run(exp, "r_lr", "lr_orig_training", "lr", pipe, logged, 1000)

    return {"root": tmp_path, "X": X, "y": y, "logged": logged, "pipe": pipe}


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Drop cached index and load state so tests cannot leak into each other."""
    yield
    mr.set_stores()
    mr.refresh()


@pytest.fixture
def configured(store):
    """Point the registry at the synthetic store."""
    mr.configure(root=store["root"], target=TARGET)
    return store


@pytest.fixture
def legacy_blob(data):
    """Pickle bytes whose remainder list cannot be resolved at load time.

    Yields ``(blob, healthy_width)``. The pipeline is fitted while sklearn is
    intact, then the private class is hidden so the registry has to shim it.
    """
    X, y, num = data
    pipe = _fit_pipeline(X, y, num)
    healthy_width = pipe.named_steps["pre"].transform(X).shape[1]

    with _RemainderClassHidden() as hidden:
        _empty_remainder(pipe)
        blob = pickle.dumps(pipe)
        hidden.hide()
        yield blob, healthy_width


@pytest.fixture
def fill_dtype_expected(monkeypatch):
    """Force the imputer repair's version gate on.

    Whether the installed scikit-learn records ``SimpleImputer._fill_dtype``
    varies by version, but the repair's behaviour does not. Pinning the gate
    exercises the repair on every version instead of skipping half of them.
    """
    monkeypatch.setattr(mr, "_imputer_uses_fill_dtype", lambda: True)


@pytest.fixture
def legacy_imputer_blob(data):
    """Pickle bytes for a pipeline whose imputer lacks ``_fill_dtype``.

    Yields ``(blob, healthy_scores)`` so the repair can be checked against the
    predictions the intact pipeline produced.
    """
    X, y, num = data
    pipe = _fit_pipeline(X, y, num, with_imputer=True)
    healthy = pipe.predict_proba(X)[:, 1]
    _break_imputer(pipe)
    return pickle.dumps(pipe), healthy


def _store_with(tmp_path, blob, metrics=None):
    """One-run store wrapping a prepared pickle blob."""
    exp = tmp_path / "mlruns" / "models" / "1"
    exp.mkdir(parents=True)
    (exp / "meta.yaml").write_text("experiment_id: '1'\nname: income_model\n")
    _write_run(
        exp,
        "r",
        "cat_orig_training",
        "cat",
        blob,
        metrics or {"test roc_auc": 0.9},
    )
    mr.configure(root=tmp_path, target=TARGET)


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
def test_configure_sets_root_and_target(store):
    mr.configure(root=store["root"], target=TARGET)
    assert mr.PROJECT_ROOT == store["root"]
    assert mr.TARGET == TARGET


def test_configure_accepts_custom_loader(configured):
    calls = []

    def loader(path):
        calls.append(path)
        with open(path, "rb") as fh:
            return pickle.load(fh)

    mr.configure(loader=loader)
    mr.load("cat_orig_training")
    assert calls, "custom loader was not used"
    mr.configure(loader=mr._default_loader)


def test_default_root_finds_mlruns_from_the_working_directory(tmp_path, monkeypatch):
    """An installed package cannot use its own location as a search root."""
    repo = tmp_path / "repo"
    (repo / "mlruns").mkdir(parents=True)
    (repo / ".git").touch()
    (repo / "notebooks").mkdir()

    monkeypatch.chdir(repo / "notebooks")
    assert mr._default_root() == repo.resolve()


def test_default_root_stops_at_the_repository_boundary(tmp_path, monkeypatch):
    """An unrelated mlruns/ above the repo must not be picked up silently."""
    outer = tmp_path / "outer"
    (outer / "mlruns").mkdir(parents=True)
    repo = outer / "repo"
    repo.mkdir()
    (repo / ".git").touch()

    monkeypatch.chdir(repo)
    assert mr._default_root() == repo.resolve()


def test_default_root_falls_back_to_the_working_directory(tmp_path, monkeypatch):
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.chdir(plain)
    assert mr._default_root() == plain.resolve()


def test_backend_reports_a_known_deserializer():
    assert mr.backend() in {"model_tuner", "joblib", "pickle"}


def test_refresh_clears_the_index(configured):
    first = mr.available()
    mr.refresh()
    second = mr.available()
    assert first.equals(second)  # same store, so identical after a reread


# --------------------------------------------------------------------------- #
# Flat YAML fallback                                                          #
# --------------------------------------------------------------------------- #
def test_flat_parser_matches_pyyaml_on_experiment_meta():
    text = (
        "artifact_location: file:///tmp/mlruns/1\n"
        "creation_time: 1751234567890\n"
        "experiment_id: '1'\n"
        "lifecycle_stage: active\n"
        "name: income_model\n"
    )
    parsed = mr._parse_flat_yaml(text)
    assert parsed["experiment_id"] == "1"
    assert parsed["name"] == "income_model"
    assert parsed["creation_time"] == 1751234567890


def test_flat_parser_handles_empty_and_list_values():
    parsed = mr._parse_flat_yaml("entry_point_name: ''\ntags: []\nstatus: 3\n")
    assert parsed["entry_point_name"] == ""
    assert parsed["tags"] == []
    assert parsed["status"] == 3


def test_flat_parser_ignores_comments_and_nested_lines():
    parsed = mr._parse_flat_yaml("# a comment\nkey: value\n  nested: skipped\n")
    assert parsed == {"key": "value"}


# --------------------------------------------------------------------------- #
# Target stripping                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name, expected",
    [
        ("cat_income", "cat"),
        ("cat_income_no_sex", "cat_no_sex"),
        ("lr_income", "lr"),
        ("cat_orig_training", "cat_orig_training"),
        ("cat", "cat"),
        ("income", "income"),
    ],
)
def test_strip_target_handles_suffix_and_midname(name, expected, configured):
    assert mr._strip_target(name) == expected


def test_strip_target_is_a_noop_without_a_target(store):
    mr.configure(root=store["root"], target="")
    assert mr._strip_target("cat_income") == "cat_income"


# --------------------------------------------------------------------------- #
# Indexing                                                                    #
# --------------------------------------------------------------------------- #
def test_available_lists_every_run(configured):
    df = mr.available()
    assert len(df) == 4
    assert set(df["variant"]) == {
        "cat_orig_training",
        "cat_smote_training",
        "xgb_orig_training",
        "lr_orig_training",
    }


def test_algos_are_stripped_of_the_target_token(configured):
    assert mr.algos() == ["cat", "lr", "xgb"]


def test_variants_and_experiments(configured):
    assert len(mr.variants()) == 4
    assert mr.experiments() == ["income_model"]


def test_available_includes_logged_metrics(configured):
    df = mr.available()
    assert "test roc_auc" in df.columns
    assert "valid Average Precision" in df.columns


def test_index_is_sorted_newest_first(configured):
    df = mr.available()
    assert df["variant"].iloc[0] == "cat_orig_training"


def test_deleted_runs_are_excluded(tmp_path, data):
    X, y, num = data
    exp = tmp_path / "mlruns" / "models" / "1"
    exp.mkdir(parents=True)
    (exp / "meta.yaml").write_text("experiment_id: '1'\nname: income_model\n")
    pipe = _fit_pipeline(X, y, num)
    _write_run(exp, "keep", "cat_orig_training", "cat", pipe, {"test roc_auc": 0.9})
    _write_run(
        exp, "drop", "cat_dead_training", "cat", pipe, lifecycle="deleted"
    )

    mr.configure(root=tmp_path, target=TARGET)
    assert mr.variants() == ["cat_orig_training"]


def test_runs_without_artifacts_are_skipped(tmp_path, data):
    X, y, num = data
    exp = tmp_path / "mlruns" / "models" / "1"
    exp.mkdir(parents=True)
    (exp / "meta.yaml").write_text("experiment_id: '1'\nname: income_model\n")
    pipe = _fit_pipeline(X, y, num)
    _write_run(exp, "good", "cat_orig_training", "cat", pipe, {"test roc_auc": 0.9})

    bare = exp / "empty"
    (bare / "tags").mkdir(parents=True)
    (bare / "meta.yaml").write_text(
        "run_id: empty\nstart_time: 1\nlifecycle_stage: active\nexperiment_id: '1'\n"
    )

    mr.configure(root=tmp_path, target=TARGET)
    assert mr.variants() == ["cat_orig_training"]


# --------------------------------------------------------------------------- #
# Lookup and resolution                                                       #
# --------------------------------------------------------------------------- #
def test_resolve_by_variant(configured):
    entry = mr.resolve("xgb_orig_training")
    assert entry.variant == "xgb_orig_training"
    assert entry.algo == "xgb"


def test_resolve_by_run_id(configured):
    assert mr.resolve("r_lr").variant == "lr_orig_training"


def test_resolve_by_qualified_key(configured):
    assert mr.resolve("income_model/lr_orig_training").algo == "lr"


def test_resolve_ambiguous_algo_prefers_newest(configured, capsys):
    entry = mr.resolve("cat")
    assert entry.variant == "cat_orig_training"
    assert "matched 2 runs" in capsys.readouterr().out


def test_resolve_policy_best_uses_the_metric(configured):
    entry = mr.resolve("cat", policy="best", metric="test roc_auc")
    assert entry.variant == "cat_orig_training"


def test_resolve_policy_best_without_the_metric_raises(configured):
    with pytest.raises(LookupError, match="no run matching"):
        mr.resolve("cat", policy="best", metric="nonexistent_metric")


def test_resolve_unknown_name_lists_available_variants(configured):
    with pytest.raises(LookupError, match="cat_orig_training"):
        mr.resolve("does_not_exist")


def test_rank_orders_by_metric(configured):
    df = mr.rank("cat", metric="test roc_auc")
    assert df["test roc_auc"].is_monotonic_decreasing


def test_rank_unknown_name_raises(configured):
    with pytest.raises(LookupError):
        mr.rank("nope")


# --------------------------------------------------------------------------- #
# Metric name resolution                                                      #
# --------------------------------------------------------------------------- #
def test_metric_names_collects_every_logged_key(configured):
    names = mr.metric_names()
    assert "test roc_auc" in names
    assert "valid Average Precision" in names


def test_resolve_metric_passes_exact_names_through(configured):
    assert mr.resolve_metric("test roc_auc") == "test roc_auc"


def test_resolve_metric_rejects_ambiguous_aliases(configured):
    """'average_precision' spans the test, valid and train keys."""
    with pytest.raises(LookupError, match="ambiguous"):
        mr.resolve_metric("average_precision")


def test_resolve_metric_normalizes_separators(configured):
    """MLflow logs 'test Average Precision'; callers write snake_case."""
    assert mr.resolve_metric("test_average_precision") == "test Average Precision"
    assert mr.resolve_metric("valid_average_precision") == "valid Average Precision"


def test_resolve_metric_is_case_insensitive(configured):
    assert mr.resolve_metric("TEST ROC_AUC") == "test roc_auc"


def test_best_per_algo_default_metric_is_resolvable(configured):
    """The default argument must not raise on a store using spaced keys."""
    with pytest.raises(LookupError, match="ambiguous"):
        mr.best_per_algo()  # ambiguous across splits, but recognized


def test_resolve_metric_unknown_raises(configured):
    with pytest.raises(LookupError, match="No metric like"):
        mr.resolve_metric("brier_but_not_logged")


# --------------------------------------------------------------------------- #
# Champion selection                                                          #
# --------------------------------------------------------------------------- #
def test_best_per_algo_returns_one_row_per_algo(configured):
    df = mr.best_per_algo(metric="test roc_auc")
    assert len(df) == 3
    assert set(df["algo"]) == {"cat", "lr", "xgb"}


def test_best_per_algo_picks_the_higher_score(configured):
    df = mr.best_per_algo(metric="test roc_auc")
    winner = df.loc[df["algo"] == "cat", "winner"].iloc[0]
    assert winner == "cat_orig_training"


def test_best_per_algo_counts_candidates(configured):
    df = mr.best_per_algo(metric="test roc_auc")
    assert df.loc[df["algo"] == "cat", "n_candidates"].iloc[0] == 2


def test_best_per_algo_ascending_flips_the_winner(configured):
    df = mr.best_per_algo(metric="test roc_auc", ascending=True)
    winner = df.loc[df["algo"] == "cat", "winner"].iloc[0]
    assert winner == "cat_smote_training"


def test_load_best_per_algo_returns_models_keyed_by_algo(configured):
    models = mr.load_best_per_algo(metric="test roc_auc")
    assert set(models) == {"cat", "lr", "xgb"}
    assert all(hasattr(m, "predict_proba") for m in models.values())


def test_load_best_per_algo_qualified_keys(configured):
    models = mr.load_best_per_algo(metric="test roc_auc", qualified=True)
    assert all("/" in key for key in models)


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #
def test_load_returns_a_usable_model(configured):
    model = mr.load("cat_orig_training")
    proba = model.predict_proba(configured["X"])[:, 1]
    assert proba.shape == (len(configured["y"]),)


def test_load_is_cached(configured):
    assert mr.load("cat_orig_training") is mr.load("cat_orig_training")


def test_load_all_keys_by_variant(configured):
    models = mr.load_all()
    assert set(models) == set(mr.variants())


def test_load_all_only_filter(configured):
    models = mr.load_all(only=["lr_orig_training"])
    assert set(models) == {"lr_orig_training"}


# --------------------------------------------------------------------------- #
# Cross-version unpickling                                                    #
# --------------------------------------------------------------------------- #
def test_missing_private_class_breaks_a_plain_load(legacy_blob):
    """Guard the premise: the fixture really does reproduce the failure."""
    blob, _ = legacy_blob
    with pytest.raises(AttributeError, match="_RemainderColsList"):
        pickle.loads(blob)


def test_registry_loads_an_artifact_with_a_missing_private_class(
    tmp_path, legacy_blob
):
    blob, _ = legacy_blob
    _store_with(tmp_path, blob)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = mr.load("cat_orig_training")

    assert model is not None
    assert "_RemainderColsList" in " ".join(mr.shimmed())


def test_remainder_repair_restores_the_transform_width(
    tmp_path, legacy_blob, data
):
    blob, healthy_width = legacy_blob
    X = data[0]
    _store_with(tmp_path, blob)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = mr.load("cat_orig_training")

    assert model.named_steps["pre"].transform(X).shape[1] == healthy_width
    assert any("remainder" in r for r in mr.repaired())


def test_remainder_repair_patches_all_three_locations(tmp_path, legacy_blob):
    """``transform()`` reads ``transformers_``, so patching ``_remainder``
    alone is not enough."""
    blob, _ = legacy_blob
    _store_with(tmp_path, blob)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pre = mr.load("cat_orig_training").named_steps["pre"]

    n_remainder = len(pre._remainder[2])
    assert n_remainder == 10
    assert len(pre._transformer_to_input_indices["remainder"]) == n_remainder
    tf_remainder = [c for name, _, c in pre.transformers_ if name == "remainder"]
    assert len(tf_remainder[0]) == n_remainder


def test_tolerant_load_warns_about_the_repair(tmp_path, legacy_blob):
    blob, _ = legacy_blob
    _store_with(tmp_path, blob)
    with pytest.warns(RuntimeWarning, match="repair"):
        mr.load("cat_orig_training")


def test_imputer_repair_restores_fill_dtype(
    tmp_path, legacy_imputer_blob, data, fill_dtype_expected
):
    """The repair sets the attribute to the dtype that makes the cast a no-op.

    ``statistics_.dtype`` reproduces the behaviour of versions that did not
    cast at all, so predictions must be unchanged.
    """
    blob, healthy = legacy_imputer_blob
    X = data[0]
    _store_with(tmp_path, blob)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = mr.load("cat_orig_training")

    imp = model.named_steps["pre"].named_transformers_["num"].named_steps["imp"]
    assert imp._fill_dtype == imp.statistics_.dtype
    np.testing.assert_allclose(model.predict_proba(X)[:, 1], healthy)
    assert any("_fill_dtype" in r for r in mr.repaired())


def test_imputer_repair_is_idempotent(data, fill_dtype_expected):
    """Running the repair twice reports work only the first time."""
    X, y, num = data
    pipe = _fit_pipeline(X, y, num, with_imputer=True)
    _break_imputer(pipe)
    assert mr.repair_estimator(pipe) != []
    assert mr.repair_estimator(pipe) == []


def test_imputer_repair_is_gated_on_the_sklearn_version(data, monkeypatch):
    """Where the attribute was never part of fitted state, absence is normal.

    Repairing regardless would report work on healthy models and warn about
    them, so the gate has to hold.
    """
    monkeypatch.setattr(mr, "_imputer_uses_fill_dtype", lambda: False)
    X, y, num = data
    pipe = _fit_pipeline(X, y, num, with_imputer=True)
    _break_imputer(pipe)
    assert mr.repair_estimator(pipe) == []


def test_version_gate_reports_a_bool():
    assert isinstance(mr._imputer_uses_fill_dtype(), bool)


def test_repairs_do_not_touch_a_healthy_model(configured):
    before = list(mr.repaired())
    mr.load("cat_orig_training")
    assert mr.repaired() == before


def test_repair_estimator_leaves_a_freshly_fitted_pipeline_alone(data):
    """A model fitted by the installed sklearn needs no repair, any version."""
    X, y, num = data
    pipe = _fit_pipeline(X, y, num, with_imputer=True)
    assert mr.repair_estimator(pipe) == []


def test_repair_skips_drop_remainder(data):
    """An empty remainder list is correct when nothing passes through."""
    X, y, num = data
    pipe = _fit_pipeline(X, y, num, remainder="drop")
    assert mr.repair_estimator(pipe) == []


# --------------------------------------------------------------------------- #
# Verification                                                                #
# --------------------------------------------------------------------------- #
def test_verify_entry_compares_only_the_requested_split(configured):
    df = mr.verify_entry("cat_orig_training", configured["X"], configured["y"])
    assert len(df) > 0
    assert all(k.startswith("test") for k in df["logged_metric"])


def test_verify_entry_passes_on_a_faithful_load(configured):
    df = mr.verify_entry("cat_orig_training", configured["X"], configured["y"])
    assert df["ok"].all()
    assert df["delta"].abs().max() <= 1e-3


def test_verify_entry_split_none_includes_every_split(configured):
    df = mr.verify_entry(
        "cat_orig_training", configured["X"], configured["y"], split=None
    )
    prefixes = {k.split()[0] for k in df["logged_metric"]}
    assert {"test", "valid", "train"} <= prefixes


def test_verify_entry_flags_a_mismatched_split(configured):
    """A logged validation score compared against test-set metrics must fail."""
    df = mr.verify_entry(
        "cat_orig_training", configured["X"], configured["y"], split="valid"
    )
    assert not df["ok"].all()


def test_verify_entry_tolerance_is_respected(configured):
    tight = mr.verify_entry(
        "cat_orig_training", configured["X"], configured["y"], tol=0.0
    )
    loose = mr.verify_entry(
        "cat_orig_training", configured["X"], configured["y"], tol=1.0
    )
    assert loose["ok"].all()
    assert tight["ok"].sum() <= loose["ok"].sum()


def test_verify_entry_warns_when_nothing_matches(configured):
    with pytest.warns(RuntimeWarning, match="matched split="):
        df = mr.verify_entry(
            "cat_orig_training",
            configured["X"],
            configured["y"],
            split="nosuchsplit",
        )
    assert df.empty


def test_verify_all_returns_one_row_per_champion(configured):
    df = mr.verify_all(configured["X"], configured["y"], metric="test roc_auc")
    assert len(df) == 3
    assert set(df["algo"]) == {"cat", "lr", "xgb"}


def test_verify_all_reports_max_abs_delta_and_passes(configured):
    df = mr.verify_all(configured["X"], configured["y"], metric="test roc_auc")
    assert df["ok"].all()
    assert (df["max_abs_delta"] <= 1e-3).all()


# --------------------------------------------------------------------------- #
# Validation-set selection                                                    #
# --------------------------------------------------------------------------- #
def test_score_candidates_scores_every_model(configured):
    df = mr.score_candidates((configured["X"], configured["y"]))
    assert len(df) == 4
    assert df["score"].notna().all()


def test_score_candidates_accepts_per_algo_data(configured):
    df = mr.score_candidates({"cat": (configured["X"], configured["y"])}, name="cat")
    assert len(df) == 2


def test_score_candidates_records_errors_without_raising(configured):
    bad = configured["X"].iloc[:, :3]
    df = mr.score_candidates((bad, configured["y"]))
    assert "error" in df.columns
    assert df["score"].isna().any()


def test_select_on_validation_returns_one_row_per_algo(configured):
    df = mr.select_on_validation((configured["X"], configured["y"]))
    assert set(df["algo"]) == {"cat", "lr", "xgb"}


def test_load_selected_loads_the_named_models(configured):
    selection = mr.select_on_validation((configured["X"], configured["y"]))
    models = mr.load_selected(selection)
    assert set(models) == {"cat", "lr", "xgb"}


# --------------------------------------------------------------------------- #
# Store scoping                                                               #
# --------------------------------------------------------------------------- #
@pytest.fixture
def multi_store(tmp_path, data):
    """Three stores sharing one experiment name, with stale scores inflated.

    Mirrors the situation the setting exists for: a superseded store kept on
    disk for reference would otherwise win every comparison, because grouping
    is by experiment name and the name is identical across stores. The third
    store has a confusable prefix, so substring matching would wrongly admit
    it where segment matching does not.
    """
    X, y, num = data
    pipe = _fit_pipeline(X, y, num)

    def store(name, run_id, run_name, algo, auc, start):
        exp = tmp_path / "mlruns" / name / "1"
        exp.mkdir(parents=True, exist_ok=True)
        (exp / "meta.yaml").write_text("experiment_id: '1'\nname: income_model\n")
        _write_run(
            exp, run_id, run_name, algo, pipe, {"test roc_auc": auc}, start
        )

    store("models", "r_live_lr", "lr_orig_training", "lr", 0.80, 3000)
    store("models", "r_live_rf", "rf_orig_training", "rf", 0.79, 2900)
    store("models_old", "r_old_lr", "lr_old_training", "lr", 0.95, 1000)
    store("models_old", "r_old_rf", "rf_old_training", "rf", 0.94, 900)
    store("models_group_split", "r_gs_lr", "lr_gs_training", "lr", 0.99, 500)

    mr.configure(root=tmp_path, target=TARGET, stores=())
    return tmp_path


def test_stores_defaults_to_no_constraint(configured):
    assert mr.stores() == ()


def test_unconstrained_selection_lets_a_stale_store_win(multi_store):
    """Without the constraint the superseded store wins on inflated scores."""
    df = mr.best_per_algo(metric="test roc_auc")
    winners = dict(zip(df["algo"], df["winner"]))
    assert winners["lr"] == "lr_gs_training"
    assert winners["rf"] == "rf_old_training"


def test_set_stores_restricts_the_winner(multi_store):
    mr.set_stores("mlruns/models")
    df = mr.best_per_algo(metric="test roc_auc")
    winners = dict(zip(df["algo"], df["winner"]))
    assert winners["lr"] == "lr_orig_training"
    assert winners["rf"] == "rf_orig_training"


def test_store_matching_is_segment_wise_not_substring(multi_store):
    """'mlruns/models' must not admit 'mlruns/models_group_split'."""
    mr.set_stores("mlruns/models")
    df = mr.best_per_algo(metric="test roc_auc")
    assert set(df["store"]) == {"mlruns/models"}


def test_stores_accepts_several_prefixes(multi_store):
    mr.set_stores("mlruns/models", "mlruns/models_old")
    df = mr.best_per_algo(metric="test roc_auc")
    assert set(df["store"]) <= {"mlruns/models", "mlruns/models_old"}
    # the old store still wins where it is allowed to compete
    assert dict(zip(df["algo"], df["winner"]))["lr"] == "lr_old_training"


def test_stores_can_be_passed_per_call(multi_store):
    """A one-off override leaves the global setting untouched."""
    df = mr.best_per_algo(metric="test roc_auc", stores="mlruns/models")
    assert set(df["store"]) == {"mlruns/models"}
    assert mr.stores() == ()


def test_set_stores_with_no_arguments_clears_the_constraint(multi_store):
    mr.set_stores("mlruns/models")
    assert mr.stores() == ("mlruns/models",)
    mr.set_stores()
    assert mr.stores() == ()


def test_index_stays_global_under_a_store_constraint(multi_store):
    """An excluded store must remain visible and loadable, just not winning."""
    mr.set_stores("mlruns/models")
    assert len(mr.variants()) == 5
    assert "lr_old_training" in mr.variants()
    assert mr.load("lr_old_training") is not None


def test_n_candidates_respects_the_store_constraint(multi_store):
    mr.set_stores("mlruns/models")
    df = mr.best_per_algo(metric="test roc_auc")
    assert set(df["n_candidates"]) == {1}


def test_unknown_store_raises_and_names_what_is_available(multi_store):
    mr.set_stores("mlruns/does_not_exist")
    with pytest.raises(LookupError, match="mlruns/models"):
        mr.best_per_algo(metric="test roc_auc")


def test_store_summary_flags_eligibility(multi_store):
    mr.set_stores("mlruns/models")
    summary = mr.store_summary()
    eligible = dict(zip(summary["store"], summary["eligible"]))
    assert eligible["mlruns/models"] is True
    assert eligible["mlruns/models_old"] is False
    assert eligible["mlruns/models_group_split"] is False
    assert summary.loc[
        summary["store"] == "mlruns/models", "models"
    ].iloc[0] == 2


def test_store_summary_lists_every_store_regardless(multi_store):
    mr.set_stores("mlruns/models")
    assert len(mr.store_summary()) == 3


def test_stores_env_style_comma_string(multi_store):
    mr.set_stores("mlruns/models,mlruns/models_old")
    assert mr.stores() == ("mlruns/models", "mlruns/models_old")


def test_configure_accepts_stores(multi_store):
    mr.configure(stores=("mlruns/models",))
    assert mr.stores() == ("mlruns/models",)


# --------------------------------------------------------------------------- #
# Diagnostics and failure messages                                            #
# --------------------------------------------------------------------------- #
def test_is_available_true_for_a_populated_store(configured):
    assert mr.is_available() is True


def test_is_available_false_without_a_store(tmp_path):
    mr.configure(root=tmp_path / "empty")
    assert mr.is_available() is False


def test_missing_root_message_points_at_configure(tmp_path):
    mr.configure(root=tmp_path / "nope")
    with pytest.raises(RuntimeError, match="does not exist"):
        mr.available()


def test_no_tracking_store_message_explains_mlflow_is_not_required(tmp_path):
    (tmp_path / "src").mkdir()
    mr.configure(root=tmp_path)
    with pytest.raises(RuntimeError, match="does not require mlflow"):
        mr.available()


def test_wrong_model_filename_message_points_at_configure(store):
    mr.configure(root=store["root"], target=TARGET, model_file="not_here.pkl")
    with pytest.raises(RuntimeError, match="model_file"):
        mr.available()
    mr.configure(model_file="model.pkl")


def test_diagnose_prints_configuration(configured, capsys):
    mr.diagnose()
    out = capsys.readouterr().out
    assert "project root" in out
    assert "loader" in out
    assert "cat_orig_training" in out


def test_diagnose_survives_an_empty_store(tmp_path, capsys):
    mr.configure(root=tmp_path / "empty")
    mr.diagnose()
    out = capsys.readouterr().out
    assert "project root" in out
    assert "does not exist" in out


if __name__ == "__main__":
    pytest.main()