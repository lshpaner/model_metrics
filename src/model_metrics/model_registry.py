"""
model_registry.py
=================

MLflow-backed model loading. No hard-coded experiment ids, run ids, or paths.

Two things this handles that plain MLflow does not:

  * Several tracking roots may live under one project (mlruns/preprocessing,
    mlruns/models, ...). All of them are indexed, not just one.
  * The recorded artifact_location may point into a different directory than
    the one the tree currently lives in, because mlruns/ trees get copied
    between machines and projects. Every MLflow artifact API will therefore
    resolve to a path that does not exist locally. Artifacts are instead
    rebased onto the store root actually found on disk.

Configuration
-------------
Two settings, both optional, both overridable by environment variable or by
calling ``configure()`` before first use:

    MODEL_REGISTRY_ROOT     directory to walk (default: the working directory,
                            walking up to the repository root to find an
                            mlruns/ tree)
    MODEL_REGISTRY_TARGET   outcome token to strip from names (default: none)
    MODEL_REGISTRY_STORES   comma-separated store prefixes eligible to win a
                            metric comparison (default: none, no constraint)

The target token exists because artifact folders are conventionally named
``<algo>_<TARGET>``. Setting it to your outcome name turns ``cat_outcome``
into ``cat`` so you group by algorithm rather than by outcome. The token is
stripped wherever it appears, so ``cat_outcome_no_sex`` becomes ``cat_no_sex``
and ablation variants stay distinguishable.

    export MODEL_REGISTRY_TARGET=outcome

or, in code:

    from core import model_registry
    model_registry.configure(target="outcome")

Naming
------
Several runs may share one artifact folder name (multiple runs all writing to
``cat_<TARGET>/``). So the folder gives the ALGO and the run name gives the
VARIANT:

    algo    : cat, lr, xgb, rf, ...
    variant : cat_orig, cat_smote, cat_orig_no_sex, ...   (the MLflow run name)

``variant`` is the addressable key. ``algo`` is what you group by.

Stores
------
The INDEX is always global: available(), rank(), load(), load_all() and
variants() see every model under PROJECT_ROOT, so nothing is hidden.

STORES constrains only the metric-based selectors, best_per_algo() and
load_best_per_algo(), so a superseded store cannot win a comparison it was
never meant to be in. Prefixes are project-relative and matched on WHOLE path
segments: "mlruns/models" matches "mlruns/models/12345" but NOT
"mlruns/models_old/12345".

Default is empty, meaning no constraint. Set it either way:

    export MODEL_REGISTRY_STORES=mlruns/models        # before import
    set_stores("mlruns/models")                       # at runtime
    set_stores()                                      # clear it

    best_per_algo(stores="mlruns/models")             # one call only

store_summary() shows every store on disk with its model count and whether it
is currently eligible, so you can see what is being excluded before trusting a
result.

Selection
---------
Default policy is "newest" (latest start_time wins a tie). Selecting by best
test metric is available but OFF by default, because picking the max-metric run
IS model selection on the evaluation set. Use it to inspect, pin by name to load.

Usage
-----
    from core.model_registry import available, load, load_all, rank

    available()                      # everything on disk, with metrics
    rank("cat", metric="roc_auc")    # see every cat run, ordered
    model = load("cat_smote")        # pin the variant you actually reported
    models = load_all()              # one model per variant
"""

from __future__ import annotations

import os
import pickle
import re
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
# model_tuner is optional. If it is installed we use its loader so that
# anything it does at save time is mirrored at load time; otherwise we fall
# back to joblib (which reads plain pickles too), then to pickle itself.
try:  # pragma: no cover - trivial
    from model_tuner import loadObjects as _mt_load

    _BACKEND = "model_tuner"
except ImportError:  # pragma: no cover - trivial
    _mt_load = None
    try:
        import joblib as _joblib

        _BACKEND = "joblib"
    except ImportError:
        _joblib = None
        _BACKEND = "pickle"


def _default_loader(path: str):
    """Read one serialized artifact using the best backend available."""
    if _mt_load is not None:
        return _mt_load(path)
    if _BACKEND == "joblib":
        return _joblib.load(path)
    with open(path, "rb") as fh:
        return pickle.load(fh)


# Override with configure(loader=...) if your artifacts need special handling.
LOADER = _default_loader


def backend() -> str:
    """Which deserializer is in use: 'model_tuner', 'joblib', or 'pickle'."""
    return _BACKEND

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def _default_root() -> Path:
    """Best guess at where an mlruns/ tree lives.

    The module's own location is useless as a search root: installed in
    site-packages it is nowhere near the user's project, and in a src/ layout
    it resolves to src/ rather than the repository. So the search starts at the
    working directory and walks up until an 'mlruns' directory appears, which
    also lets the registry work when called from a notebooks/ subdirectory.

    The walk stops at the repository boundary, so an unrelated mlruns/ higher
    up the filesystem is never picked up silently. Falls back to the working
    directory, which produces a clear error rather than a wrong answer.
    """
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "mlruns").is_dir():
            return candidate
        if (candidate / ".git").exists():
            break  # do not search above the repository root
    return cwd


PROJECT_ROOT = Path(os.environ.get("MODEL_REGISTRY_ROOT", "") or _default_root())

# Outcome token embedded in artifact folder names ("<algo>_<TARGET>").
# Empty means no stripping, which is the safe default for an unknown project.
TARGET = os.environ.get("MODEL_REGISTRY_TARGET", "")

MODEL_FILE = os.environ.get("MODEL_REGISTRY_FILE", "model.pkl")


def _parse_stores(value) -> tuple:
    """
    Normalize a stores setting into a tuple of POSIX-style prefixes.

    Accepts a single string, a comma-separated string, or any iterable of
    either, so that the environment variable form and the call form behave
    identically:

        "mlruns/models"
        "mlruns/models,mlruns/archive"
        ("mlruns/models", "mlruns/archive")
        ("mlruns/models,mlruns/archive",)
    """
    if not value:
        return ()
    parts = [value] if isinstance(value, str) else list(value)

    out = []
    for part in parts:
        for piece in str(part).split(","):
            piece = piece.strip().replace("\\", "/").strip("/")
            if piece:
                out.append(piece)
    return tuple(out)


# Restrict which tracking stores may WIN a metric comparison.
#
# The index is always global: available(), rank(), load(), load_all() and
# variants() see every model on disk, so nothing is hidden. STORES constrains
# only best_per_algo() and load_best_per_algo(), so a superseded store such as
# mlruns/models_old cannot win a comparison it was never meant to be in.
#
# Prefixes are project-relative and matched on WHOLE path segments, so
# "mlruns/models" matches "mlruns/models/12345" but not
# "mlruns/models_old/12345". Empty means no constraint.
STORES = _parse_stores(os.environ.get("MODEL_REGISTRY_STORES", ""))

# Directory names never worth walking into.
SKIP_DIRS = {".git", "node_modules", ".ipynb_checkpoints", ".venv", "__pycache__"}


def configure(
    root: Optional[str | Path] = None,
    target: Optional[str] = None,
    model_file: Optional[str] = None,
    loader=None,
    stores=None,
) -> None:
    """
    Override configuration at runtime and drop the cached index.

    Call before the first lookup, or after moving/regenerating mlruns/.

        model_registry.configure(target="outcome")
    """
    global PROJECT_ROOT, TARGET, MODEL_FILE, LOADER, STORES
    if root is not None:
        PROJECT_ROOT = Path(root)
    if target is not None:
        TARGET = target
    if model_file is not None:
        MODEL_FILE = model_file
    if loader is not None:
        LOADER = loader
    if stores is not None:
        STORES = _parse_stores(stores)
    refresh()


def set_stores(*prefixes) -> tuple:
    """
    Restrict which stores may win a metric comparison. Returns the new value.

        set_stores("mlruns/models")                    # only the live store
        set_stores("mlruns/models", "mlruns/archive")  # two of them
        set_stores()                                   # clear the constraint
    """
    global STORES
    if len(prefixes) == 1 and not isinstance(prefixes[0], str):
        STORES = _parse_stores(prefixes[0])
    else:
        STORES = _parse_stores(prefixes)
    return STORES


def stores() -> tuple:
    """The store prefixes currently constraining metric-based selection."""
    return STORES


def refresh() -> None:
    """Drop the cached filesystem index and the model load cache."""
    _index.cache_clear()
    load.cache_clear()


# ---------------------------------------------------------------------------
# tolerant unpickling
# ---------------------------------------------------------------------------
# Artifacts are often written under a different scikit-learn than the one
# installed. Pickles reference private sklearn classes by module path, and
# those get renamed or removed between versions, so a plain load dies with
#
#     AttributeError: Can't get attribute '_RemainderColsList'
#     on <module 'sklearn.compose._column_transformer'>
#
# Every load in this module goes through _load_model(), which retries with
# synthesized stand-ins for missing sklearn internals. Cross-version loading
# is not officially supported; see verify_entry() for the check that the
# loaded object still reproduces its logged metrics.

# (module path, attribute) -> base class to synthesize.
# These are pure data containers, so a bare subclass restores state exactly.
SKLEARN_SHIMS = {
    ("sklearn.compose._column_transformer", "_RemainderColsList"): list,
    ("sklearn.utils._bunch", "Bunch"): dict,
    ("sklearn.utils", "Bunch"): dict,
}

# Set False to make version mismatches raise instead of warn.
TOLERANT_LOAD = os.environ.get("MODEL_REGISTRY_TOLERANT", "1") != "0"

_shimmed: List[str] = []
_repaired: List[str] = []


def _install_shims() -> List[str]:
    """Create any missing private sklearn classes listed in SKLEARN_SHIMS."""
    import importlib

    patched = []
    for (mod_path, attr), base in SKLEARN_SHIMS.items():
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue
        if hasattr(mod, attr):
            continue
        setattr(mod, attr, type(attr, (base,), {"__module__": mod_path}))
        patched.append(f"{mod_path}.{attr}")
    return patched


class _TolerantUnpickler(pickle.Unpickler):
    """Synthesize missing sklearn classes rather than raising."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.synthesized: List[str] = []

    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (AttributeError, ImportError):
            if not module.startswith("sklearn"):
                raise
            self.synthesized.append(f"{module}.{name}")
            return type(
                name,
                (list,),
                {"__module__": module, "_registry_placeholder": True},
            )


def _load_model(path: Path):
    """
    Load one artifact. Tries the configured loader first, then falls back to
    a shimmed unpickle if the installed sklearn is missing private classes the
    pickle references.
    """
    try:
        return _post_load(LOADER(str(path)), path)
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        if not TOLERANT_LOAD:
            raise
        first = exc

    global _shimmed
    patched = _install_shims()
    if patched:
        _shimmed = sorted(set(_shimmed) | set(patched))

    # Retry the normal path now that the known shims exist.
    try:
        return _post_load(LOADER(str(path)), path)
    except (AttributeError, ImportError, ModuleNotFoundError):
        pass

    # Anything still missing gets synthesized on demand.
    with open(path, "rb") as fh:
        up = _TolerantUnpickler(fh)
        try:
            obj = up.load()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not load {path} even with sklearn shims.\n"
                f"Original error: {first}\n"
                f"Retry error: {exc}\n"
                "The training and runtime sklearn versions are too far apart. "
                "Check the run's artifacts/*/requirements.txt for the pin."
            ) from exc

    if up.synthesized:
        warnings.warn(
            f"{path.name}: synthesized {len(up.synthesized)} missing sklearn "
            f"class(es) {up.synthesized}. The model loaded but may not behave "
            "as it did at training time. Run verify_entry() before reporting "
            "anything derived from it.",
            RuntimeWarning,
            stacklevel=3,
        )
    return _post_load(obj, path)


def _post_load(obj, path: Path):
    """Apply fitted-state repairs to a freshly loaded artifact."""
    applied = repair_estimator(obj)
    if applied:
        global _repaired
        _repaired = sorted(set(_repaired) | set(applied))
        warnings.warn(
            f"{path.name}: applied {len(applied)} fitted-state repair(s) "
            f"{applied} for scikit-learn version drift. Run verify_entry() "
            "before reporting anything derived from this model.",
            RuntimeWarning,
            stacklevel=4,
        )
    return obj


def shimmed() -> List[str]:
    """sklearn internals this process had to synthesize, if any."""
    return list(_shimmed)


def repaired() -> List[str]:
    """Fitted-state attributes this process had to restore, if any."""
    return list(_repaired)


# ---------------------------------------------------------------------------
# post-load repair
# ---------------------------------------------------------------------------
# Shimming fixes *unpickling*. It does not fix *fitted state*: newer sklearn
# transform() code reads instance attributes that an older fit() never wrote,
# e.g.
#
#     AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'
#
# Each entry restores the attribute to the value that reproduces the ORIGINAL
# (training-version) behaviour, not the current version's behaviour. For
# _fill_dtype that means statistics_.dtype, which makes the new astype() call
# a no-op, matching versions that did not cast at all.
#
# This is a stopgap. The correct fix is to pin the training sklearn version.


@lru_cache(maxsize=1)
def _imputer_uses_fill_dtype() -> bool:
    """
    True if the installed scikit-learn's SimpleImputer records _fill_dtype.

    The attribute was introduced partway through the 1.x line. On versions
    that never had it, its absence in a fitted imputer is normal and no repair
    is warranted; repairing anyway would fill repaired() with noise and warn
    about perfectly healthy models.
    """
    try:
        from sklearn.impute import SimpleImputer

        probe = SimpleImputer(strategy="mean").fit([[1.0], [2.0]])
        return hasattr(probe, "_fill_dtype")
    except Exception:  # noqa: BLE001
        return False


def _repair_simple_imputer(obj) -> Optional[str]:
    if type(obj).__name__ != "SimpleImputer":
        return None
    if not _imputer_uses_fill_dtype():
        return None  # this sklearn never sets it; absence is not a defect
    if hasattr(obj, "_fill_dtype"):
        return None
    stats = getattr(obj, "statistics_", None)
    if stats is None or not hasattr(stats, "dtype"):
        return None
    obj._fill_dtype = stats.dtype
    return f"SimpleImputer._fill_dtype = {stats.dtype}"


def _repair_remainder_cols(obj) -> Optional[str]:
    """
    Restore a ColumnTransformer's remainder column list.

    When the pickle references sklearn's private _RemainderColsList and the
    installed sklearn no longer defines it, the synthesized stand-in unpickles
    as an EMPTY list. With remainder='passthrough' that silently drops every
    passthrough column: the transformer emits fewer features than the
    downstream estimator was fitted on, and you get

        ValueError: Feature shape mismatch, expected: 25, got 15

    The remainder set is fully determined by fitted state, so it is rebuilt
    exactly as sklearn computes it at fit time:

        sorted(set(range(n_features_in_)) - union(explicit transformer cols))

    Three places hold this list and all three must agree. transform() reads
    `transformers_` when fitted, so repairing `_remainder` alone is not
    enough.
    """
    if type(obj).__name__ != "ColumnTransformer":
        return None

    remainder_kind = getattr(obj, "remainder", None)
    if remainder_kind == "drop" or remainder_kind is None:
        return None  # nothing passes through; empty lists are correct

    n_in = getattr(obj, "n_features_in_", None)
    mapping = getattr(obj, "_transformer_to_input_indices", None)
    if n_in is None or not mapping:
        return None

    consumed = set()
    for name, idx in mapping.items():
        if name == "remainder":
            continue
        try:
            consumed.update(int(i) for i in idx)
        except TypeError:
            return None

    remaining = sorted(set(range(int(n_in))) - consumed)
    if not remaining:
        return None  # genuinely nothing left over

    fixed = []

    # 1. the index mapping
    if len(mapping.get("remainder", [])) != len(remaining):
        mapping["remainder"] = list(remaining)
        fixed.append("_transformer_to_input_indices")

    # 2. the private tuple used by the unfitted _iter path
    rem = getattr(obj, "_remainder", None)
    if isinstance(rem, tuple) and len(rem) == 3 and len(rem[2]) != len(remaining):
        obj._remainder = (rem[0], rem[1], list(remaining))
        fixed.append("_remainder")

    # 3. transformers_, which is what transform() actually reads when fitted
    tf = getattr(obj, "transformers_", None)
    if isinstance(tf, list):
        rebuilt, found = [], False
        for entry in tf:
            if (
                isinstance(entry, tuple)
                and len(entry) == 3
                and entry[0] == "remainder"
            ):
                found = True
                if len(entry[2]) != len(remaining):
                    rebuilt.append((entry[0], entry[1], list(remaining)))
                    fixed.append("transformers_")
                else:
                    rebuilt.append(entry)
            else:
                rebuilt.append(entry)
        if not found:
            rebuilt.append(("remainder", remainder_kind, list(remaining)))
            fixed.append("transformers_[appended]")
        if fixed:
            obj.transformers_ = rebuilt

    if not fixed:
        return None

    return (
        f"ColumnTransformer restored {len(remaining)} '{remainder_kind}' "
        f"remainder column(s) in {', '.join(sorted(set(fixed)))}"
    )


REPAIRS = (_repair_simple_imputer, _repair_remainder_cols)


def _walk(obj, seen=None):
    """Yield every estimator-ish object reachable from a fitted pipeline."""
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return
    seen.add(id(obj))
    yield obj

    for attr in ("steps", "transformers", "transformers_", "named_steps",
                 "estimators", "estimators_", "_final_estimator", "estimator",
                 "base_estimator", "calibrated_classifiers_"):
        v = getattr(obj, attr, None)
        if v is None:
            continue
        if isinstance(v, dict):
            v = list(v.values())
        if isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, (list, tuple)):
                    for sub in item:
                        yield from _walk(sub, seen)
                else:
                    yield from _walk(item, seen)
        else:
            yield from _walk(v, seen)

    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        for k, item in d.items():
            if k.startswith("__"):
                continue
            if hasattr(item, "__dict__") or isinstance(item, (list, tuple)):
                if isinstance(item, (list, tuple)):
                    for sub in item:
                        if hasattr(sub, "__dict__"):
                            yield from _walk(sub, seen)
                else:
                    yield from _walk(item, seen)


def repair_estimator(model, verbose: bool = False) -> List[str]:
    """
    Patch fitted attributes the installed sklearn expects but the pickle
    predates. Returns a description of each repair applied.
    """
    applied = []
    for node in _walk(model):
        for fn in REPAIRS:
            try:
                msg = fn(node)
            except Exception:  # noqa: BLE001
                continue
            if msg:
                applied.append(msg)
    if applied and verbose:
        print(f"[model_registry] repaired: {applied}")
    return applied


@dataclass(frozen=True)
class ModelEntry:
    algo: str  # from the artifact folder: cat, lr, ...
    variant: str  # from the run name: cat_orig, cat_smote, ...
    path: Path  # local, verified to exist
    run_id: str
    experiment_id: str
    experiment_name: str
    store_root: Path
    start_time: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.experiment_name}/{self.variant}"


# ---------------------------------------------------------------------------
# filesystem walk
# ---------------------------------------------------------------------------


# PyYAML is optional. MLflow's meta.yaml files are flat scalar key/value
# documents, so a minimal parser covers every field this module reads
# (experiment_id, name, run_id, run_uuid, run_name, lifecycle_stage,
# start_time). PyYAML is used when present, for robustness against future
# format changes.
try:  # pragma: no cover - trivial
    import yaml as _yaml
except ImportError:  # pragma: no cover - trivial
    _yaml = None


def _parse_flat_yaml(text: str) -> dict:
    """Parse a flat 'key: value' document without PyYAML."""
    out: dict = {}
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#") or line.startswith((" ", "\t")):
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if not key:
            continue

        if val in ("", "null", "~"):
            out[key] = "" if val == "" else None
            continue
        if val in ("[]", "{}"):
            out[key] = [] if val == "[]" else {}
            continue
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            out[key] = val[1:-1]
            continue
        if val in ("true", "True"):
            out[key] = True
            continue
        if val in ("false", "False"):
            out[key] = False
            continue
        try:
            out[key] = int(val)
            continue
        except ValueError:
            pass
        try:
            out[key] = float(val)
            continue
        except ValueError:
            pass
        out[key] = val
    return out


def _read_yaml(path: Path) -> dict:
    try:
        text = path.read_text()
    except Exception:  # noqa: BLE001
        return {}
    if _yaml is not None:
        try:
            loaded = _yaml.safe_load(text)
            if isinstance(loaded, dict):
                return loaded
            return {}
        except Exception:  # noqa: BLE001
            pass
    try:
        return _parse_flat_yaml(text)
    except Exception:  # noqa: BLE001
        return {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text().strip()
    except Exception:  # noqa: BLE001
        return ""


def _read_metrics(run_dir: Path) -> Dict[str, float]:
    """FileStore writes one file per metric: '<timestamp> <value> <step>'."""
    out: Dict[str, float] = {}
    mdir = run_dir / "metrics"
    if not mdir.is_dir():
        return out
    for f in mdir.iterdir():
        if not f.is_file():
            continue
        lines = [ln for ln in _read_text(f).splitlines() if ln.strip()]
        if not lines:
            continue
        parts = lines[-1].split()
        if len(parts) >= 2:
            try:
                out[f.name] = float(parts[1])
            except ValueError:
                pass
    return out


def _strip_target(name: str) -> str:
    """
    Remove the outcome token from a folder or run name.

    Strips the token wherever it appears, not only as a suffix, so that
    ablation and sub-variant names survive intact:

        cat_outcome          -> cat
        cat_outcome_no_sex   -> cat_no_sex
    """
    if not TARGET:
        return name
    token = f"_{TARGET}"
    if name.endswith(token):
        return name[: -len(token)]
    if token + "_" in name:
        return name.replace(token + "_", "_", 1)
    if name == TARGET:
        return name
    return name


def _find_models_in_run(run_dir: Path) -> List[tuple]:
    """[(algo, path)] for every model file this run wrote, on the LOCAL disk."""
    artifacts = run_dir / "artifacts"
    if not artifacts.is_dir():
        return []

    found = []
    direct = artifacts / MODEL_FILE
    if direct.is_file():
        found.append((_strip_target(run_dir.name), direct))

    for sub in sorted(p for p in artifacts.iterdir() if p.is_dir()):
        pkl = sub / MODEL_FILE
        if pkl.is_file():
            found.append((_strip_target(sub.name), pkl))
    return found


def _no_models_message() -> str:
    """Explain *why* nothing was found, rather than just that nothing was."""
    if not PROJECT_ROOT.exists():
        return (
            f"Search root does not exist: {PROJECT_ROOT}\n"
            "Set it with configure(root=...) or the MODEL_REGISTRY_ROOT "
            "environment variable."
        )

    stores = [
        p for p in PROJECT_ROOT.rglob("meta.yaml")
        if not any(s in p.parts for s in SKIP_DIRS)
    ]
    if not stores:
        return (
            f"No MLflow tracking store found under {PROJECT_ROOT}.\n"
            "This module reads a local MLflow FileStore directory (an "
            "'mlruns/' tree of meta.yaml files) directly from disk. It does "
            "not require mlflow to be installed, but something must have "
            "written that tree first.\n"
            "The search root defaults to the working directory (walking up to "
            "the repository root to find an mlruns/ tree). Set it explicitly "
            "with configure(root=...) or MODEL_REGISTRY_ROOT.\n"
            "If your runs live on a remote tracking server or a SQL backend, "
            "this module cannot read them; use the mlflow client instead."
        )

    return (
        f"Found {len(stores)} MLflow metadata file(s) under {PROJECT_ROOT}, "
        f"but no '{MODEL_FILE}' artifact in any run.\n"
        "If your models are saved under a different filename, set it with "
        "configure(model_file=...) or MODEL_REGISTRY_FILE."
    )


def is_available() -> bool:
    """
    True if at least one loadable model was found, without raising.

    Use this to gate registry-dependent code paths in environments that may
    not have an MLflow store at all.
    """
    try:
        return bool(_index())
    except RuntimeError:
        return False


@lru_cache(maxsize=1)
def _index() -> List[ModelEntry]:
    """Index every model file in every MLflow store under the project root."""
    entries: List[ModelEntry] = []

    for exp_meta in PROJECT_ROOT.rglob("meta.yaml"):
        if any(p in SKIP_DIRS for p in exp_meta.parts):
            continue

        meta = _read_yaml(exp_meta)
        if "experiment_id" not in meta or "run_id" in meta:
            continue  # this is a run, not an experiment

        exp_dir = exp_meta.parent
        exp_id = str(meta["experiment_id"])
        exp_name = str(meta.get("name") or exp_id)

        for run_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
            run_meta = _read_yaml(run_dir / "meta.yaml")
            if not run_meta:
                continue
            if str(run_meta.get("lifecycle_stage", "active")) == "deleted":
                continue

            models = _find_models_in_run(run_dir)
            if not models:
                continue

            run_id = str(
                run_meta.get("run_id") or run_meta.get("run_uuid") or run_dir.name
            )
            run_name = (
                _read_text(run_dir / "tags" / "mlflow.runName")
                or str(run_meta.get("run_name") or "")
                or run_id[:8]
            )
            metrics = _read_metrics(run_dir)

            for algo, path in models:
                entries.append(
                    ModelEntry(
                        algo=algo,
                        variant=_strip_target(run_name),
                        path=path,
                        run_id=run_id,
                        experiment_id=exp_id,
                        experiment_name=exp_name,
                        store_root=exp_dir.parent,
                        start_time=int(run_meta.get("start_time") or 0),
                        metrics=metrics,
                    )
                )

    if not entries:
        raise RuntimeError(_no_models_message())

    entries.sort(key=lambda e: e.start_time, reverse=True)
    return entries


# ---------------------------------------------------------------------------
# lookup
# ---------------------------------------------------------------------------


def _rel_store(entry: "ModelEntry") -> str:
    """The entry's store root as a project-relative POSIX path."""
    try:
        return entry.store_root.resolve().relative_to(
            PROJECT_ROOT.resolve()
        ).as_posix()
    except ValueError:
        return entry.store_root.as_posix()


def _in_stores(entry: "ModelEntry", prefixes: tuple = None) -> bool:
    """
    True if the entry's store sits under one of the given prefixes.

    Matching is on whole path segments, so "mlruns/models" accepts
    "mlruns/models" and "mlruns/models/12345" but rejects
    "mlruns/models_old/12345". A substring test would wrongly accept the
    latter, which is the exact confusion this setting exists to prevent.
    """
    prefixes = STORES if prefixes is None else _parse_stores(prefixes)
    if not prefixes:
        return True

    parts = _rel_store(entry).split("/")
    for prefix in prefixes:
        want = prefix.split("/")
        if parts[: len(want)] == want:
            return True
    return False


def store_summary() -> pd.DataFrame:
    """
    One row per tracking store on disk, with how many models it holds and
    whether it is eligible to win a metric comparison.

    Use this to see what STORES is actually excluding before trusting a
    best_per_algo() result.
    """
    rows = {}
    for e in _index():
        rel = _rel_store(e)
        row = rows.setdefault(
            rel,
            {
                "store": rel,
                "models": 0,
                "experiments": set(),
                "eligible": _in_stores(e),
            },
        )
        row["models"] += 1
        row["experiments"].add(e.experiment_name)

    return pd.DataFrame(
        [
            {
                "store": r["store"],
                "models": r["models"],
                "experiments": ", ".join(sorted(r["experiments"])),
                "eligible": r["eligible"],
            }
            for r in sorted(rows.values(), key=lambda r: r["store"])
        ]
    )


def _matches(name: str) -> List[ModelEntry]:
    """Entries whose variant, algo, run_id, or qualified key matches `name`."""
    return [e for e in _index() if name in (e.variant, e.algo, e.run_id, e.key)]


def rank(
    name: str,
    metric: str = "roc_auc",
    ascending: bool = False,
    experiment: Optional[str] = None,
) -> pd.DataFrame:
    """
    Order every run matching `name` by a metric. For LOOKING, not for loading.

    e.g. rank("cat") shows every cat run so you can see which is cat_smote.
    """
    hits = _matches(name)
    if experiment:
        hits = [e for e in hits if experiment in (e.experiment_id, e.experiment_name)]
    if not hits:
        raise LookupError(f"Nothing matches '{name}'. Try available().")

    df = pd.DataFrame(
        [
            {
                "variant": e.variant,
                "algo": e.algo,
                "experiment": e.experiment_name,
                "run_id": e.run_id,
                **e.metrics,
            }
            for e in hits
        ]
    )
    if metric in df.columns:
        df = df.sort_values(metric, ascending=ascending)
    return df.reset_index(drop=True)


def resolve(
    name: str,
    experiment: Optional[str] = None,
    policy: str = "newest",
    metric: str = "roc_auc",
) -> ModelEntry:
    """
    Resolve one entry.

    name    : a variant (cat_smote), an algo (cat), a run_id, or 'exp/variant'
    policy  : 'newest' (default) or 'best' (max `metric`; see the docstring
              warning about selecting on the evaluation set)
    """
    if "/" in name and not experiment:
        experiment, name = name.split("/", 1)

    hits = _matches(name)
    if experiment:
        hits = [e for e in hits if experiment in (e.experiment_id, e.experiment_name)]
    if not hits:
        raise LookupError(
            f"No model for '{name}'"
            + (f" in experiment '{experiment}'" if experiment else "")
            + ".\nVariants available:\n  "
            + "\n  ".join(sorted({e.variant for e in _index()}))
        )

    if len(hits) > 1:
        if policy == "best":
            scored = [e for e in hits if metric in e.metrics]
            if not scored:
                raise LookupError(
                    f"policy='best' but no run matching '{name}' logged "
                    f"'{metric}'. Metrics present: "
                    f"{sorted({m for e in hits for m in e.metrics})}"
                )
            hits = sorted(scored, key=lambda e: e.metrics[metric], reverse=True)
        else:
            hits = sorted(hits, key=lambda e: e.start_time, reverse=True)

        print(
            f"[model_registry] '{name}' matched {len(hits)} runs; "
            f"policy='{policy}' selected {hits[0].variant} "
            f"({hits[0].experiment_name}, {hits[0].run_id[:8]}). "
            "Pass a variant name to pin it."
        )

    return hits[0]


# ---------------------------------------------------------------------------
# metric-based selection
# ---------------------------------------------------------------------------

# Whatever you happened to call average precision when you logged it.
AP_ALIASES = (
    "average_precision",
    "average_precision_score",
    "avg_precision",
    "aucpr",
    "pr_auc",
    "ap",
)


def metric_names() -> List[str]:
    """Every metric key logged anywhere in the store."""
    return sorted({m for e in _index() for m in e.metrics})


def _normalize_metric(name: str) -> str:
    """
    Lower-case a metric key and collapse separators.

    MLflow keys are commonly written with spaces ('test Average Precision')
    while callers pass snake_case ('average_precision'). Without normalizing,
    a raw substring test never matches, and the default metric argument to
    best_per_algo() raises on a perfectly ordinary store.
    """
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def resolve_metric(metric: str) -> str:
    """
    Map a loose metric name onto the key actually logged.

    'average_precision' will find 'test_average_precision', 'valid ap',
    'test Average Precision', and so on. Separators are normalized, so spaces
    and underscores are interchangeable.

    Raises if the guess is ambiguous, rather than silently ranking on the
    wrong split.
    """
    names = metric_names()
    if metric in names:
        return metric

    norm_names = {n: _normalize_metric(n) for n in names}
    norm_query = _normalize_metric(metric)

    candidates = AP_ALIASES if norm_query in AP_ALIASES else (metric,)
    for cand in candidates:
        low = _normalize_metric(cand)

        exact = [n for n, norm in norm_names.items() if norm == low]
        if exact:
            return exact[0]

        partial = [n for n, norm in norm_names.items() if low in norm]
        if len(partial) == 1:
            return partial[0]
        if len(partial) > 1:
            raise LookupError(
                f"'{metric}' is ambiguous. Matches: {partial}\n"
                "Pass the exact key so the ranking split is unambiguous."
            )

    raise LookupError(f"No metric like '{metric}'. Logged metrics: {names}")


def _best_entries(
    metric: str,
    experiment: Optional[str] = None,
    per_experiment: bool = True,
    ascending: bool = False,
    stores: Optional[tuple] = None,
) -> tuple:
    """(resolved_metric, {group_key: winning ModelEntry})."""
    key = resolve_metric(metric)

    entries = [e for e in _index() if key in e.metrics]
    if experiment:
        entries = [
            e for e in entries if experiment in (e.experiment_id, e.experiment_name)
        ]

    # Store constraint applies here and nowhere else: an excluded store stays
    # fully visible to available(), rank() and load(), it just cannot win.
    active = STORES if stores is None else _parse_stores(stores)
    if active:
        eligible = [e for e in entries if _in_stores(e, active)]
        if not eligible:
            seen = sorted({_rel_store(e) for e in entries})
            raise LookupError(
                f"No runs logged '{key}' in stores {list(active)}.\n"
                f"Stores that do have it: {seen}\n"
                "Widen the constraint with set_stores(...) or pass "
                "stores=() to ignore it."
            )
        entries = eligible

    if not entries:
        raise LookupError(f"No runs logged '{key}'.")

    winners: Dict[tuple, ModelEntry] = {}
    for e in entries:
        group = (e.experiment_name, e.algo) if per_experiment else (e.algo,)
        cur = winners.get(group)
        if cur is None:
            winners[group] = e
            continue
        better = (
            e.metrics[key] < cur.metrics[key]
            if ascending
            else e.metrics[key] > cur.metrics[key]
        )
        if better:
            winners[group] = e

    return key, winners


def best_per_algo(
    metric: str = "average_precision",
    experiment: Optional[str] = None,
    per_experiment: bool = True,
    ascending: bool = False,
    stores: Optional[tuple] = None,
) -> pd.DataFrame:
    """
    The winning run for each algo, ranked by `metric`. Inspect before loading.

    per_experiment=True keeps runs from different experiments in separate
    groups instead of letting them compete.
    """
    key, winners = _best_entries(
        metric, experiment, per_experiment, ascending, stores
    )

    rows = [
        {
            "algo": e.algo,
            "store": _rel_store(e),
            "winner": e.variant,
            "experiment": e.experiment_name,
            key: e.metrics[key],
            "run_id": e.run_id,
            "n_candidates": sum(
                1
                for o in _index()
                if o.algo == e.algo
                and key in o.metrics
                and _in_stores(o, STORES if stores is None else _parse_stores(stores))
                and (not per_experiment or o.experiment_name == e.experiment_name)
            ),
            **{m: v for m, v in e.metrics.items() if m != key},
        }
        for e in winners.values()
    ]
    return (
        pd.DataFrame(rows).sort_values(key, ascending=ascending).reset_index(drop=True)
    )


def load_best_per_algo(
    metric: str = "average_precision",
    experiment: Optional[str] = None,
    per_experiment: bool = True,
    ascending: bool = False,
    qualified: bool = False,
    stores: Optional[tuple] = None,
) -> Dict[str, object]:
    """
    Load the top run for each algo by `metric`.

    NOTE: if `metric` is computed on your test set, this is model selection on
    the test set and the winning score is optimistically biased. Fine for
    exploration. For anything you report, pin the variant by name.
    """
    _, winners = _best_entries(
        metric, experiment, per_experiment, ascending, stores
    )
    out = {}
    for e in winners.values():
        out[e.key if qualified else e.algo] = _load_model(e.path)
    return out


# ---------------------------------------------------------------------------
# validation-set selection (the honest path)
# ---------------------------------------------------------------------------


def _proba(model, X):
    """Positive-class scores from a wrapper, sklearn estimator, or booster."""
    for attr in ("predict_proba", "decision_function", "predict"):
        fn = getattr(model, attr, None)
        if fn is None:
            continue
        out = fn(X)
        if attr == "predict_proba":
            arr = out.values if hasattr(out, "values") else out
            return arr[:, 1] if getattr(arr, "ndim", 1) > 1 else arr
        return out
    raise TypeError(f"{type(model)} exposes no scoring method.")


def _data_for(entry: ModelEntry, data):
    """
    data may be:
      (X, y)                      one matrix for every model
      {algo_or_variant: (X, y)}   models needing different feature spaces
      callable(entry) -> (X, y)
    """
    if callable(data):
        return data(entry)
    if isinstance(data, dict):
        for k in (entry.variant, entry.algo, entry.key):
            if k in data:
                return data[k]
        raise KeyError(
            f"No validation data for '{entry.variant}' (algo '{entry.algo}'). "
            f"Keys given: {sorted(data)}"
        )
    return data


def score_candidates(data, scorer=None, name: Optional[str] = None) -> pd.DataFrame:
    """
    Score EVERY indexed model against a held-out set. Default scorer is
    average precision.

        from sklearn.metrics import average_precision_score
        score_candidates((X_valid, y_valid))
        score_candidates({"cat": (X_val, y_val),
                          "cat_no_sex": (X_val_ablated, y_val)})
    """
    if scorer is None:
        from sklearn.metrics import average_precision_score as scorer  # noqa: N813

    entries = _index() if name is None else _matches(name)
    rows = []
    for e in entries:
        try:
            X, y = _data_for(e, data)
            model = _load_model(e.path)
            rows.append(
                {
                    "variant": e.variant,
                    "algo": e.algo,
                    "experiment": e.experiment_name,
                    "score": float(scorer(y, _proba(model, X))),
                    "run_id": e.run_id,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "variant": e.variant,
                    "algo": e.algo,
                    "experiment": e.experiment_name,
                    "score": float("nan"),
                    "run_id": e.run_id,
                    "error": str(exc)[:120],
                }
            )

    return (
        pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    )


def select_on_validation(
    data,
    scorer=None,
    per_experiment: bool = True,
) -> pd.DataFrame:
    """
    Champion of each family, chosen on the data you pass. Pass VALIDATION data.
    Then report test metrics for these winners only.
    """
    df = score_candidates(data, scorer=scorer).dropna(subset=["score"])
    group = ["experiment", "algo"] if per_experiment else ["algo"]
    idx = df.groupby(group)["score"].idxmax()
    return df.loc[idx].sort_values("score", ascending=False).reset_index(drop=True)


def load_selected(
    selection: pd.DataFrame, qualified: bool = False
) -> Dict[str, object]:
    """Load the models named in a select_on_validation() frame."""
    out = {}
    for _, row in selection.iterrows():
        entry = next(e for e in _index() if e.run_id == row["run_id"])
        out[entry.key if qualified else entry.algo] = _load_model(entry.path)
    return out


@lru_cache(maxsize=None)
def load(
    name: str,
    experiment: Optional[str] = None,
    policy: str = "newest",
    metric: str = "roc_auc",
):
    """Load one model. Cached."""
    return _load_model(resolve(name, experiment, policy, metric).path)


load_model = load  # alias


def load_all(
    only: Optional[List[str]] = None, qualified: bool = False
) -> Dict[str, object]:
    """
    One model per VARIANT (not per algo), so runs sharing an artifact folder
    stay distinct.

    qualified=True keys by '<experiment_name>/<variant>' so nothing collides
    across experiments.
    """
    entries = _index()
    if only:
        entries = [
            e for e in entries if e.variant in only or e.algo in only or e.key in only
        ]

    out: Dict[str, object] = {}
    for e in entries:  # newest-first
        key = e.key if qualified else e.variant
        if key not in out:
            out[key] = _load_model(e.path)
    return out


def verify_entry(
    name: str,
    X,
    y,
    split: Optional[str] = "test",
    tol: float = 1e-3,
    **resolve_kw,
) -> pd.DataFrame:
    """
    Recompute metrics for a loaded model and compare against what was logged.

    This is the check that makes a cross-version load defensible. If the
    recomputed score matches the logged one, the estimator state survived
    whatever shimming and repair was needed. If it does not, do not use it.

    split : which logged metrics to compare against, matched as a prefix on
            the logged key ('test', 'valid', 'train'). MUST correspond to the
            data you pass in X and y. Comparing a logged validation score
            against metrics recomputed on the test set will always disagree
            and means nothing. Pass None to see every logged metric.

    tol   : MLflow often stores metrics rounded to 3 decimals, so the default
            tolerance is 1e-3 rather than something tighter.

        verify_entry("cat_orig_training", X_test, y_test)                # test
        verify_entry("cat_orig_training", X_valid, y_valid, split="valid")
    """
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )

    entry = resolve(name, **resolve_kw)
    model = _load_model(entry.path)
    s = _proba(model, X)

    got = {
        "roc_auc": float(roc_auc_score(y, s)),
        "average_precision": float(average_precision_score(y, s)),
        "brier": float(brier_score_loss(y, s)),
    }

    rows, skipped = [], 0
    for logged_key, logged_val in sorted(entry.metrics.items()):
        norm = logged_key.lower().replace(" ", "_")

        if split is not None and not norm.startswith(split.lower()):
            skipped += 1
            continue

        match = next((k for k in got if k in norm or norm.endswith(k)), None)
        if match is None:
            continue

        delta = got[match] - logged_val
        rows.append(
            {
                "logged_metric": logged_key,
                "logged": logged_val,
                "recomputed": got[match],
                "delta": delta,
                "ok": abs(delta) <= tol,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        warnings.warn(
            f"No logged metric for '{entry.variant}' matched split="
            f"{split!r}. Logged keys: {sorted(entry.metrics)}. "
            "Pass split=None to compare against all of them.",
            RuntimeWarning,
            stacklevel=2,
        )
    return df


def verify_all(
    X,
    y,
    metric: str = "average_precision",
    split: Optional[str] = "test",
    tol: float = 1e-3,
    **best_kw,
) -> pd.DataFrame:
    """
    Verify every champion returned by best_per_algo() in one call.

    Returns one row per model with a single pass/fail, plus the worst
    absolute delta observed, so a bad load cannot hide behind an aggregate.

        verify_all(X_test, y_test, metric="valid Average Precision")
    """
    winners = best_per_algo(metric=metric, **best_kw)
    rows = []
    for _, w in winners.iterrows():
        try:
            df = verify_entry(w["winner"], X, y, split=split, tol=tol)
            rows.append(
                {
                    "algo": w["algo"],
                    "variant": w["winner"],
                    "n_compared": len(df),
                    "max_abs_delta": float(df["delta"].abs().max()) if len(df) else None,
                    "ok": bool(df["ok"].all()) if len(df) else None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "algo": w["algo"],
                    "variant": w["winner"],
                    "n_compared": 0,
                    "max_abs_delta": None,
                    "ok": False,
                    "error": str(exc)[:120],
                }
            )
    return pd.DataFrame(rows)


def variants() -> List[str]:
    return sorted({e.variant for e in _index()})


def algos() -> List[str]:
    return sorted({e.algo for e in _index()})


def experiments() -> List[str]:
    return sorted({e.experiment_name for e in _index()})


def available() -> pd.DataFrame:
    """Every model found on disk, newest first, with its logged metrics."""
    return pd.DataFrame(
        [
            {
                "variant": e.variant,
                "algo": e.algo,
                "experiment": e.experiment_name,
                "experiment_id": e.experiment_id,
                "run_id": e.run_id,
                "path": str(e.path.relative_to(PROJECT_ROOT)),
                **e.metrics,
            }
            for e in _index()
        ]
    )


def diagnose() -> None:
    print(f"project root : {PROJECT_ROOT}")
    print(f"target token : {TARGET or '(none)'}")
    print(f"model file   : {MODEL_FILE}")
    print(f"loader       : {backend()}")
    print(f"yaml parser  : {'PyYAML' if _yaml is not None else 'builtin (flat)'}")
    print(f"stores filter: {list(STORES) or '(none, all eligible)'}")
    try:
        entries = _index()
    except RuntimeError as exc:
        print()
        print(exc)
        return
    print(f"sklearn shims: {shimmed() or '(none applied yet)'}")
    print(f"state repairs: {repaired() or '(none applied yet)'}")
    print()
    print(store_summary().to_string(index=False))
    print()
    print(f"experiments  : {experiments()}")
    print(f"models       : {len(entries)}")
    print(f"algos        : {algos()}")
    print(f"variants     : {variants()}")
    print()
    print(available().to_string(index=False))


if __name__ == "__main__":
    diagnose()