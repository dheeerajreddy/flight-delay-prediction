"""Build model instances from the config's dotted import paths.

Add a model by adding a block to config.yaml — no code change required.
This is the 'open for extension, closed for modification' bit.
"""
from __future__ import annotations

import importlib
from typing import Any


def _import_class(dotted: str):
    module_path, cls_name = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), cls_name)


def build_models(models_cfg: dict[str, Any], random_state: int) -> dict[str, dict]:
    """Return {name: {'estimator': obj, 'native_categoricals': bool}}."""
    built = {}
    for name, spec in models_cfg.items():
        cls = _import_class(spec["import"])
        params = dict(spec.get("params", {}))
        # Inject a shared seed under whatever keyword the library expects.
        for seed_kw in ("random_state", "random_seed", "random_seed"):
            try:
                cls(**{**params, seed_kw: random_state})
                params[seed_kw] = random_state
                break
            except TypeError:
                continue
        built[name] = {
            "estimator": cls(**params),
            "native_categoricals": spec.get("native_categoricals", False),
        }
    return built
