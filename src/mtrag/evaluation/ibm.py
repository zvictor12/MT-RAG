"""Load evaluation code from the sibling IBM benchmark repository."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from types import ModuleType


def load_ibm_module(
    benchmark_root: str | Path,
    script_name: str,
    *,
    module_overrides: Mapping[str, ModuleType] | None = None,
) -> ModuleType:
    root = Path(benchmark_root).resolve()
    if module_overrides:
        return _execute_module(root, script_name, module_overrides)
    return _load_ibm_module(root, script_name)


@cache
def _load_ibm_module(benchmark_root: Path, script_name: str) -> ModuleType:
    return _execute_module(benchmark_root, script_name, {})


def _execute_module(
    benchmark_root: Path,
    script_name: str,
    module_overrides: Mapping[str, ModuleType],
) -> ModuleType:
    script = benchmark_root / "scripts" / "evaluation" / script_name
    if not script.is_file():
        raise FileNotFoundError(f"IBM evaluation script not found: {script}")

    suffix = hashlib.sha256(str(script).encode()).hexdigest()[:12]
    module_name = f"_mtrag_ibm_{script.stem}_{suffix}"
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load IBM evaluation script: {script}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    absent = object()
    previous_modules = {
        name: sys.modules.get(name, absent)
        for name in module_overrides
    }
    sys.modules.update(module_overrides)
    sys.path.insert(0, str(script.parent))
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as error:
        sys.modules.pop(module_name, None)
        raise RuntimeError(
            f"IBM evaluator requires missing package {error.name!r}"
        ) from error
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.path.pop(0)
        for name, previous in previous_modules.items():
            if previous is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module
