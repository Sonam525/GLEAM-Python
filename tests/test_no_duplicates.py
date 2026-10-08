"""Port of tests/testthat/test-no_duplicate_functions.R.

In R every file of R/ shares one package namespace, so a function defined in
two files is silently replaced by the one collated last; the R test fails
when any top-level function name is defined more than once. Python modules
have separate namespaces, so the equivalent hazards are:

* a public function or class defined in two modules of the package (callers
  may import the wrong one, and the two copies drift apart);
* one public name registered twice in ``gleampy.__init__`` (the later
  registration silently wins), or an export that does not come from the
  module it is registered to.

Private helpers (leading underscore) are module-local and may repeat.
"""

from __future__ import annotations

import ast
import importlib
from collections import defaultdict
from pathlib import Path

import gleampy

PACKAGE_DIR = Path(gleampy.__file__).resolve().parent


def _top_level_definitions() -> dict[str, list[str]]:
    defs: dict[str, list[str]] = defaultdict(list)
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            names = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Lambda):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            for name in names:
                defs[name].append(path.relative_to(PACKAGE_DIR).as_posix())
    return defs


def test_no_public_function_is_defined_more_than_once():
    defs = _top_level_definitions()
    assert len(defs) > 300  # the scan found the package
    dupes = {name: files for name, files in defs.items() if len(files) > 1 and not name.startswith("_")}
    report = "\n".join(f"  '{name}' defined in: {', '.join(files)}" for name, files in sorted(dupes.items()))
    assert not dupes, f"Duplicate function definitions found:\n{report}"


def _registered_names() -> list[str]:
    tree = ast.parse((PACKAGE_DIR / "__init__.py").read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_register":
            names += [a.value for a in node.args[1:] if isinstance(a, ast.Constant)]
    return names


def test_no_export_is_registered_twice():
    names = _registered_names()
    seen, dupes = set(), set()
    for n in names:
        (dupes if n in seen else seen).add(n)
    assert not dupes, f"registered more than once: {sorted(dupes)}"
    assert len(names) == len(gleampy._EXPORTS) > 80  # every registration was found
    assert len(gleampy.__all__) == len(set(gleampy.__all__))


def test_every_export_comes_from_its_registered_module():
    wrong = {}
    for name, module_name in gleampy._EXPORTS.items():
        module = importlib.import_module(module_name)
        obj = getattr(gleampy, name)
        assert obj is getattr(module, name)
        defined_in = getattr(obj, "__module__", module_name)
        if defined_in != module_name:
            wrong[name] = (module_name, defined_in)
    assert not wrong, f"exports defined elsewhere than registered: {wrong}"


def test_every_run_and_calc_function_is_exported_once():
    defs = _top_level_definitions()
    public_api = {n for n in defs if n.startswith(("run_", "calc_"))}
    # every calc_* / run_* of the core and module packages is part of the API
    missing = sorted(
        n for n in public_api
        if any(f.startswith(("core/", "modules/")) for f in defs[n]) and n not in gleampy._EXPORTS
    )
    assert not missing, f"public run_/calc_ functions not exported: {missing}"
