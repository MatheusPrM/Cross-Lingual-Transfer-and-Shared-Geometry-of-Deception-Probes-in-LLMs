"""Probe loading without the model-training import chain."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def load_detector_stack():
    import importlib.util
    import types
    if "project.detectors" in sys.modules and "_load_probes" in sys.modules:
        det = sys.modules["project.detectors"]
        lp = sys.modules["_load_probes"]
        return det.LogisticRegressionDetector, lp.probe_path
    tr = types.ModuleType("transformers")
    tr.PreTrainedModel = object
    tr.PreTrainedTokenizerBase = object
    sys.modules.setdefault("transformers", tr)
    td = types.ModuleType("project.tokenized_data")
    td.TokenizedDataset = object
    td.stub_tokenized_for_masked_counts = None
    sys.modules.setdefault("project.tokenized_data", td)
    pkg = types.ModuleType("project")
    pkg.__path__ = [str(_REPO / "project")]
    sys.modules.setdefault("project", pkg)
    for name in ("types", "scores", "activations", "detectors"):
        if f"project.{name}" not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                f"project.{name}", _REPO / "project" / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[f"project.{name}"] = mod
            spec.loader.exec_module(mod)
    spec = importlib.util.spec_from_file_location(
        "_load_probes", _REPO / "scripts" / "analysis" / "load_probes.py")
    lp = importlib.util.module_from_spec(spec)
    sys.modules["_load_probes"] = lp
    spec.loader.exec_module(lp)
    return sys.modules["project.detectors"].LogisticRegressionDetector, \
        lp.probe_path
