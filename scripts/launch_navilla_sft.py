#!/usr/bin/env python3
"""Launcher for NaVILA multi-action sequence SFT training.

Runs the vendored LLaVA train_mem.py from the correct directory context
so that relative imports work properly.
"""
import os
import sys
import importlib
from unittest import mock

# Pin the vendored LLaVA package location to prevent conflicts with
# other copies (e.g., in falcon_collect_data or DPed_pro)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NAVILLA_LLAVA = os.path.join(
    _PROJECT_ROOT, "habitat-baselines", "habitat_baselines",
    "rl", "ddppo", "policy", "navila", "llava",
)
_NAVILLA_LLAVA = os.path.abspath(_NAVILLA_LLAVA)

# Force llava -> our vendored copy
if "llava" in sys.modules:
    del sys.modules["llava"]
spec = importlib.util.spec_from_file_location(
    "llava",
    os.path.join(_NAVILLA_LLAVA, "__init__.py"),
    submodule_search_locations=[_NAVILLA_LLAVA],
)
if spec is not None:
    mod = importlib.util.module_from_spec(spec)
    sys.modules["llava"] = mod
    spec.loader.exec_module(mod)

if _NAVILLA_LLAVA not in sys.path:
    sys.path.insert(0, _NAVILLA_LLAVA)
_train_dir = os.path.join(_NAVILLA_LLAVA, "train")
if _train_dir not in sys.path:
    sys.path.insert(0, _train_dir)

# Import and run train()
from train.train_mem import __len__, __iter__  # monkey-patch helpers
from train.train import train
from train.transformer_normalize_monkey_patch import patched_normalize

if __name__ == "__main__":
    with (
        mock.patch("transformers.image_processing_utils.normalize", new=patched_normalize),
        mock.patch("accelerate.data_loader.BatchSamplerShard.__len__", new=__len__),
        mock.patch("accelerate.data_loader.BatchSamplerShard.__iter__", new=__iter__),
    ):
        train()
