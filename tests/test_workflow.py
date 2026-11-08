"""Checks for checkpoint selection and workflow input safety."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("amp_workflow", Path(__file__).parents[1] / "scripts/workflow.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


def test_latest_uses_iteration_not_filename_or_mtime(tmp_path):
    for name in ("model_9.pt", "model_100.pt", "model_20.pt", "model_latest.pt"):
        (tmp_path / name).touch()
    assert workflow.latest(tmp_path).name == "model_100.pt"


def test_latest_requires_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="No model_"):
        workflow.latest(tmp_path)


def test_bundle_and_checkpoint_are_mutually_exclusive(tmp_path):
    args = SimpleNamespace(bundle=tmp_path, checkpoint="latest")
    with pytest.raises(ValueError, match="either"):
        workflow.artifact(args)
