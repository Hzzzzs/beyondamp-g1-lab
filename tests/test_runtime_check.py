import sys
from types import ModuleType

import pytest
from amp_lab.training import runtime_check


def test_actual_isaac_version_tuple_and_wrong_runtime(monkeypatch):
    lab = ModuleType("isaaclab")
    lab.__version__ = "0.36.21"
    version = ModuleType("isaacsim.core.version")
    # Tuple observed on the actual 4.5.0 GPU image, including its build suffix.
    version.get_version = lambda: ("4.5.0", "rc.36", "4", "5", "0", "rc", "36", "isaaclab_minimal_45")
    monkeypatch.setitem(sys.modules, "isaaclab", lab)
    monkeypatch.setitem(sys.modules, "isaacsim.core.version", version)
    monkeypatch.setattr(runtime_check.platform, "python_version_tuple", lambda: ("3", "10", "15"))
    assert runtime_check.check_runtime()["isaac_sim"] == "4.5.0"
    version.get_version = lambda: ("5.0.0", "", "5", "0", "0")
    with pytest.raises(RuntimeError, match="got extension.*Sim 5.0.0"):
        runtime_check.check_runtime()
    version.get_version = lambda: ("4.5.0", "rc.36", "4", "5", "0")
    lab.__version__ = "0.40.0"
    with pytest.raises(RuntimeError, match="Expected IsaacLab"):
        runtime_check.check_runtime()
