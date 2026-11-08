"""Check actual runtime versions rather than infer them from an image's display name."""

import platform


def check_runtime():
    import isaaclab
    from isaacsim.core.version import get_version

    # Isaac Sim returns (full_version, prerelease, major, minor, patch, ...).
    # Joining the first three entries would turn 4.5.0 into 4.5.0.rc.36.4.
    sim = get_version()[0]
    if platform.python_version_tuple()[:2] != ("3", "10"):
        raise RuntimeError("The training target requires Python 3.10")
    if isaaclab.__version__ != "0.36.21" or sim != "4.5.0":
        raise RuntimeError(f"Expected IsaacLab v2.1.0 (extension 0.36.21) / Isaac Sim 4.5.0; "
                           f"got extension {isaaclab.__version__} / Sim {sim}")
    return {"python": platform.python_version(), "isaaclab_extension": isaaclab.__version__, "isaac_sim": sim}
