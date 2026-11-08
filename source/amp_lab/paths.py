from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / ".cache" / "upstream"
ASSETS = UPSTREAM / "assetslib"
G1_XML = ASSETS / "unitree/unitree_g1/mjcf/scene_29dof.xml"


def contained_path(root: Path, relative: str) -> Path:
    """Resolve package-local resources without allowing traversal or external symlinks."""
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Resource escapes package directory: {relative}")
    return path
