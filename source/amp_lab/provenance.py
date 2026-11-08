import hashlib
from pathlib import Path

from .motion import sha256


def asset_manifest(directory):
    """Hash robot XML, URDF, meshes and USD together, including every relative filename."""
    directory = Path(directory)
    files = {path.relative_to(directory).as_posix(): sha256(path)
             for path in sorted(directory.rglob("*")) if path.is_file()}
    aggregate = hashlib.sha256()
    for name, checksum in files.items():
        aggregate.update(f"{checksum}  {name}\n".encode())
    return {"tree_sha256": aggregate.hexdigest(), "files": files}
