import json
import shutil
from pathlib import Path

from .motion import sha256
from .paths import contained_path
from .spec import DeploymentSpec


def write_checksums(directory):
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p != directory / "checksums.sha256")
    (directory / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(directory).as_posix()}\n" for path in files), encoding="utf-8")


def verify_bundle(directory: Path):
    directory = directory.resolve()
    spec = DeploymentSpec.load(directory / "deployment.yaml")
    listing = directory / "checksums.sha256"
    if not listing.is_file():
        raise ValueError("Missing checksums.sha256")
    indexed = set()
    for line in listing.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = contained_path(directory, relative)
        if relative in indexed or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Checksum failure or duplicate entry: {relative}")
        indexed.add(relative)
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
              if p.is_file() and p != listing}
    if actual != indexed:
        raise ValueError("Bundle contains unindexed files or missing files")
    for relative in ("deployment.yaml", spec.policy_file, spec.model_xml):
        if relative not in indexed:
            raise ValueError(f"Missing required bundle member: {relative}")
    return spec


def create_bundle(policy, spec, robot_directory, output, *, metadata=None):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        raise FileExistsError(f"Incomplete bundle already exists: {temporary}")
    temporary.mkdir()
    try:
        policy_target = contained_path(temporary, spec.policy_file)
        policy_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(policy, policy_target)
        shutil.copytree(robot_directory, temporary / "robot")
        from .provenance import asset_manifest
        manifest = asset_manifest(temporary / "robot")
        expected = spec.provenance.get("model_tree_sha256")
        if expected and expected != manifest["tree_sha256"]:
            raise ValueError("Robot asset tree differs from the live training asset manifest")
        (temporary / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
        if not contained_path(temporary, spec.model_xml).is_file():
            raise ValueError(f"Robot directory does not contain {spec.model_xml}")
        spec.save(temporary / "deployment.yaml")
        (temporary / "metadata.json").write_text(json.dumps(metadata or {
            "status": "development", "trained_policy": False, "public_release_ready": False,
        }, indent=2), encoding="utf-8")
        write_checksums(temporary)
        verify_bundle(temporary)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return output
