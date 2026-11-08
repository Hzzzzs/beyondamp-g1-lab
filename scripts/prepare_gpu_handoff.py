"""Create a local transfer archive; does not rent, connect to, or start a GPU machine."""

import argparse
import io
import json
import os
import shlex
import subprocess
import tarfile
from pathlib import Path

from amp_lab.motion import sha256
from amp_lab.paths import ROOT, UPSTREAM, contained_path


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


def tracked_files(directory):
    listing = subprocess.check_output(["git", "-C", str(directory), "ls-files", "-z"])
    return [directory / name.decode() for name in listing.split(b"\0") if name]


def build_file_index(entries):
    checksums, links = [], {}
    for path, relative in entries:
        if path.is_symlink():
            target = os.readlink(path)
            if Path(target).is_absolute():
                raise ValueError(f"Absolute symlink is not portable: {relative}")
            links[relative] = target
        else:
            checksums.append(f"{sha256(path)}  {relative}\n")
    return "".join(checksums), links


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared dataset manifest.json")
    parser.add_argument("--output", type=Path, required=True, help="Local .tar.gz destination")
    parser.add_argument("--include-upstream", action="store_true", help="Include locked dependency checkouts")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if git(ROOT, "status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("Commit project changes before creating a reproducible GPU handoff")
    dataset_path = contained_path(ROOT, str(args.dataset.resolve().relative_to(ROOT)))
    dataset = json.loads(dataset_path.read_text())
    files = set(tracked_files(ROOT))
    for clip in dataset["clips"]:
        motion = contained_path(dataset_path.parent, clip["motion"])
        if sha256(motion) != clip["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {motion}")
    # Retain per-clip provenance/qpos/audits and both splits, but never existing training outputs.
    files.update(path for path in dataset_path.parent.rglob("*") if path.is_file())
    upstream_lock = json.loads((ROOT / "configs/upstream.lock.json").read_text())
    if args.include_upstream:
        for name, entry in upstream_lock["repositories"].items():
            checkout = UPSTREAM / name
            if (git(checkout, "rev-parse", "HEAD") != entry["commit"]
                    or git(checkout, "status", "--porcelain", "--untracked-files=no")):
                raise RuntimeError(f"Upstream checkout does not match lock: {name}")
            if git(checkout, "remote", "get-url", "origin") != entry["url"]:
                raise RuntimeError(f"Unexpected remote URL; do not archive credentials: {name}")
            files.update(tracked_files(checkout))
            # Preserve repository metadata so fetch-upstream verifies the pin without re-cloning.
            metadata = checkout / ".git"
            if not metadata.is_dir():
                raise ValueError("Handoff expects regular dependency clones, not linked worktrees")
            files.update(path for path in metadata.rglob("*") if path.is_file())
    relative_files = []
    for path in sorted(files):
        relative = path.relative_to(ROOT).as_posix()
        contained_path(ROOT, relative)  # Reject symlinks escaping the source tree.
        relative_files.append((path, relative))
    dataset_relative = dataset_path.parent.relative_to(ROOT).as_posix()
    manifest = {"schema_version": 1, "purpose": "private_gpu_development_handoff_not_public_release",
                "source_git_revision": git(ROOT, "rev-parse", "HEAD"),
                "dataset_manifest": dataset_path.relative_to(ROOT).as_posix(),
                "dataset_manifest_sha256": sha256(dataset_path), "upstream_included": args.include_upstream,
                "upstream_lock": upstream_lock, "file_count": len(relative_files)}
    instructions = f'''# GPU 开发交接包（非公开 Release）

目标：Isaac Sim 4.5.0 / IsaacLab v2.1.0 / Python 3.10 / RTX 4090。
在 Linux 上使用 GNU tar 解压以保留原始符号链接：`tar -xzf {shlex.quote(output.name)}`。
先启用镜像的 Isaac Python 环境，再进入解压后的 amp-lab 目录。
安装依赖需要软件源网络；本包不包含镜像、CUDA、pip wheels 或本地 .venv。
依赖源码是否已包含：{args.include_upstream}。
候选动作仍需动力学验证，没有附带可行走的示例权重。

```bash
AMP_ISAACLAB=/workspace/isaaclab  # 按实际安装目录修改
"$AMP_ISAACLAB/isaaclab.sh" -p -m pip install -c constraints/isaac45.txt -e '.[mujoco,export]'
"$AMP_ISAACLAB/isaaclab.sh" -p -m amp_lab.cli fetch-upstream --install
"$AMP_ISAACLAB/isaaclab.sh" -p -m amp_lab.cli doctor --training
"$AMP_ISAACLAB/isaaclab.sh" -p scripts/preflight_isaac.py --headless --report outputs/gpu_preflight.json
"$AMP_ISAACLAB/isaaclab.sh" -p scripts/train.py --headless --num-envs 32 --iterations 10 --motions {dataset_relative}/train/*/candidate_amp.npz --output outputs/gpu_smoke
```

先检查 preflight 和短训练结果，再扩大并行数或训练时长。保留镜像 torch/CUDA；不要安装 CPU torch。
原始代码版本和数据指纹见 GPU_HANDOFF.json，普通文件指纹见 GPU_HANDOFF.sha256，符号链接目标记录在 GPU_HANDOFF.json。
保留 THIRD_PARTY.md 和第三方原始许可；本包是私人开发传输包，未经公开发布审核。
'''
    checksums, links = build_file_index(relative_files)
    manifest["symlinks"] = links
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with tarfile.open(temporary, "w:gz", compresslevel=1, dereference=False) as archive:
            for path, relative in relative_files:
                archive.add(path, arcname=f"amp-lab/{relative}", recursive=False)
            for name, content in (("GPU_HANDOFF.json", json.dumps(manifest, indent=2)),
                                  ("GPU_HANDOFF.sha256", checksums), ("GPU_HANDOFF.md", instructions)):
                payload = content.encode()
                info = tarfile.TarInfo(f"amp-lab/{name}")
                info.size, info.mode = len(payload), 0o644
                archive.addfile(info, io.BytesIO(payload))
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(json.dumps(dict(manifest, archive=str(output), archive_sha256=sha256(output)), indent=2))


if __name__ == "__main__":
    main()
