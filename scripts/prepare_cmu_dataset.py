"""Prepare a small checksum-locked training/held-out clip split; quality remains explicit."""

import argparse
import json
from pathlib import Path

from amp_lab.cmu import retarget_cmu
from amp_lab.download import fetch_files
from amp_lab.motion import qpos_to_motion
from amp_lab.paths import G1_XML, ROOT
from amp_lab.quality import audit_motion

parser = argparse.ArgumentParser()
parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/cmu")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--config", type=Path, help="Retargeting JSON preset; explicit CLI options override it")
parser.add_argument("--experimental-floor-ik", action=argparse.BooleanOptionalAction, default=None,
                    help="Compare floor-aware joint IK; candidate data, not contact validation")
parser.add_argument("--floor-ik-variant", choices=("anchor_v2", "smooth_v3", "smooth_v4"))
parser.add_argument("--joint-speed-limit", type=float, help="Optional IK per-joint finite-difference cap in rad/s")
parser.add_argument("--root-height-offset", type=float, help="Experimental constant root height offset in metres")
parser.add_argument("--swing-height-scale", type=float, help="Experimental swing clearance scale in (0,1]")
args = parser.parse_args()
options = {"experimental_floor_ik": False, "floor_ik_variant": "smooth_v4", "root_height_offset": 0.0,
           "swing_height_scale": 1.0, "joint_speed_limit": None}
if args.config:
    preset = json.loads(args.config.read_text())
    if preset.get("schema_version") != 1 or not isinstance(preset.get("retarget"), dict):
        raise ValueError("Expected schema_version=1 and a retarget object")
    if preset["retarget"].keys() - options.keys():
        raise ValueError("Unknown retargeting preset option")
    options.update(preset["retarget"])
for key in options:
    if getattr(args, key) is not None:
        options[key] = getattr(args, key)
if args.output.exists():
    raise FileExistsError("Use a new output directory")
lock = json.loads((ROOT / "configs/cmu_walk.lock.json").read_text())
fetch_files(lock["files"], args.raw)
manifest = {"schema_version": 1, "source": lock["source"], "split_note": lock["split_note"],
            "quality_status": "candidate_requires_contact_and_dynamic_review",
            "retarget_options": options, "clips": []}
for entry in lock["files"]:
    if "split" not in entry:
        continue
    amc = args.raw / entry["path"]
    directory = args.output / entry["split"] / amc.stem
    directory.mkdir(parents=True)
    qpos = directory / "candidate_qpos.npz"
    motion = directory / "candidate_amp.npz"
    retarget = retarget_cmu(args.raw / "07/07.asf", amc, G1_XML, qpos, fps=lock["fps"],
                            **options)
    conversion = qpos_to_motion(qpos, G1_XML, motion)
    audit = audit_motion(motion, G1_XML)
    (directory / "audit.json").write_text(json.dumps(audit, indent=2))
    manifest["clips"].append({"split": entry["split"], "motion": motion.relative_to(args.output).as_posix(),
                              "sha256": conversion["output_sha256"], "audit": audit, "retarget": retarget})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"clip": amc.stem, "split": entry["split"], **audit}), flush=True)
