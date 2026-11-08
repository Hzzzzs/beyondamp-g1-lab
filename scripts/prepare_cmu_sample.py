"""Download checksum-locked public CMU data and build a clearly labeled retargeting candidate."""

import argparse
import json
from pathlib import Path

from amp_lab.cmu import retarget_cmu
from amp_lab.contact import project_walking_contacts
from amp_lab.download import fetch_files
from amp_lab.motion import qpos_to_motion
from amp_lab.paths import G1_XML, ROOT

parser = argparse.ArgumentParser()
parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/cmu")
parser.add_argument("--output", type=Path, default=ROOT / "data/processed/cmu07_01")
parser.add_argument("--max-frames", type=int)
parser.add_argument("--experimental-contact-projection", action="store_true",
                    help="Also generate the experimental foot-lock variant for comparison; not the default data")
args = parser.parse_args()
lock = json.loads((ROOT / "configs/cmu_sample.lock.json").read_text())
fetch_files(lock["files"], args.raw)
args.output.mkdir(parents=True, exist_ok=True)
qpos = args.output / "candidate_qpos.npz"
if qpos.exists() or (args.output / "candidate_amp.npz").exists():
    raise FileExistsError("Use a new output directory to preserve data provenance")
print(json.dumps(retarget_cmu(args.raw / "07/07.asf", args.raw / "07/07_01.amc", G1_XML, qpos,
                              fps=lock["fps"], max_frames=args.max_frames), indent=2))
print(json.dumps(qpos_to_motion(qpos, G1_XML, args.output / "candidate_amp.npz"), indent=2))
if args.experimental_contact_projection:
    locked = args.output / "contact_candidate_qpos.npz"
    print(json.dumps(project_walking_contacts(qpos, G1_XML, locked), indent=2))
    print(json.dumps(qpos_to_motion(locked, G1_XML, args.output / "contact_candidate_amp.npz"), indent=2))
