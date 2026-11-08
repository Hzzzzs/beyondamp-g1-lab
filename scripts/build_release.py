"""Stage a matched release locally. This command does not publish to GitHub."""

import argparse
from pathlib import Path

from amp_lab.paths import ASSETS
from amp_lab.release import build_release

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--spec", type=Path, required=True)
parser.add_argument("--policy", type=Path, required=True)
parser.add_argument("--motions", type=Path, nargs="+", required=True)
parser.add_argument("--reports", type=Path, nargs="+", required=True)
parser.add_argument("--robot", type=Path, default=ASSETS / "unitree/unitree_g1")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
print(build_release(args.checkpoint, args.spec, args.policy, args.motions, args.reports, args.robot, args.output))
