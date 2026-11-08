"""Plot actual metrics.jsonl files; iteration, sample count and time are separate comparisons."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("paths", type=Path, nargs="+")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--x", choices=["iteration", "total_steps", "wall_seconds"], default="total_steps")
parser.add_argument("--metrics", nargs="+", default=["task_reward_rate", "amp_reward", "amp_loss"],
                    help="Metric names; e.g. rollout_forward_velocity rollout_velocity_mse_xy")
args = parser.parse_args()
import matplotlib.pyplot as plt  # noqa: E402

figure, axes = plt.subplots(1, len(args.metrics), figsize=(4.7 * len(args.metrics), 4), squeeze=False)
for path in args.paths:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No records: {path}")
    for previous, current in zip(records, records[1:]):
        if current["iteration"] <= previous["iteration"]:
            raise ValueError("Repeated/out-of-order iterations: use each resume segment as a separate input")
    elapsed = 0
    for record in records:
        elapsed += record["seconds"]
        record["wall_seconds"] = elapsed
    for axis, key in zip(axes[0], args.metrics):
        if not any(row.get(key) is not None for row in records):
            raise ValueError(f"No values for {key!r} in {path}; earlier runs may lack this metric")
        axis.plot([row[args.x] for row in records],
                  [float("nan") if row.get(key) is None else row[key] for row in records],
                  label=path.parent.name)
        axis.set(xlabel=args.x, ylabel=key)
        axis.grid(alpha=0.3)
        axis.legend()
figure.tight_layout()
args.output.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(args.output, dpi=180)
