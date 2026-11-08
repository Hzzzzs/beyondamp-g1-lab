"""Plot descriptive gait metrics at a specified command speed."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reports", type=Path, nargs="+")
parser.add_argument("--labels", nargs="+", required=True)
parser.add_argument("--speed", type=float, default=0.9)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if len(args.labels) != len(args.reports):
    parser.error("Provide one label per report")
reports = [json.loads(p.read_text()) for p in args.reports]
metrics = [next(r["metrics"] for r in d["rollouts"] if np.isclose(r["command"][0], args.speed))
           for d in reports]
references = [r["metrics"] for r in reports[0]["references"]]


def cadence(m):
    values = [f["cadence_steps_per_minute"] for f in m["hip_periodicity"]["feet"]]
    return float(np.mean(values)) if all(x is not None for x in values) else np.nan


extractors = [cadence, lambda m: 100 * np.mean(m["foot_bottom_height_p95_left_right_m"]),
              lambda m: m["lowest_foot_support_proxy"]["selected_foot_horizontal_speed_mean_m_s"]]
titles = ["Hip-cycle cadence (steps/min)", "Foot-bottom height P95 (cm)", "Lowest-foot slip proxy (m/s)"]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for axis, extract, title in zip(axes, extractors, titles):
    values = [extract(m) for m in metrics]
    reference_values = [extract(m) for m in references]
    axis.axhspan(min(reference_values), max(reference_values), color="#32a883", alpha=0.2,
                 label="3 expert clips: range")
    axis.bar(args.labels, values, color=["#8696a7", "#4d83c4", "#245b90"][:len(values)])
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("Checkpoint iteration")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.2)
axes[0].legend(fontsize=8)
speeds = [m["mean_root_speed_m_s"] for m in references]
fig.suptitle(f"Policy command {args.speed:g}m/s; expert root-path speeds {min(speeds):.2f}–"
             f"{max(speeds):.2f}m/s\nHeight/contact proxies; descriptive comparison, no similarity percentage",
             fontsize=11)
fig.tight_layout()
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=180)
plt.close(fig)
