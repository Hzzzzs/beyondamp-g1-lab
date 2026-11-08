"""Back up complete checkpoints to a persistent mount while training continues."""

import argparse
import json
import shutil
import time
from pathlib import Path

from amp_lab.motion import sha256


def copy_verified(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    for attempt in range(3):
        try:
            shutil.copyfile(source, temporary)
            digest = sha256(source)
            if sha256(temporary) != digest:
                raise OSError("Checkpoint changed during copy or persistent readback differs")
            temporary.replace(target)
            target.with_suffix(".sha256").write_text(digest + "\n")
            return digest
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--every", type=int, default=500)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--timeout", type=float, default=86400)
    args = parser.parse_args()
    if args.every < 1 or args.poll_seconds <= 0 or args.timeout <= 0:
        parser.error("Intervals and timeout must be positive")
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        for run in args.runs:
            directory = args.output / run.name
            directory.mkdir(parents=True, exist_ok=True)
            try:
                for name in ["deployment.yaml", "training.json", "metrics.jsonl"]:
                    source = run / name
                    if source.exists():
                        shutil.copyfile(source, directory / name)
            except OSError as error:
                print(json.dumps({"run": str(run), "metadata_backup_failed": str(error)}), flush=True)
            for checkpoint in sorted(run.glob("model_*.pt")):
                iteration = int(checkpoint.stem.split("_")[-1])
                if iteration % args.every:
                    continue
                target = args.output / run.name / checkpoint.name
                try:
                    if not (target.exists() and target.with_suffix(".sha256").exists()):
                        digest = copy_verified(checkpoint, target)
                        print(json.dumps({"checkpoint": str(checkpoint), "persistent": str(target),
                                          "sha256": digest}), flush=True)
                    # Evaluation can arrive after a checkpoint was already copied.
                    report = run / "evaluations" / checkpoint.stem / "mujoco.json"
                    if report.exists():
                        shutil.copyfile(report, target.parent / (checkpoint.stem + "_mujoco.json"))
                except OSError as error:
                    print(json.dumps({"checkpoint": str(checkpoint), "backup_failed": str(error),
                                      "retry_next_poll": True}), flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
