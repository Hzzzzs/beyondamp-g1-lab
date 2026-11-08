"""Re-audit existing, hash-locked motion splits without retargeting or modifying the data."""

import argparse
import json
from pathlib import Path

from amp_lab.motion import sha256
from amp_lab.paths import G1_XML, contained_path
from amp_lab.quality import audit_motion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--xml", type=Path, default=G1_XML)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new report path; existing results are preserved")
    source = json.loads(args.manifest.read_text())
    results = []
    for clip in source["clips"]:
        path = contained_path(args.manifest.parent, clip["motion"])
        if sha256(path) != clip["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {clip['motion']}")
        audit = audit_motion(path, args.xml)
        results.append({"split": clip["split"], "motion": clip["motion"], "audit": audit})
        print(json.dumps({"motion": clip["motion"], "support": audit["support_hypothesis"],
                          "warnings": audit["review_warnings"]}), flush=True)
    report = {"schema_version": 2, "manifest_sha256": sha256(args.manifest), "clips": results,
              "dynamic_feasibility_validated": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
