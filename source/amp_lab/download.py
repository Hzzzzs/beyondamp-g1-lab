"""Download declared public files atomically and verify locked content hashes."""

import urllib.request

from .motion import sha256
from .paths import contained_path


def fetch_files(entries, directory):
    for entry in entries:
        target = contained_path(directory, entry["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".downloading")
            try:
                with urllib.request.urlopen(entry["url"], timeout=60) as response:
                    temporary.write_bytes(response.read())
                if sha256(temporary) != entry["sha256"]:
                    raise ValueError(f"Download checksum mismatch: {entry['path']}")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        if sha256(target) != entry["sha256"]:
            raise ValueError(f"Input checksum mismatch: {target}")
