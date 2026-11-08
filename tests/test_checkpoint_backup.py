import runpy
import shutil
from pathlib import Path

from amp_lab.motion import sha256


def test_backup_retries_a_bad_readback_before_publishing(tmp_path, monkeypatch):
    helpers = runpy.run_path(str(Path(__file__).parents[1] / "scripts/backup_training.py"))
    source, target = tmp_path / "model.pt", tmp_path / "persistent/model.pt"
    source.write_bytes(b"complete checkpoint")
    original = shutil.copyfile
    copies = []

    def transient_bad_copy(src, dst):
        copies.append(dst)
        if len(copies) == 1:
            Path(dst).write_bytes(b"incomplete")
        else:
            original(src, dst)

    monkeypatch.setattr(shutil, "copyfile", transient_bad_copy)
    monkeypatch.setattr(helpers["time"], "sleep", lambda _: None)
    digest = helpers["copy_verified"](source, target)
    assert len(copies) == 2
    assert digest == sha256(source) == sha256(target)
    assert target.with_suffix(".sha256").read_text().strip() == digest
    assert not target.with_suffix(".tmp").exists()
