import pytest
from amp_lab.download import fetch_files
from amp_lab.motion import sha256


def test_download_verifies_before_promoting_file(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("locked public content")
    target = tmp_path / "cache"
    entry = {"url": source.as_uri(), "path": "sample.txt", "sha256": sha256(source)}
    fetch_files([entry], target)
    assert (target / "sample.txt").read_text() == source.read_text()
    (target / "sample.txt").write_text("unexpected change")
    with pytest.raises(ValueError, match="Input checksum mismatch"):
        fetch_files([entry], target)
    (target / "sample.txt").unlink()
    source.write_text("server content changed")
    with pytest.raises(ValueError, match="Download checksum mismatch"):
        fetch_files([entry], target)
    assert not (target / "sample.txt").exists()
    assert not (target / "sample.txt.downloading").exists()
