import json
import os
import runpy
import subprocess
import sys
import tarfile

from amp_lab.motion import sha256
from amp_lab.paths import ROOT


def test_handoff_preserves_directory_symlinks_and_excludes_untracked_outputs(tmp_path, monkeypatch):
    project = tmp_path / 'project'
    project.mkdir()
    (project/'configs').mkdir()
    (project/'configs/upstream.lock.json').write_text('{"schema_version":1,"repositories":{}}')
    (project/'.gitignore').write_text('data/\noutputs/\n')
    (project/'target').mkdir()
    (project/'target/code.txt').write_text('tracked source')
    os.symlink('target/', project/'shortcut', target_is_directory=True)
    subprocess.run(['git', 'init', str(project)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(project), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(project), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@localhost',
                    'commit', '-m', 'fixture'], check=True, capture_output=True)
    data = project/'data/candidate'
    (data/'train/clip').mkdir(parents=True)
    motion = data/'train/clip/candidate_amp.npz'
    motion.write_bytes(b'fixture data for transfer integrity')
    manifest = data/'manifest.json'
    manifest.write_text(json.dumps({'clips': [{'split': 'train', 'motion': 'train/clip/candidate_amp.npz',
                                             'sha256': sha256(motion)}]}))
    (project/'outputs').mkdir()
    (project/'outputs/old_weight.pt').write_bytes(b'must not transfer')
    output = project/'outputs/handoff.tar.gz'
    script = runpy.run_path(str(ROOT/'scripts/prepare_gpu_handoff.py'))
    script['main'].__globals__['ROOT'] = project
    monkeypatch.setattr(sys, 'argv', ['prepare_gpu_handoff.py', '--dataset', str(manifest), '--output', str(output)])
    script['main']()
    with tarfile.open(output) as archive:
        names = archive.getnames()
        assert 'amp-lab/outputs/old_weight.pt' not in names
        assert archive.getmember('amp-lab/shortcut').issym()
        assert archive.getmember('amp-lab/shortcut').linkname == 'target/'
        info = json.load(archive.extractfile('amp-lab/GPU_HANDOFF.json'))
        assert info['symlinks'] == {'shortcut': 'target/'}
        listing = archive.extractfile('amp-lab/GPU_HANDOFF.sha256').read().decode()
        assert f'{sha256(motion)}  data/candidate/train/clip/candidate_amp.npz' in listing
        assert 'shortcut' not in listing
        assert 'amp-lab/target/code.txt' in names
