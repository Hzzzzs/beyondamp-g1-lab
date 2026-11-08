import json

import numpy as np
import pytest
from amp_lab.motion import sha256
from amp_lab.smoothing import smooth_qpos


def test_joint_filter_attenuates_jitter_preserves_root_and_provenance(tmp_path):
    time = np.arange(201) / 50
    q = np.sin(2*np.pi*12*time)[:, None]
    source, output = tmp_path / 'raw.npz', tmp_path / 'filtered.npz'
    root = np.column_stack([time, time*0, time*0+.8])
    quat = np.tile([1., 0, 0, 0], (len(time), 1))
    np.savez(source, fps=50, joint_pos=q, joint_names=['knee'], root_pos=root, root_quat_wxyz=quat)
    source.with_suffix('.retarget.json').write_text(json.dumps({
        'qpos_sha256': sha256(source), 'source': 'fixture', 'dynamics_validated': True,
    }))
    report = smooth_qpos(source, output)
    with np.load(output) as filtered:
        np.testing.assert_array_equal(filtered['root_pos'], root)
        np.testing.assert_array_equal(filtered['root_quat_wxyz'], quat)
        assert np.abs(filtered['joint_pos'][10:-10]).max() < .05
        assert filtered['joint_pos'].min() >= q.min()
        assert filtered['joint_pos'].max() <= q.max()
    assert report['qpos_sha256'] == sha256(output)
    assert report['parent_retarget']['qpos_sha256'] == sha256(source)
    assert report['dynamics_validated'] is False
    with pytest.raises(FileExistsError):
        smooth_qpos(source, output)
    source.with_suffix('.retarget.json').write_text('{"qpos_sha256":"wrong"}')
    with pytest.raises(ValueError, match='provenance'):
        smooth_qpos(source, tmp_path/'other.npz')
    assert not (tmp_path/'other.npz').exists()
