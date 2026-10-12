from pathlib import Path

import pytest

from tools.prepare_cache import diverse_frames, full_validation_scenes, scene_cache_complete


def test_full_validation_keeps_all_scans_of_the_same_room(tmp_path):
    scenes = ['scene0001_00', 'scene0001_01', 'scene0002_00']
    split = tmp_path / 'val.txt'
    split.write_text('\n'.join(scenes) + '\n')
    for scene in scenes:
        root = tmp_path / 'scans' / scene
        root.mkdir(parents=True)
        for suffix in ['.sens', '_vh_clean_2.ply', '_vh_clean_2.0.010000.segs.json', '.aggregation.json']:
            (root / (scene + suffix)).touch()
    assert full_validation_scenes(tmp_path, split) == scenes
    (tmp_path / 'scans/scene0001_01/scene0001_01.sens').unlink()
    with pytest.raises(FileNotFoundError, match='scene0001_01.sens'):
        full_validation_scenes(tmp_path, split)


def test_incomplete_cache_is_not_skipped_on_resume(tmp_path):
    scene = 'scene0001_00'
    files = [Path('scannet200/val') / f'{scene}.pth', Path('intrinsic') / f'{scene}.txt',
             Path('masks/val') / scene / 'sam2/0.pth']
    files += [Path('scannet_2d/val') / scene / folder / f'0.{suffix}'
              for folder, suffix in [('color', 'jpg'), ('depth', 'png'), ('pose', 'txt')]]
    for file in files:
        target = tmp_path / file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    assert scene_cache_complete(tmp_path, scene, [0])
    (tmp_path / 'scannet_2d/val' / scene / 'depth/0.png').unlink()
    assert not scene_cache_complete(tmp_path, scene, [0])
    assert not scene_cache_complete(tmp_path, scene, [])


def test_all_views_include_every_valid_camera_without_the_default_limit():
    import numpy as np
    from libs.inference import parse_frame_count, select_camera_frames
    from types import SimpleNamespace

    poses = np.repeat(np.eye(4)[None], 65, axis=0)
    poses[12] = np.nan
    valid = np.flatnonzero(np.isfinite(poses).all(axis=(1, 2)))
    count = parse_frame_count('all')
    assert count == 0
    assert diverse_frames(poses, valid, count) == valid.tolist()
    reader = SimpleNamespace(get_pose=lambda index: poses[index])
    assert select_camera_frames(list(range(65)), reader, count) == valid.tolist()
    assert len(diverse_frames(poses, valid, 50)) == 50


@pytest.mark.parametrize('value', ['0', '-1', '3.5', 'invalid'])
def test_invalid_frame_limits_are_rejected(value):
    import argparse
    from libs.inference import parse_frame_count

    with pytest.raises(argparse.ArgumentTypeError, match='positive integer or all'):
        parse_frame_count(value)
