"""Check exported geometry, labels, and synchronized point sampling."""

import json

import numpy as np
import pytest
import torch

from tools.export_web_examples import POINT_DTYPE, export_scene


def test_browser_examples_preserve_geometry_and_assignment_alignment(tmp_path):
    source = tmp_path / 'data/scannet200/val'
    source.mkdir(parents=True)
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=np.float32)
    torch.save({'coords': coords, 'colors': np.ones((4, 3)),
                'labels': np.array([0, 2, 3, -100]), 'instance_ids': np.array([0, 1, 2, -1])}, source / 'scene.pth')
    root = tmp_path / 'predictions'; root.mkdir()
    torch.save({'assignment': torch.tensor([0, 1, 2, 0]), 'class_ids': torch.tensor([0, -1]),
                'class_names': ['chair', 'table'], 'scores': torch.tensor([.8, .4])}, root / 'scene.pth')
    output = tmp_path / 'web'
    metadata = export_scene(tmp_path / 'data', root, output, 'scene', 4)
    points = np.fromfile(output / 'scene/points.bin', dtype=POINT_DTYPE)
    reconstructed = points['xyz'] * metadata['scale_meters'] + metadata['center']
    np.testing.assert_allclose(reconstructed, coords)
    assert POINT_DTYPE.itemsize == 22
    assert points['gt_instance'].tolist() == [0, 2, 3, 0]
    assert points['gt_class'].tolist() == [-1, 0, 1, -1]
    assert points['prediction'].tolist() == [0, 1, 2, 0]
    assert metadata['prediction'][1]['label'] == 'unknown'
    assert (points['rgb'] == 255).all()
    assert json.loads((output / 'scene/scene.json').read_text()) == metadata
    before = (output / 'scene/points.bin').read_bytes()
    export_scene(tmp_path / 'data', root, output, 'scene', 4)
    assert (output / 'scene/points.bin').read_bytes() == before


def test_browser_examples_reject_point_count_mismatch(tmp_path):
    source = tmp_path / 'data/scannet200/val'; source.mkdir(parents=True)
    torch.save({'coords': np.zeros((3, 3))}, source / 'scene.pth')
    root = tmp_path / 'predictions'; root.mkdir()
    torch.save({'assignment': torch.tensor([0, 1])}, root / 'scene.pth')
    with pytest.raises(ValueError, match='must match the source cloud'):
        export_scene(tmp_path / 'data', root, tmp_path / 'web', 'scene', 4)


@pytest.mark.parametrize('class_id,score', [(2, .8), (-2, .8), (0, float('nan'))])
def test_browser_examples_reject_invalid_classes_and_scores(tmp_path, class_id, score):
    source = tmp_path / 'data/scannet200/val'; source.mkdir(parents=True)
    torch.save({'coords': np.zeros((3, 3))}, source / 'scene.pth')
    root = tmp_path / 'predictions'; root.mkdir()
    torch.save({'assignment': torch.ones(3, dtype=torch.int32), 'class_ids': torch.tensor([class_id]),
                'scores': torch.tensor([score]), 'class_names': ['chair', 'table']}, root / 'scene.pth')
    with pytest.raises(ValueError, match='class ID|nonfinite instance scores'):
        export_scene(tmp_path / 'data', root, tmp_path / 'web', 'scene', 3)


def test_example_accuracy_uses_full_cloud_before_display_sampling(tmp_path):
    source = tmp_path / 'data/scannet200/val'; source.mkdir(parents=True)
    coords = np.column_stack([np.arange(240), np.zeros(240), np.ones(240)]).astype(np.float32)
    torch.save({'coords': coords, 'colors': np.ones((240, 3)),
                'labels': np.repeat([2, 3], 120), 'instance_ids': np.repeat([0, 1], 120)}, source / 'scene.pth')
    root = tmp_path / 'predictions'; root.mkdir()
    torch.save({'assignment': torch.repeat_interleave(torch.tensor([1, 2]), 120),
                'class_ids': torch.tensor([0, 0]), 'scores': torch.tensor([.8, .8]),
                'class_names': ['chair', 'table']}, root / 'scene.pth')
    result = export_scene(tmp_path / 'data', root, tmp_path / 'web', 'scene', 10)
    assert result['displayed_points'] == 10
    assert result['localized_class_accuracy']['total'] == 2
    assert result['localized_class_accuracy']['correct'] == 1
    assert result['localized_class_accuracy']['accuracy'] == .5
    assert result['localized_class_accuracy']['point_count'] == 240
