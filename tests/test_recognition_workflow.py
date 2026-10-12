import io
import struct

import numpy as np
import pytest
import torch
from PIL import Image

from exp.eval_inst import BaseEval
from tools.recognize_instances import (detailed_metrics, evaluation_metrics, export_selected_colors,
                                       instance_predictions, localization_predictions, localization_targets,
                                       preparation_settings, validate_preparation_cache)


def test_selected_native_frames_keep_original_jpeg_bytes(tmp_path):
    frames = []
    for color in ['red', 'blue']:
        buffer = io.BytesIO()
        Image.new('RGB', (12, 8), color).save(buffer, format='JPEG')
        frames.append(buffer.getvalue())
    sensor = tmp_path / 'scan.sens'
    identity = np.eye(4, dtype='<f4').tobytes()
    with sensor.open('wb') as handle:
        handle.write(struct.pack('<IQ', 4, 4))
        handle.write(b'test' + identity * 4)
        handle.write(struct.pack('<iiIIIIfQ', 2, 1, 12, 8, 6, 4, 1000., 2))
        for frame in frames:
            handle.write(identity + struct.pack('<QQQQ', 0, 0, len(frame), 3) + frame + b'abc')
    size = export_selected_colors(sensor, {1}, tmp_path / 'rgb')
    assert size == (12, 8)
    assert not (tmp_path / 'rgb/0.jpg').exists()
    assert (tmp_path / 'rgb/1.jpg').read_bytes() == frames[1]
    with pytest.raises(ValueError, match='outside'):
        export_selected_colors(sensor, {2}, tmp_path / 'invalid')


def test_classification_diagnostic_requires_localized_object_instances():
    # Instance 0 is an object; instance 1 is floor and must not enter recognition accuracy.
    frozen = {'assignment': torch.tensor([1] * 120 + [2] * 120 + [3] * 30 + [0] * 70),
              'gt_instance': torch.tensor([0] * 120 + [1] * 120 + [2] * 100),
              'gt_semantic': torch.tensor([2] * 120 + [1] * 120 + [3] * 100),
              'objectness': torch.ones(3)}
    assert localization_targets(frozen).tolist() == [0, -1, -1]


def test_report_excludes_absent_classes():
    evaluator = BaseEval(['chair', 'table'], use_label=True, dataset_name='scannet200')
    ap = np.ones((1, 2, len(evaluator.ious)))
    ap[:, 1] = np.nan
    metrics = evaluation_metrics(evaluator, ap, ap)
    assert metrics['all_ap_50%'] == 1.
    assert not any('legacy' in key for key in metrics)
    assert metrics['evaluated_classes'] == 1


def test_workflow_scores_reranked_class_and_keeps_point_fallback():
    frozen = {'mask_sha256': 'same', 'point_probabilities': torch.tensor([[.1, .9, 0.], [.6, .3, .1]]),
              'objectness': torch.tensor([.8, .5])}
    data = {'mask_sha256': 'same', 'probabilities': torch.tensor([[.7, .2, .1], [0., 0., 0.]]),
            'supported': torch.tensor([True, False])}
    reranked = {'mask_sha256': 'same', 'records': [{'instance': 0, 'class_id': 1},
                                               {'instance': 1, 'class_id': None}]}
    classes, scores, probabilities = instance_predictions(frozen, data, reranked)
    assert classes.tolist() == [1, 0]
    torch.testing.assert_close(scores, torch.tensor([.16, .3]))
    torch.testing.assert_close(probabilities[1], frozen['point_probabilities'][1])


def test_workflow_rejects_recognition_from_different_masks():
    with pytest.raises(ValueError, match='do not match'):
        instance_predictions({'mask_sha256': 'first'}, {'mask_sha256': 'second'}, {'mask_sha256': 'first'})


def test_empty_scene_metrics_are_json_serializable():
    import json

    evaluator = BaseEval(['chair'], use_label=True, dataset_name='scannet200')
    ap = np.full((1, 1, len(evaluator.ious)), np.nan)
    with pytest.warns(RuntimeWarning, match='Mean of empty slice'):
        metrics = evaluation_metrics(evaluator, ap, ap)
    assert metrics['all_ap'] is None
    assert metrics['evaluated_classes'] == 0
    json.dumps(metrics, allow_nan=False)


def test_frequency_groups_exclude_absent_classes():
    import json
    from exp.scannet200_splits import COMMON_CATS_SCANNET_200, TAIL_CATS_SCANNET_200

    names = ['chair', COMMON_CATS_SCANNET_200[0], TAIL_CATS_SCANNET_200[0]]
    evaluator = BaseEval(names, use_label=True, dataset_name='scannet200')
    ap = np.full((1, 3, len(evaluator.ious)), np.nan)
    ap[0, 0] = .2
    ap[0, 2] = .6
    details = detailed_metrics(evaluator, ap, ap)
    groups = details['frequency_groups']
    assert groups['head']['ap'] == pytest.approx(.2)
    assert groups['common']['evaluated_classes'] == 0
    assert groups['common']['ap'] is None
    assert groups['tail']['ap50%'] == pytest.approx(.6)
    assert details['per_class'][names[1]]['ap'] is None
    json.dumps(details, allow_nan=False)


def test_localization_ap_retains_masks_with_unknown_recognition_labels():
    from exp.scannet200_splits import INSTANCE_CAT_SCANNET_200

    frozen = {'assignment': torch.ones(120, dtype=torch.int32), 'objectness': torch.tensor([.8]),
              'gt_semantic': torch.full((120,), 2), 'gt_instance': torch.zeros(120, dtype=torch.long)}
    predictions = localization_predictions('scene', frozen)
    evaluator = BaseEval(INSTANCE_CAT_SCANNET_200, use_label=False, dataset_name='scannet200')
    gt, pred = evaluator.assign_instances_for_scan(predictions, frozen['gt_semantic'].numpy(), frozen['gt_instance'].numpy())
    ap, recall = evaluator.evaluate_matches({'scene': {'gt': gt, 'pred': pred}})
    assert evaluation_metrics(evaluator, ap, recall)['all_ap_50%'] == 1.
    assert predictions[0]['conf'] == pytest.approx(.8)


def test_zero_objectness_masks_are_excluded_from_localization_predictions():
    frozen = {'assignment': torch.tensor([1, 1, 2, 2], dtype=torch.int32), 'objectness': torch.tensor([.8, 0.])}
    predictions = localization_predictions('scene', frozen)
    assert len(predictions) == 1
    np.testing.assert_array_equal(predictions[0]['pred_mask'], [1, 1, 0, 0])


def test_preparation_applies_configured_graph_and_clustering(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from tools import recognize_instances as workflow

    cache = tmp_path / 'cache'
    cache.mkdir()
    (cache / 'manifest.json').write_text(json.dumps({'clip_model': 'ViT-B/32', 'clip_dim': 512}))
    (cache / 'scene_preparation.json').write_text(json.dumps({'frames50': [7]}))
    config = tmp_path / 'config.yaml'
    config.write_text('graph:\n  radius: 0.12\n  window_size: 64\n  mask_threshold: 0.6\n'
                      '  appearance_mode: histogram_rgb\n  semantic_top_k: 3\n  mask_top_m: 2\n  rgb_weight: 0.4\n'
                      'clustering:\n  cluster_budget: 256\n  merge_threshold: 0.65\n  non_neighbor_weight: 0.05\n')
    args = SimpleNamespace(config=config, data_root=cache, output_root=tmp_path / 'results',
                           raw_root=tmp_path / 'raw', frames=50)
    loaded = []
    class Model:
        def __init__(self, cfg, **kwargs):
            loaded.append(cfg)
        def eval(self):
            return self
    monkeypatch.setattr(workflow.sam2mask, 'SAM2Mask', Model)
    monkeypatch.setattr(workflow, 'preparation_metadata',
                        lambda args, scene, settings, selected: {'settings': settings, 'selected': selected})
    import hashlib
    empty = torch.zeros(0, dtype=torch.int32)
    monkeypatch.setattr(workflow, 'freeze_scene', lambda *args:
                        ({'mask_sha256': hashlib.sha256(empty.numpy().tobytes()).hexdigest(), 'assignment': empty},
                         torch.zeros(0, 0)))
    monkeypatch.setattr(workflow, 'projected_views', lambda *args: [])
    monkeypatch.setattr(workflow, 'export_selected_colors', lambda *args: (640, 480))
    workflow.prepare(args, ['scene'])
    assert loaded[0].graph.radius == .12
    assert loaded[0].graph.window_size == 64
    assert loaded[0].graph.mask_threshold == .6
    assert loaded[0].graph.appearance_mode == 'histogram_rgb'
    assert loaded[0].graph.semantic_top_k == 3
    assert loaded[0].graph.mask_top_m == 2
    assert loaded[0].graph.rgb_weight == .4
    assert loaded[0].clustering.cluster_budget == 256
    assert loaded[0].clustering.merge_threshold == .65
    assert loaded[0].clustering.non_neighbor_weight == .05
    # Matching completed work is reused without loading a model.
    workflow.prepare(args, ['scene'])
    assert len(loaded) == 1
    config.write_text('graph:\n  radius: 0.3\n')
    with pytest.raises(ValueError, match='different settings or inputs'):
        workflow.prepare(args, ['scene'])
    assert len(loaded) == 1


def test_preparation_rejects_incompatible_cached_feature_model(tmp_path):
    import json
    from types import SimpleNamespace

    (tmp_path / 'manifest.json').write_text(json.dumps({'clip_model': 'ViT-B/32', 'clip_dim': 512}))
    config = tmp_path / 'config.yaml'
    config.write_text('foundation_model:\n  clip_model: ViT-L/14@336px\n  clip_dim: 768\n')
    with pytest.raises(ValueError, match='does not match cached mask features'):
        preparation_settings(SimpleNamespace(config=config, data_root=tmp_path, frames=50))


def test_preparation_cache_rejects_changed_inputs_and_legacy_metadata():
    metadata = {'settings': {'graph': {'radius': .2}}, 'input_sha256': {'pose': 'original'}}
    validate_preparation_cache({'metadata': metadata, 'mask_sha256': 'same'}, metadata, 'same')
    with pytest.raises(ValueError, match='different settings or inputs'):
        validate_preparation_cache({'metadata': metadata, 'mask_sha256': 'same'},
                                   dict(metadata, input_sha256={'pose': 'changed'}), 'same')
    with pytest.raises(ValueError, match='different settings or inputs'):
        validate_preparation_cache({'mask_sha256': 'same'}, metadata, 'same')


def test_clip_cache_rejects_changed_target_pixels_before_loading_weights(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from tools import recognize_instances as workflow

    root = tmp_path / 'scene'
    (root / 'rgb').mkdir(parents=True)
    (root / 'target_masks').mkdir()
    views = [[{'frame': 7, 'target_mask': 'target_masks/0_7.png'}]]
    (root / 'views.json').write_text(json.dumps(views))
    (root / 'frozen.pth').write_bytes(b'fixed masks')
    Image.new('RGB', (4, 2), 'red').save(root / 'rgb/7.jpg')
    Image.new('L', (4, 2), 255).save(root / 'target_masks/0_7.png')
    metadata = workflow.crop_recognition_metadata(root, views)
    torch.save({'metadata': metadata}, root / 'l336.pth')
    monkeypatch.setattr(workflow.clip, 'load', lambda *args, **kwargs: pytest.fail('cached scene loaded weights'))
    args = SimpleNamespace(output_root=tmp_path)
    workflow.extract(args, ['scene'])
    Image.new('L', (4, 2), 0).save(root / 'target_masks/0_7.png')
    with pytest.raises(ValueError, match='different settings or inputs'):
        workflow.extract(args, ['scene'])


def test_native_frame_fingerprint_detects_jpeg_changes(tmp_path):
    import hashlib
    from tools.recognize_instances import read_selected_colors

    sensor = tmp_path / 'scan.sens'
    identity = np.eye(4, dtype='<f4').tobytes()
    colors = [b'first', b'second']
    with sensor.open('wb') as handle:
        handle.write(struct.pack('<IQ', 4, 0) + identity * 4)
        handle.write(struct.pack('<iiIIIIfQ', 2, 1, 12, 8, 6, 4, 1000., 2))
        for color in colors:
            handle.write(identity + struct.pack('<QQQQ', 0, 0, len(color), 0) + color)
    observed = {}
    size = read_selected_colors(sensor, [1], lambda frame, color: observed.update({frame: hashlib.sha256(color).hexdigest()}))
    assert size == (12, 8)
    assert observed == {1: hashlib.sha256(b'second').hexdigest()}
    content = sensor.read_bytes().replace(b'second', b'edited')
    sensor.write_bytes(content)
    read_selected_colors(sensor, [1], lambda frame, color: observed.update({frame: hashlib.sha256(color).hexdigest()}))
    assert observed[1] == hashlib.sha256(b'edited').hexdigest()


@pytest.mark.parametrize('sam_masks', [None, torch.empty(0, 1, 1)])
def test_projected_targets_work_without_sam_masks_and_use_configured_dimensions(tmp_path, sam_masks):
    from types import SimpleNamespace
    from tools.recognize_instances import projected_views

    points = torch.zeros(12, 3)
    locations = torch.tensor([[1, 1], [1, 10], [10, 1], [10, 10]]).repeat(3, 1)
    mapping = torch.cat((locations, torch.ones(12, 1, dtype=torch.long)), dim=1)
    model = SimpleNamespace(img_dim=[12, 12], depth_scale=1000., pointcloud_mapper=SimpleNamespace(
        compute_mapping_torch=lambda *args, **kwargs: mapping))
    reader = SimpleNamespace(get_pcd=lambda *args: (points, None, None, None),
                             get_image_dirs=lambda *args: ['scene/color/7.jpg'],
                             get_pose=lambda *args: np.eye(4), get_depth=lambda *args: np.ones((12, 12)) * 1000.,
                             get_intrinsic=lambda *args: np.eye(3),
                             get_grounded_data=lambda *args: {'masks': sam_masks})
    views = projected_views(model, reader, 'scene', torch.ones(1, 12, dtype=torch.bool), tmp_path / 'target_masks')
    assert len(views[0]) == 1
    assert views[0][0]['box'] == [1, 1, 11, 11]
    with Image.open(tmp_path / views[0][0]['target_mask']) as foreground:
        assert foreground.size == (12, 12)
        assert foreground.getbbox() is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_crop_extraction_reuses_injected_backbone_and_matching_cache(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from tools import recognize_instances as workflow

    encoded = []
    class Encoder:
        logit_scale = torch.tensor(0., device='cuda')
        visual = SimpleNamespace(input_resolution=336)
        def eval(self):
            return self
        def encode_image(self, batch):
            encoded.append(len(batch))
            return torch.ones(len(batch), 768, device=batch.device)
    monkeypatch.setattr(workflow.clip, 'load', lambda *args, **kwargs: pytest.fail('injected backbone loaded weights'))
    monkeypatch.setattr(workflow, 'class_text_embeddings', lambda *args: torch.ones(len(workflow.NAMES), 768, device='cuda'))
    for scene in ['first', 'second']:
        root = tmp_path / scene
        (root / 'rgb').mkdir(parents=True)
        (root / 'target_masks').mkdir()
        views = [[{'frame': 7, 'target_mask': 'target_masks/0_7.png', 'native_box': [0, 0, 4, 4], 'weight': 1.}]]
        (root / 'views.json').write_text(json.dumps(views))
        (root / 'prepared.json').write_text(json.dumps({'mask_sha256': 'same'}))
        torch.save({'assignment': torch.ones(12, dtype=torch.int32)}, root / 'frozen.pth')
        Image.new('RGB', (4, 4), 'red').save(root / 'rgb/7.jpg')
        Image.new('L', (4, 4), 255).save(root / 'target_masks/0_7.png')
    args = SimpleNamespace(output_root=tmp_path, batch_size=2)
    with pytest.raises(ValueError, match='matched CLIP'):
        workflow.extract(args, ['first'], backbone=(SimpleNamespace(visual=SimpleNamespace(input_resolution=224)), None))
    backbone = (Encoder(), lambda image: torch.ones(3, 2, 2))
    workflow.extract(args, ['first', 'second'], backbone=backbone)
    assert sum(encoded) == 12
    data = torch.load(tmp_path / 'first/l336.pth', weights_only=False)
    assert data['supported'].tolist() == [True]
    torch.testing.assert_close(data['features'].norm(dim=-1), torch.ones(1))
    workflow.extract(args, ['first', 'second'], backbone=backbone)
    assert sum(encoded) == 12
