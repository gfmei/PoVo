import json

import numpy as np
import pytest
import torch

from datasets.scannet import preprocess_scannet200 as preprocess
from exp.eval_inst import BaseEval


@pytest.mark.parametrize('split', ['val', 'test'])
def test_preprocessing_preserves_unannotated_points(tmp_path, monkeypatch, split):
    scene = tmp_path / 'scene0000_00'
    scene.mkdir()
    (scene / f'{scene.name}.txt').write_text(
        'axisAlignment = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\nsceneType = Copy/Mail Room\n')
    (scene / f'{scene.name}_vh_clean_2.0.010000.segs.json').write_text(
        json.dumps({'segIndices': [0, 1, 2]}))
    if split != 'test':
        (scene / f'{scene.name}.aggregation.json').write_text(
            json.dumps({'segGroups': [{'id': 0, 'label': 'chair', 'segments': [0]}]}))
    vertices = np.array([[0., 0., 0., 100., 100., 100.],
                         [1., 0., 0., 100., 100., 100.],
                         [0., 1., 0., 100., 100., 100.]])
    monkeypatch.setattr(preprocess, 'read_plymesh', lambda path: (vertices, np.array([[0, 1, 2]])))
    monkeypatch.setattr(preprocess, 'point_indices_from_group',
                        lambda *args: (vertices[:1], np.array([0]), 2))
    output = tmp_path / 'output'
    preprocess.handle_process(str(scene), str(output), None, [], [scene.name] if split == 'val' else [])
    data = torch.load(output / split / f'{scene.name}.pth', weights_only=False)
    assert data['labels'][1:].tolist() == [-100, -100]
    assert data['instance_ids'][1:].tolist() == [-1, -1]
    if split == 'test':
        assert data['labels'][0] == -100
        assert data['instance_ids'][0] == -1
    else:
        assert data['labels'][0] == preprocess.NORMALIZED_CKASS_IDS_200[2]
        assert data['instance_ids'][0] == 0


def test_official_evaluator_ignores_negative_instance_ids():
    evaluator = BaseEval(class_labels=['chair'], use_label=False, dataset_name='scannet200')
    gt, pred = evaluator.assign_instances_for_scan([], np.full(100, 2), np.full(100, -1))
    assert gt['class_agnostic'] == []
    assert pred['class_agnostic'] == []


def test_official_evaluator_perfect_prediction():
    evaluator = BaseEval(class_labels=['chair'], use_label=False, dataset_name='scannet200')
    gt, pred = evaluator.assign_instances_for_scan(
        [{'scan_id': 'scene', 'label_id': 0, 'conf': .8, 'pred_mask': np.ones(100, dtype=np.uint8)}],
        np.full(100, 2), np.zeros(100, dtype=np.int64))
    ap, recall = evaluator.evaluate_matches({'scene': {'gt': gt, 'pred': pred}})
    np.testing.assert_allclose(ap, 1.)
    np.testing.assert_allclose(recall, 1.)


@pytest.mark.parametrize('predict_absent', [False, True])
def test_absent_ground_truth_classes_are_excluded_from_average(predict_absent):
    evaluator = BaseEval(['chair', 'table'], use_label=True, dataset_name='scannet200')
    predictions = [{'scan_id': 'scene', 'label_id': 1, 'conf': .8, 'pred_mask': np.ones(100, dtype=np.uint8)}]
    if predict_absent:
        predictions.append({'scan_id': 'scene', 'label_id': 2, 'conf': .9, 'pred_mask': np.ones(100, dtype=np.uint8)})
    gt, pred = evaluator.assign_instances_for_scan(predictions, np.full(100, 2), np.zeros(100, dtype=np.int64))
    ap, recall = evaluator.evaluate_matches({'scene': {'gt': gt, 'pred': pred}})
    np.testing.assert_allclose(ap[0, 0], 1.)
    assert np.isnan(ap[0, 1]).all() and np.isnan(recall[0, 1]).all()
    averages = evaluator.compute_averages(ap, recall)
    assert averages['all_ap_50%'] == 1.
    assert averages['all_rc_50%'] == 1.


def test_class_present_elsewhere_in_cohort_still_counts_missing_predictions():
    evaluator = BaseEval(['chair', 'table'], use_label=True, dataset_name='scannet200')
    matches = {}
    for scene, label, predictions in [('chair_scene', 2, [{'scan_id': 'chair_scene', 'label_id': 1, 'conf': .8,
                                                         'pred_mask': np.ones(100, dtype=np.uint8)}]),
                                      ('table_scene', 3, [])]:
        gt, pred = evaluator.assign_instances_for_scan(predictions, np.full(100, label), np.zeros(100, dtype=np.int64))
        matches[scene] = {'gt': gt, 'pred': pred}
    ap, recall = evaluator.evaluate_matches(matches)
    np.testing.assert_allclose(ap[0, 0], 1.)
    np.testing.assert_allclose(ap[0, 1], 0.)
    assert evaluator.compute_averages(ap, recall)['all_ap_50%'] == .5
