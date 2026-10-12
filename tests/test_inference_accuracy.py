"""Regression checks for production accuracy fixes without downloading models."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from libs.inference import accumulate_mask_features, resize_depth_and_intrinsic, score_instances, select_camera_frames
from libs.instance_clustering import FrameEvidence, cluster_graph, frame_mask_evidence
from models import sam2mask
from exp.eval_inst import BaseEval
from tests.test_pipeline import tiny_model

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA is unavailable'))]


def test_camera_selection_covers_position_and_direction_deterministically():
    poses = np.repeat(np.eye(4)[None], 10, axis=0)
    poses[:, 0, 3] = np.arange(10)
    poses[4, :3, 2] = [0, 0, -1]
    poses[9] = np.nan
    reader = SimpleNamespace(get_pose=lambda path: poses[path])
    selected = select_camera_frames(list(range(10)), reader, 3)
    assert selected == select_camera_frames(list(range(10)), reader, 3)
    assert selected == [0, 4, 8]
    assert select_camera_frames(list(range(10)), reader, 0) == list(range(9))
    assert select_camera_frames([], reader, 0) == []
    with pytest.raises(ValueError):
        select_camera_frames([0], reader, -1)


def test_resized_depth_uses_corresponding_pixel_calibration():
    depth = np.ones((4, 8), dtype=np.float32)
    intrinsic = np.array([[4., 0, 3.5], [0, 4., 1.5], [0, 0, 1.]])
    resized, scaled = resize_depth_and_intrinsic(depth, intrinsic, (4, 2))
    assert resized.shape == (2, 4)
    np.testing.assert_allclose(scaled, [[2., 0, 1.5], [0, 2., .5], [0, 0, 1.]])
    assert intrinsic[0, 0] == 4


@pytest.mark.parametrize('device', DEVICES)
def test_feature_fusion_counts_supported_views_instead_of_overlapping_masks(device):
    mapping = torch.tensor([[1, 0, 0, 1], [1, 0, 1, 1], [1, 0, 2, 1]], device=device)
    masks = torch.tensor([[[1, 1, 0]], [[1, 0, 0]]], dtype=torch.bool, device=device)
    embeddings = torch.eye(2, device=device)
    confidence = torch.tensor([.2, .8], device=device)
    features, mass = torch.zeros((3, 2), device=device), torch.zeros((3, 1), device=device)
    accumulate_mask_features(features, mass, masks, embeddings, confidence, mapping, chunk_size=1)
    torch.testing.assert_close(mass[:, 0], torch.tensor([.8, .2, 0.], device=device))
    assert features[0, 1] > features[0, 0]
    assert not features[2].any()
    duplicated, duplicate_mass = torch.zeros_like(features), torch.zeros_like(mass)
    accumulate_mask_features(duplicated, duplicate_mass, masks.repeat(4, 1, 1), embeddings.repeat(4, 1),
                             confidence.repeat(4), mapping)
    torch.testing.assert_close(features, duplicated)
    torch.testing.assert_close(mass, duplicate_mass)


@pytest.mark.parametrize('device', DEVICES)
def test_clip_scoring_uses_temperature_objectness_and_ignores_unknown_features(device):
    features = torch.tensor([[1., 0.], [0., 0.], [0., 1.]], device=device)
    text = torch.eye(2, device=device)
    masks = torch.tensor([[1, 1, 0], [0, 0, 1], [0, 1, 0]], device=device)
    objectness = torch.tensor([.2, .8, .9], device=device)
    adapter = SimpleNamespace(logit_scale=torch.tensor(np.log(100), device=device))
    scores, semantic = score_instances(features, text, masks, objectness, adapter, batch_size=1)
    torch.testing.assert_close(scores, torch.tensor([.2, .8, 0.], device=device))
    assert semantic.tolist() == [0, -100, 1]


@pytest.mark.parametrize('device', DEVICES)
def test_decomposition_fallback_preserves_cannot_link_constraints(device, monkeypatch):
    graph = torch.ones((3, 3), device=device) - torch.eye(3, device=device)
    forbidden = torch.zeros((3, 3), device=device, dtype=torch.bool)
    forbidden[0, 2] = forbidden[2, 0] = True
    evidence = FrameEvidence(graph, forbidden, graph, graph, torch.ones(3, device=device))
    def failure(*args):
        raise torch.linalg.LinAlgError('forced failure')
    monkeypatch.setattr(torch.linalg, 'eigh', failure)
    with pytest.warns(RuntimeWarning, match='constrained average linkage'):
        _, labels, _ = cluster_graph(graph, torch.arange(3, device=device), evidence.confidence,
                                      evidence=evidence)
    assert labels[0] != labels[2]


@pytest.mark.parametrize('model_type', [sam2mask.SAM2Mask])
def test_constructor_vocabulary_is_preserved_when_forward_omits_names(model_type):
    model = tiny_model(model_type)
    model.class_names = ('chair', 'table')
    model.config = SimpleNamespace(clustering=SimpleNamespace(cluster_budget=128))
    calls = []
    model.text_encoder = lambda names: calls.append(names) or torch.eye(2)
    _, confidence, semantic, *_ = model('scene')
    assert list(calls[0]) == ['chair', 'table']
    assert semantic is not None and torch.isfinite(confidence).all()


@pytest.mark.parametrize('device', DEVICES)
def test_clustering_is_deterministic_across_global_rng_states(device):
    graph = torch.tensor([[0., 1.], [1., 0.]], device=device)
    evidence = FrameEvidence(graph, torch.zeros_like(graph, dtype=torch.bool), graph,
                             torch.zeros_like(graph), torch.ones(2, device=device))
    first = cluster_graph(graph, torch.arange(2, device=device), evidence.confidence, evidence)
    torch.manual_seed(839)
    second = cluster_graph(graph, torch.arange(2, device=device), evidence.confidence, evidence)
    for actual, expected in zip(first, second):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('conflicts,occluded,expected', [(2, False, False), (4, False, True), (2, True, True)])
def test_conflict_consensus_counts_only_jointly_supported_views(device, conflicts, occluded, expected):
    labels = torch.arange(2, device=device)
    geometry = torch.tensor([[0., 1.], [1., 0.]], device=device)
    mapping = torch.tensor([[1, 0, 0, 1], [1, 0, 1, 1]], device=device)
    children = torch.tensor([[[1, 1]], [[1, 0]], [[0, 1]]], device=device, dtype=torch.bool)
    parent = children[:1]
    other = mapping.clone()
    if occluded:
        other[1, 3] = 0
    masks = [children] * conflicts + [parent] * (10 - conflicts)
    mappings = [mapping] * conflicts + [other] * (10 - conflicts)
    scores = [torch.tensor([.99, .95, .95], device=device)] * conflicts + [torch.tensor([.99], device=device)] * (10 - conflicts)
    evidence = frame_mask_evidence(labels, masks, mappings, scores, geometry, minimum_conflict_fraction=.25)
    assert bool(evidence.cannot_link[0, 1]) == expected


def test_large_instance_ids_do_not_change_evaluated_semantic_classes():
    evaluator = BaseEval(['chair', 'table'], use_label=True, dataset_name='scannet200')
    semantics = np.repeat([2, 3], 100)
    instances = np.repeat([2049, 2051], 100)
    gt, _ = evaluator.assign_instances_for_scan([], semantics, instances)
    assert len(gt['chair']) == len(gt['table']) == 1
    assert gt['chair'][0]['vert_count'] == gt['table'][0]['vert_count'] == 100
    assert gt['chair'][0]['label_id'] == 1
    assert gt['table'][0]['label_id'] == 2
    np.testing.assert_array_equal(instances, np.repeat([2049, 2051], 100))
