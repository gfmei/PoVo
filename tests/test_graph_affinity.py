"""Synthetic semantic/color boundaries, missing cues, and graph invariants."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from datasets.data_loader import ScanReader
from libs.graph_affinity import (graph_appearance_settings, histogram_rgb_affinity,
                                 select_superpoint_masks, superpoint_label_histograms)
from libs.instance_graph import construct_superpoint_adjacency_graph_radius
from libs.instance_clustering import cluster_graph
from libs.instance_clustering import modulate_geometry
from models import povo, sam2mask
from tests.test_pipeline import TinyReader, tiny_model


DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA is unavailable'))]


@pytest.mark.parametrize('device', DEVICES)
def test_histograms_use_only_global_top_m_masks_ranked_by_overlap(device):
    labels = torch.tensor([7, 7, 7, 11, 30], device=device)
    mapping = torch.tensor([[1, 0, x, int(x < 4)] for x in range(5)], device=device)
    masks = [torch.tensor([[[1, 1, 1, 1, 1, 1]], [[1, 1, 1, 0, 0, 0]],
                            [[1, 1, 0, 0, 0, 0]]], dtype=torch.bool, device=device),
             torch.tensor([[[1, 1, 1, 0, 0, 0]], [[0, 0, 0, 1, 0, 0]]],
                          dtype=torch.bool, device=device)]
    mappings = [mapping, mapping]
    features = [torch.eye(3, device=device), torch.eye(3, device=device)[[2, 0]]]
    scores = [torch.tensor([1., .25, 1.], device=device), torch.tensor([.75, 1.], device=device)]
    frames, selected, overlaps = select_superpoint_masks(labels, masks, mappings, top_m=2)
    assert frames[0].tolist() == [0, 1] and selected[0].tolist() == [1, 0]
    torch.testing.assert_close(overlaps[0], torch.ones(2, device=device))
    assert frames[2].tolist() == [-1, -1] and not overlaps[2].any()
    histogram = superpoint_label_histograms(
        labels, masks, mappings, features, torch.eye(3, device=device), scores,
        top_m=2, top_k=1, logit_scale=1., chunk_size=1)
    torch.testing.assert_close(histogram, torch.tensor([[0., .25, .75], [1., 0., 0.], [0., 0., 0.]], device=device))
    best_only = superpoint_label_histograms(
        labels, masks, mappings, features, torch.eye(3, device=device), scores,
        top_m=1, top_k=1, logit_scale=1.)
    torch.testing.assert_close(best_only[0], torch.tensor([0., 1., 0.], device=device))
    # The broad mask has higher confidence, but a lower IoU and no vote in this node.
    assert histogram[0, 0] == 0


@pytest.mark.parametrize('device', DEVICES)
def test_histogram_keeps_top_k_probabilities_and_is_independent_of_chunk_size(device):
    features = torch.tensor([[1., 0.], [0., 1.], [float('nan'), 0.]], device=device)
    classes = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]], device=device)
    labels = torch.tensor([0, 1, 2], device=device)
    _, _, mapping = graph_inputs(device, 3)
    masks = torch.eye(3, dtype=torch.bool, device=device).reshape(3, 1, 3)
    arguments = (labels, [masks], [mapping], [features], classes, [torch.ones(3, device=device)])
    first = superpoint_label_histograms(*arguments, top_m=1, top_k=2, logit_scale=1., chunk_size=1)
    second = superpoint_label_histograms(*arguments, top_m=1, top_k=2, logit_scale=1., chunk_size=4)
    torch.testing.assert_close(first, second)
    expected = torch.tensor([1., 1 / np.sqrt(2)], device=device).softmax(0).float()
    torch.testing.assert_close(first[0, [0, 2]], expected)
    assert first[0, 1] == first[0, 3] == 0
    assert not first[2].any() and torch.isfinite(first).all()
    all_classes = superpoint_label_histograms(*arguments, top_m=1, top_k=100)
    assert all_classes.shape == (3, 4) and not all_classes[:, 3].any()


@pytest.mark.parametrize('device', DEVICES)
def test_mask_iou_counts_each_projected_pixel_once(device):
    labels = torch.tensor([7, 7, 7, 11], device=device)
    mapping = torch.tensor([[1, 0, 0, 1]] * 3 + [[1, 0, 1, 1]], device=device)
    masks = torch.tensor([[[1, 0, 0]], [[1, 1, 1]]], dtype=torch.bool, device=device)
    frames, selected, overlap = select_superpoint_masks(labels, [masks], [mapping], top_m=2)
    assert frames[0].tolist() == [0, 0] and selected[0].tolist() == [0, 1]
    torch.testing.assert_close(overlap[0], torch.tensor([1., 1 / 3], device=device))
    assert (overlap >= 0).all() and (overlap <= 1).all()


@pytest.mark.parametrize('device', DEVICES)
def test_occluded_and_insufficiently_visible_nodes_cast_no_mask_vote(device):
    labels = torch.tensor([7] * 20 + [11], device=device)
    mapping = torch.tensor([[1, 0, 0, 0]] * 20 + [[1, 0, 1, 1]], device=device)
    mapping[0, 3] = 1
    masks = torch.tensor([[[1, 1]], [[0, 0]]], dtype=torch.bool, device=device)
    frames, selected, overlap = select_superpoint_masks(labels, [masks], [mapping], top_m=2)
    assert frames[0].tolist() == [-1, -1] and not overlap[0].any()
    assert frames[1].tolist() == [0, -1] and selected[1].tolist() == [0, -1]
    torch.testing.assert_close(overlap[1], torch.tensor([.5, 0.], device=device))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('unusable', ['features', 'confidence'])
def test_unusable_selected_mask_does_not_pull_votes_from_below_top_m(device, unusable):
    labels = torch.tensor([7], device=device)
    mapping = torch.tensor([[1, 0, 0, 1]], device=device)
    masks = torch.tensor([[[1, 0, 0]], [[1, 1, 1]]], dtype=torch.bool, device=device)
    features, confidence = torch.eye(2, device=device), torch.ones(2, device=device)
    if unusable == 'features':
        features[0] = 0
    else:
        confidence[0] = 0
    arguments = (labels, [masks], [mapping], [features], torch.eye(2, device=device), [confidence])
    capped = superpoint_label_histograms(*arguments, top_m=1, top_k=1)
    assert not capped.any()
    expanded = superpoint_label_histograms(*arguments, top_m=2, top_k=1)
    torch.testing.assert_close(expanded, torch.tensor([[0., 1.]], device=device))


def graph_inputs(device, count=2):
    points = torch.zeros((count, 3), device=device)
    points[:, 0] = torch.arange(count, device=device) * .005
    mapping = torch.ones((count, 4), dtype=torch.long, device=device)
    mapping[:, 1] = 0
    mapping[:, 2] = torch.arange(count, device=device)
    return points, torch.arange(count, device=device), mapping


def shared_mask_votes(device):
    _, _, mapping = graph_inputs(device)
    return dict(masks_list=[torch.ones((1, 1, 2), dtype=torch.bool, device=device)],
                mappings=[mapping], mask_features_list=[torch.ones((1, 2), device=device)],
                class_features=torch.ones((1, 2), device=device),
                scores_list=[torch.ones(1, device=device)])


@pytest.mark.parametrize('device', DEVICES)
def test_same_predicted_labels_can_connect_different_visual_features(device):
    points, labels, mapping = graph_inputs(device)
    masks = torch.ones((1, 1, 2), dtype=torch.bool, device=device)
    args = (points, labels, [masks], [mapping], [torch.tensor([.9], device=device)])
    features = torch.eye(2, device=device)
    old, _ = construct_superpoint_adjacency_graph_radius(*args, min_points=1, point_feas=features)
    new, _ = construct_superpoint_adjacency_graph_radius(
        *args, min_points=1, point_feas=features, class_features=torch.ones((1, 2), device=device),
        mask_features_list=[torch.ones((1, 2), device=device)],
        appearance_mode='histogram_rgb', semantic_weight=1., visual_weight=0., rgb_weight=0.)
    assert old[0, 1] == 0 and new[0, 1] > .9
    torch.testing.assert_close(new, new.T)


@pytest.mark.parametrize('device', DEVICES)
def test_different_predicted_labels_reduce_edges_even_with_identical_rgb(device):
    points, labels, mapping = graph_inputs(device)
    masks = [torch.ones((1, 1, 2), dtype=torch.bool, device=device),
             torch.eye(2, dtype=torch.bool, device=device).reshape(2, 1, 2)]
    adjacency, _ = construct_superpoint_adjacency_graph_radius(
        points, labels, masks, [mapping, mapping],
        [torch.tensor([.9], device=device), torch.full((2,), .7, device=device)],
        min_points=1, point_feas=torch.ones((2, 2), device=device),
        class_features=torch.eye(2, device=device), point_rgb=torch.zeros((2, 3), device=device),
        mask_features_list=[torch.ones((1, 2), device=device), torch.eye(2, device=device)],
        appearance_mode='histogram_rgb', mask_top_m=1, semantic_top_k=1, semantic_weight=.8,
        visual_weight=0., rgb_weight=.2)
    assert not adjacency.any()


@pytest.mark.parametrize('device', DEVICES)
def test_rgb_separates_color_groups_with_identical_labels_and_masks(device):
    points, labels, mapping = graph_inputs(device, 4)
    colors = torch.tensor([[1., 0., 0.]] * 2 + [[0., 0., 1.]] * 2, device=device)
    adjacency, confidence, evidence = construct_superpoint_adjacency_graph_radius(
        points, labels, [torch.ones((1, 1, 4), dtype=torch.bool, device=device)], [mapping],
        [torch.tensor([.9], device=device)], min_points=1, point_feas=torch.ones((4, 2), device=device),
        class_features=torch.ones((1, 2), device=device), point_rgb=colors,
        appearance_mode='histogram_rgb', semantic_weight=0., visual_weight=0., rgb_weight=1.,
        return_evidence=True)
    assert adjacency[0, 1] > .9 and adjacency[2, 3] > .9
    assert not adjacency[:2, 2:].any() and not adjacency.diagonal().any()
    _, merged, _ = cluster_graph(adjacency, labels, confidence, evidence)
    assert merged[0] == merged[1] and merged[2] == merged[3] and merged[0] != merged[2]


@pytest.mark.parametrize('device', DEVICES)
def test_missing_cues_are_neutral_and_available_weights_are_renormalized(device):
    geometry = torch.tensor([[0., .8], [.8, 0.]], device=device)
    agreement = torch.ones_like(geometry)
    labels = torch.arange(2, device=device)
    missing = histogram_rgb_affinity(geometry, agreement, labels,
                                     point_features=torch.zeros((2, 3), device=device))
    torch.testing.assert_close(missing, geometry)
    valid_rgb = histogram_rgb_affinity(geometry, agreement, labels,
                                       point_features=torch.zeros((2, 3), device=device),
                                       point_rgb=torch.zeros((2, 3), device=device))
    torch.testing.assert_close(valid_rgb, geometry)  # Black is a valid RGB observation.
    unknown_rgb = torch.tensor([[float('nan'), 0., 0.], [1., 1., 1.]], device=device)
    torch.testing.assert_close(histogram_rgb_affinity(
        geometry, agreement, labels, point_rgb=unknown_rgb), geometry)


@pytest.mark.parametrize('device', DEVICES)
def test_similar_appearance_cannot_override_spatial_or_mask_constraints(device):
    points, labels, mapping = graph_inputs(device)
    masks = torch.tensor([[[1, 1]], [[1, 0]], [[0, 1]]], dtype=torch.bool, device=device)
    kwargs = dict(min_points=1, point_feas=torch.ones((2, 2), device=device),
                  class_features=torch.ones((1, 2), device=device),
                  point_rgb=torch.zeros((2, 3), device=device), appearance_mode='histogram_rgb')
    adjacency, _, evidence = construct_superpoint_adjacency_graph_radius(
        points, labels, [masks] * 2, [mapping] * 2,
        [torch.tensor([.85, .95, .95], device=device)] * 2, return_evidence=True,
        mask_features_list=[torch.ones((3, 2), device=device)] * 2, **kwargs)
    assert evidence.cannot_link[0, 1] and not adjacency.any()
    points[1, 0] = 1.
    adjacency, _ = construct_superpoint_adjacency_graph_radius(
        points, labels, [masks[:1]], [mapping], [torch.tensor([.9], device=device)],
        mask_features_list=[torch.ones((1, 2), device=device)], **kwargs)
    assert not adjacency.any()
    adjacency, _ = construct_superpoint_adjacency_graph_radius(
        points, labels, [], [], [], mask_features_list=[], **kwargs)
    assert not adjacency.any()


@pytest.mark.parametrize('model_type', [sam2mask.SAM2Mask, povo.SAM2Mask])
def test_both_pipelines_pass_predicted_vocabulary_and_mesh_rgb_to_graph(monkeypatch, model_type):
    captured, reads = [], []

    class Reader(TinyReader):
        def get_point_colors(self, scene, split):
            reads.append(scene)
            return torch.zeros((4, 3))

    module = sam2mask if model_type is sam2mask.SAM2Mask else povo
    original = module.construct_superpoint_adjacency_graph_radius

    def capture(*args, **kwargs):
        captured.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, 'construct_superpoint_adjacency_graph_radius', capture)
    model = tiny_model(model_type, Reader())
    model.config = SimpleNamespace(graph=SimpleNamespace(appearance_mode='histogram_rgb'))
    model('synthetic', class_names=['chair', 'table'])
    assert reads == ['synthetic']
    assert captured[0]['class_features'].shape == (2, 2)
    assert captured[0]['point_rgb'].shape == (4, 3)
    assert captured[0]['mask_features_list'][0].shape == (2, 2)
    assert captured[0]['mask_top_m'] == 5
    model.config.graph.appearance_mode = 'clip'
    model('synthetic', class_names=['chair', 'table'])
    assert reads == ['synthetic']  # The baseline does not load an extra color tensor.
    assert captured[1]['mask_features_list'] is None


def test_mesh_colors_are_aligned_and_converted_without_semantic_annotations(tmp_path):
    path = tmp_path / 'scannet200/val/scene.pth'
    path.parent.mkdir(parents=True)
    data = {'coords': torch.zeros((2, 3)), 'colors': torch.tensor([[-1., 0., 1.], [1., -1., 0.]])}
    torch.save(data, path)
    reader = ScanReader(str(tmp_path), 'cpu')
    torch.testing.assert_close(reader.get_point_colors('scene', 'val'),
                               torch.tensor([[0., .5, 1.], [1., 0., .5]]))
    del data['colors']
    torch.save(data, path)
    assert reader.get_point_colors('scene', 'val') is None


@pytest.mark.parametrize('settings', [
    {'appearance_mode': 'unknown'}, {'semantic_top_k': 0}, {'semantic_top_k': True},
    {'mask_top_m': 0}, {'mask_top_m': True},
    {'semantic_weight': -1}, {'rgb_sigma': 0}, {'visual_weight': float('nan')},
    {'semantic_weight': 0, 'visual_weight': 0, 'rgb_weight': 0},
])
def test_invalid_affinity_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        graph_appearance_settings(settings)


def test_affinity_does_not_depend_on_the_absolute_scale_of_modality_weights():
    geometry = torch.tensor([[0., 1.], [1., 0.]])
    labels = torch.arange(2)
    options = dict(point_features=torch.ones((2, 2)), visual_weight=0., rgb_weight=0.,
                   **shared_mask_votes('cpu'))
    first = histogram_rgb_affinity(geometry, geometry, labels, semantic_weight=1., **options)
    second = histogram_rgb_affinity(geometry, geometry, labels, semantic_weight=1e-15, **options)
    torch.testing.assert_close(first, second)


@pytest.mark.parametrize('device', DEVICES)
def test_combined_edge_weight_matches_the_configured_cue_formula(device):
    geometry = torch.tensor([[0., .8], [.8, 0.]], device=device)
    agreement = torch.full_like(geometry, .6)
    features = torch.tensor([[1., 0.], [1., 1.]], device=device)
    colors = torch.tensor([[0., 0., 0.], [.1, 0., 0.]], device=device)
    actual = histogram_rgb_affinity(
        geometry, agreement, torch.arange(2, device=device), point_features=features,
        point_rgb=colors, **shared_mask_votes(device))
    visual = (1 + 1 / np.sqrt(2)) / 2
    rgb = np.exp(-.1 ** 2 / (2 * .2 ** 2))
    expected = .8 * .6 * (.5 + .25 * visual + .25 * rgb)
    torch.testing.assert_close(actual[0, 1], torch.tensor(expected, device=device).float())


@pytest.mark.parametrize('device', DEVICES)
def test_visual_only_blend_matches_legacy_weighting_with_missing_features(device):
    geometry = torch.tensor([[0., .8], [.8, 0.]], device=device)
    labels = torch.tensor([7, 7, 11, 11], device=device)
    features = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., 0.]], device=device)
    agreement = torch.ones_like(geometry)
    expected = modulate_geometry(geometry, agreement, features, labels)
    actual = histogram_rgb_affinity(geometry, agreement, labels, point_features=features,
                                    semantic_weight=0., visual_weight=1., rgb_weight=0.)
    torch.testing.assert_close(actual, expected)
