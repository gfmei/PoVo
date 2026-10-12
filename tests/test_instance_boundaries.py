"""Instance boundaries must survive proximity, occlusion, and merge chains."""

import pytest
import torch

from libs import lib_cluster
from libs.instance_graph import construct_superpoint_adjacency_graph_radius
from libs.instance_clustering import FrameEvidence, cluster_graph, constrained_average_linkage


DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA is unavailable'))]


def graph(points, labels, masks, mappings, *, device, features=None, scores=None, **kwargs):
    return construct_superpoint_adjacency_graph_radius(
        points.to(device), labels.to(device), [mask.to(device) for mask in masks],
        [mapping.to(device) for mapping in mappings],
        scores or [torch.full((len(mask),), .9, device=device) for mask in masks],
        point_feas=None if features is None else features.to(device),
        min_points=1, **kwargs,
    )


@pytest.mark.parametrize('device', DEVICES)
def test_adjacent_objects_with_different_masks_do_not_connect(device):
    points = torch.tensor([[0., 0., 1.], [.05, 0., 1.]])
    mapping = torch.tensor([[1, 0, 0, 1], [1, 0, 1, 1]])
    masks = torch.eye(2, dtype=torch.bool).reshape(2, 1, 2)
    adjacency, confidence = graph(points, torch.arange(2), [masks], [mapping], device=device)
    assert not adjacency.any()
    torch.testing.assert_close(confidence, torch.full_like(confidence, .9))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('missing_features', [False, True])
def test_shared_mask_connects_same_object_but_contradictory_features_veto_it(device, missing_features):
    points = torch.tensor([[0., 0., 1.], [.05, 0., 1.]])
    mapping = torch.tensor([[1, 0, 0, 1], [1, 0, 1, 1]])
    features = torch.zeros((2, 2)) if missing_features else torch.tensor([[1., 0.], [-1., 0.]])
    adjacency, _ = graph(points, torch.arange(2), [torch.ones((1, 1, 2), dtype=torch.bool)],
                         [mapping], device=device, features=features, sim_thres=.8)
    assert bool(adjacency[0, 1] > 0) == missing_features
    torch.testing.assert_close(adjacency, adjacency.T)


@pytest.mark.parametrize('device', DEVICES)
def test_partial_visibility_does_not_discard_mask_membership(device):
    points = torch.tensor([[0., 0., 1.]] * 10 + [[.05, 0., 1.]] * 10)
    labels = torch.tensor([0] * 10 + [1] * 10)
    mapping = torch.zeros((20, 4), dtype=torch.long)
    mapping[[0, 1, 10, 11], 3] = 1
    adjacency, confidence = graph(points, labels, [torch.ones((1, 1, 1), dtype=torch.bool)],
                                 [mapping], device=device, mask_thres=.3)
    assert adjacency[0, 1] > .9
    torch.testing.assert_close(confidence, torch.full_like(confidence, .9))


@pytest.mark.parametrize('device', DEVICES)
def test_occlusion_in_another_frame_is_not_mask_disagreement(device):
    points = torch.tensor([[0., 0., 1.], [.05, 0., 1.]])
    only_first_visible = torch.tensor([[1, 0, 0, 1], [1, 0, 1, 0]])
    both_visible = torch.tensor([[1, 0, 0, 1], [1, 0, 1, 1]])
    adjacency, _ = graph(points, torch.arange(2), [torch.tensor([[[True, False]]]),
                        torch.ones((1, 1, 2), dtype=torch.bool)],
                        [only_first_visible, both_visible], device=device)
    assert adjacency[0, 1] > .9
    torch.testing.assert_close(adjacency, adjacency.T)


@pytest.mark.parametrize('device', DEVICES)
def test_one_object_can_cover_more_than_ten_superpoints(device):
    count = 14
    points = torch.zeros((count, 3))
    points[:, 0] = torch.arange(count) * .005
    mapping = torch.ones((count, 4), dtype=torch.long)
    mapping[:, 1] = 0
    mapping[:, 2] = torch.arange(count)
    adjacency, confidence = graph(points, torch.arange(count),
                                 [torch.ones((1, 1, count), dtype=torch.bool)],
                                 [mapping], device=device)
    assert (adjacency.sum(1) > 0).all()
    torch.testing.assert_close(confidence, torch.full_like(confidence, .9))


@pytest.mark.parametrize('device', DEVICES)
def test_a_stray_mask_pixel_does_not_create_a_confident_instance(device):
    points = torch.zeros((10, 3))
    mapping = torch.ones((10, 4), dtype=torch.long)
    mapping[:, 1] = 0
    mapping[:, 2] = torch.arange(10)
    masks = torch.zeros((1, 1, 10), dtype=torch.bool)
    masks[0, 0, 0] = True
    adjacency, confidence = graph(points, torch.zeros(10, dtype=torch.long), [masks],
                                 [mapping], device=device, mask_thres=.3)
    assert not adjacency.any()
    assert not confidence.any()


@pytest.mark.parametrize('device', DEVICES)
def test_sparse_boundary_contact_does_not_merge_whole_objects(device):
    # Two internally coherent objects each have four superpoints. They touch
    # at one strong edge, but their overall inter-object affinity is only .05.
    adjacency = torch.block_diag(torch.ones((4, 4)), torch.ones((4, 4))).to(device)
    adjacency.fill_diagonal_(0)
    adjacency[3, 4] = adjacency[4, 3] = .8
    labels = torch.tensor([0] * 4 + [1] * 4, device=device)
    weights = lib_cluster.compute_cluster_edge_weights_optimized(adjacency, labels, 2)
    merged = constrained_average_linkage(weights, torch.tensor([4, 4], device=device), torch.zeros_like(weights, dtype=torch.bool), .1)
    assert merged[0] != merged[1]


@pytest.mark.parametrize('device', DEVICES)
def test_merge_chain_requires_agreement_with_the_entire_merged_group(device):
    # A-B and B-C are strong, A-C has no support. A connected-components merge
    # would swallow all three; average linkage must retain two groups.
    affinity = torch.tensor([[0., .9, 0.], [.9, 0., .8], [0., .8, 0.]], device=device)
    labels = constrained_average_linkage(affinity, torch.ones(3, device=device), torch.zeros_like(affinity, dtype=torch.bool), .5)
    assert labels[0] == labels[1]
    assert labels[1] != labels[2]
    forbidden = torch.zeros_like(affinity, dtype=torch.bool)
    forbidden[0, 1] = forbidden[1, 0] = True
    blocked = constrained_average_linkage(affinity, torch.ones(3, device=device), forbidden, .5)
    assert blocked[0] != blocked[1]


@pytest.mark.parametrize('device', DEVICES)
def test_disconnected_and_unobserved_superpoints_are_not_one_instance(device):
    adjacency = torch.zeros((5, 5), device=device)
    adjacency[0, 1] = adjacency[1, 0] = 1
    adjacency[2, 3] = adjacency[3, 2] = 1
    labels = lib_cluster.split_disconnected_clusters(adjacency, torch.zeros(5, dtype=torch.long, device=device))
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert labels[4] != labels[0] and labels[4] != labels[2]


@pytest.mark.parametrize('device', DEVICES)
def test_automatic_spectral_clustering_does_not_fragment_coherent_objects(device):
    # Asking for 128 clusters must not force two dense objects into fragments.
    adjacency = torch.block_diag(torch.ones((5, 5)), torch.ones((4, 4)), torch.zeros((1, 1))).to(device)
    adjacency.fill_diagonal_(0)
    evidence = FrameEvidence(adjacency, torch.zeros_like(adjacency, dtype=torch.bool), adjacency,
                             torch.zeros_like(adjacency), torch.ones(10, device=device))
    points, superpoints, _ = cluster_graph(adjacency, torch.arange(10, device=device), evidence.confidence, evidence, budget=128)
    assert torch.unique(points[:5]).numel() == 1
    assert torch.unique(points[5:9]).numel() == 1
    assert points[0] != points[5] and points[9] != points[0] and points[9] != points[5]
    torch.testing.assert_close(points, superpoints)


@pytest.mark.parametrize('device', DEVICES)
def test_no_eigendecomposition_fallback_can_swallow_a_weak_bridge(device, monkeypatch):
    adjacency = torch.tensor([[0., .9, 0., 0.], [.9, 0., .01, 0.],
                              [0., .01, 0., .9], [0., 0., .9, 0.]], device=device)
    labels = torch.arange(4, device=device)
    confidence = torch.full((4,), .9, device=device)
    same = torch.arange(4, device=device)[:, None] // 2 == torch.arange(4, device=device)[None, :] // 2
    positive = same.float() * 3
    positive.fill_diagonal_(0)
    negative = (~same).float() * 3
    evidence = FrameEvidence(adjacency, negative > 1, positive, negative, confidence)
    _, spectral, spectral_scores = cluster_graph(adjacency, labels, confidence, evidence, merge_threshold=.5)

    def failure(*args):
        raise torch.linalg.LinAlgError('forced failure')
    monkeypatch.setattr(torch.linalg, 'eigh', failure)
    with pytest.warns(RuntimeWarning, match='constrained average linkage'):
        _, fallback, fallback_scores = cluster_graph(adjacency, labels, confidence, evidence, merge_threshold=.5)
    for result, scores in [(spectral, spectral_scores), (fallback, fallback_scores)]:
        assert result[0] == result[1] and result[2] == result[3]
        assert result[0] != result[2]
        torch.testing.assert_close(scores, torch.full_like(scores, .9))
