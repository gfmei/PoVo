"""Behavioral checks for quality spectral clustering and mandatory constraints."""

import pytest
import torch

from libs.instance_clustering import (
    FrameEvidence, cluster_graph, frame_mask_evidence,
    normalized_cut, quality_spectral_labels, kmeans, constrained_edge_linkage,
)

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA is unavailable'))]


@pytest.mark.parametrize('device', DEVICES)
def test_cut_quality_cannot_be_improved_by_leaving_cluster_ids_unused(device):
    adjacency = torch.ones((4, 4), device=device) - torch.eye(4, device=device)
    compact = torch.tensor([0, 0, 1, 1], device=device)
    with_holes = torch.tensor([0, 0, 7, 7], device=device)
    torch.testing.assert_close(normalized_cut(adjacency, compact), normalized_cut(adjacency, with_holes))


@pytest.mark.parametrize('device', DEVICES)
def test_lloyd_rounding_retains_unequal_cluster_sizes(device):
    points = torch.tensor([[[0., 0.], [.01, 0.], [.02, 0.], [.03, 0.],
                            [.04, 0.], [10., 0.], [10.01, 0.]]], device=device)
    gamma, _, _ = kmeans(points, 2)
    labels = gamma.argmax(-1)[0]
    assert torch.unique(labels[:5]).numel() == 1
    assert torch.unique(labels[5:]).numel() == 1
    assert labels[0] != labels[-1]
    single, _, center = kmeans(points[:, :2], 1, weights=torch.tensor([[1., 3.]], device=device))
    torch.testing.assert_close(center[0, 0], torch.tensor([.0075, 0.], device=device))
    assert (single == 1).all()




@pytest.mark.parametrize('device', DEVICES)
def test_frame_vote_is_invariant_to_duplicate_masks(device):
    labels = torch.arange(2, device=device)
    mapping = torch.tensor([[1, 0, 0, 1], [1, 0, 1, 1]], device=device)
    shared = torch.ones((1, 1, 2), device=device, dtype=torch.bool)
    separate = torch.eye(2, device=device, dtype=torch.bool).reshape(2, 1, 2)
    geometry = torch.tensor([[0., 1.], [1., 0.]], device=device)
    first = frame_mask_evidence(labels, [shared, separate], [mapping, mapping],
                                [torch.tensor([.9], device=device), torch.tensor([.9, .9], device=device)], geometry)
    second = frame_mask_evidence(labels, [shared.repeat(9, 1, 1), separate], [mapping, mapping],
                                 [torch.full((9,), .9, device=device), torch.tensor([.9, .9], device=device)], geometry)
    for name in first.__dataclass_fields__:
        torch.testing.assert_close(getattr(first, name), getattr(second, name))
    torch.testing.assert_close(first.agreement[0, 1], torch.tensor(.5, device=device))
    assert not first.cannot_link.any()


@pytest.mark.parametrize('device', DEVICES)
def test_occluded_superpoint_casts_no_negative_vote(device):
    labels = torch.arange(2, device=device)
    mapping = torch.tensor([[1, 0, 0, 1], [1, 0, 1, 0]], device=device)
    separate = torch.eye(2, device=device, dtype=torch.bool).reshape(2, 1, 2)
    geometry = torch.tensor([[0., 1.], [1., 0.]], device=device)
    evidence = frame_mask_evidence(labels, [separate, separate], [mapping, mapping],
                                   [torch.full((2,), .9, device=device)] * 2, geometry)
    assert not evidence.agreement.any()
    assert not evidence.negative_votes.any()
    assert not evidence.cannot_link.any()


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('weight', [1., .05])
@pytest.mark.parametrize('fraction', [0., .5])
def test_repeated_exclusive_masks_block_merging_through_an_intermediate(device, weight, fraction):
    labels = torch.arange(3, device=device)
    mapping = torch.tensor([[1, 0, 0, 1], [1, 0, 1, 1], [1, 0, 2, 1]], device=device)
    # A broad parent overlaps all three nodes. Its two reliable children
    # repeatedly show that the end nodes belong to different objects.
    masks = torch.tensor([[[1, 1, 1]], [[1, 0, 0]], [[0, 0, 1]]], device=device, dtype=torch.bool)
    geometry = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]], device=device)
    evidence = frame_mask_evidence(labels, [masks] * 2, [mapping] * 2,
                                   [torch.tensor([.85, .95, .95], device=device)] * 2, geometry,
                                   minimum_conflict_fraction=fraction)
    assert evidence.cannot_link[0, 2]
    _, merged, _ = cluster_graph(geometry, labels, torch.full((3,), .9, device=device),
                                 evidence=evidence, non_neighbor_weight=weight)
    assert merged[0] != merged[2]


@pytest.mark.parametrize('device', DEVICES)
def test_quality_selection_splits_weak_bridge_with_consistent_view_evidence(device):
    adjacency = torch.block_diag(torch.ones((4, 4)), torch.ones((4, 4))).to(device)
    adjacency.fill_diagonal_(0)
    adjacency[3, 4] = adjacency[4, 3] = .02
    same = torch.arange(8, device=device)[:, None] // 4 == torch.arange(8, device=device)[None, :] // 4
    positive = same.float() * 3
    positive.fill_diagonal_(0)
    negative = (~same).float() * 3
    evidence = FrameEvidence(torch.zeros_like(adjacency), negative > 1, positive, negative, torch.ones(8, device=device))
    labels = quality_spectral_labels(adjacency, evidence)
    assert torch.unique(labels[:4]).numel() == 1
    assert torch.unique(labels[4:]).numel() == 1
    assert labels[0] != labels[-1]
    # The same graph with unanimous single-object masks should remain whole.
    evidence.positive_votes = torch.ones_like(adjacency) * 3
    evidence.positive_votes.fill_diagonal_(0)
    evidence.negative_votes.zero_()
    whole = quality_spectral_labels(adjacency, evidence)
    assert torch.unique(whole).numel() == 1


@pytest.mark.parametrize('device', DEVICES)
def test_observed_edge_counts_are_accumulated_after_each_union(device):
    sums = torch.tensor([[0., 9., 0.], [9., 0., 5.], [0., 5., 0.]], device=device)
    counts = torch.tensor([[0., 10., 1.], [10., 0., 5.], [1., 5., 0.]], device=device)
    forbidden = torch.zeros((3, 3), device=device, dtype=torch.bool)
    original_sums, original_counts = sums.clone(), counts.clone()
    merged = constrained_edge_linkage(sums, counts, forbidden, .8)
    assert torch.unique(merged).numel() == 1
    torch.testing.assert_close(sums, original_sums)
    torch.testing.assert_close(counts, original_counts)
    assert not forbidden.any()
    forbidden[0, 2] = forbidden[2, 0] = True
    blocked = constrained_edge_linkage(sums, counts, forbidden, .8)
    assert blocked[0] != blocked[2]


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('size,expected_groups', [(3, 1), (20, 2)])
def test_weighted_spatial_evidence_can_join_parts_but_keeps_large_weak_bridges(
        monkeypatch, device, size, expected_groups):
    from libs import instance_clustering
    count = size * 2
    fine = torch.arange(count, device=device) // size
    monkeypatch.setattr(instance_clustering, 'quality_spectral_labels', lambda *args: fine.clone())
    adjacency = torch.block_diag(torch.ones((size, size)), torch.ones((size, size))).to(device)
    adjacency.fill_diagonal_(0)
    adjacency[size - 1, size] = adjacency[size, size - 1] = .9
    confidence = torch.ones(count, device=device)
    evidence = FrameEvidence(adjacency, torch.zeros_like(adjacency, dtype=torch.bool), adjacency,
                             torch.zeros_like(adjacency), confidence, adjacency > 0)
    _, legacy, _ = cluster_graph(adjacency, torch.arange(count, device=device), confidence, evidence)
    assert torch.unique(legacy).numel() == 2
    _, merged, _ = cluster_graph(adjacency, torch.arange(count, device=device), confidence, evidence,
                                 non_neighbor_weight=.05)
    assert torch.unique(merged).numel() == expected_groups


@pytest.mark.parametrize('device', DEVICES)
def test_weighted_merge_handles_unknown_edges_without_creating_connections(device):
    sums = torch.zeros((3, 3), device=device)
    counts = torch.zeros_like(sums)
    merged = constrained_edge_linkage(sums, counts, torch.zeros_like(sums, dtype=torch.bool))
    assert torch.unique(merged).numel() == 3


@pytest.mark.parametrize('weight', [-.1, 1.1, float('nan'), float('inf'), True])
def test_invalid_non_neighbor_weights_are_rejected(weight):
    with pytest.raises(ValueError, match='non_neighbor_weight'):
        cluster_graph(torch.empty((0, 0)), torch.empty(0, dtype=torch.long), torch.empty(0), None,
                      non_neighbor_weight=weight)
