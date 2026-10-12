"""Curve encoders and point-level superpoint boundary discovery."""

import itertools

import pytest
import torch
from hilbertcurve.hilbertcurve import HilbertCurve

from libs.spatial_neighbors import DEFAULT_CURVE_ORDERS, encode_spatial_curve, superpoint_curve_adjacency


DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA is unavailable'))]


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('bits', [1, 2, 3, 10, 21])
@pytest.mark.parametrize('orientation,axes', [('hilbert', (0, 1, 2)), ('hilbert_yzx', (1, 2, 0))])
def test_hilbert_encoder_matches_independent_reference(device, bits, orientation, axes):
    generator = torch.Generator().manual_seed(14)
    cells = torch.randint(0, 1 << bits, (128, 3), generator=generator)
    expected = HilbertCurve(bits, 3).distances_from_points(cells[:, axes].tolist())
    encoded = encode_spatial_curve(cells.to(device), bits, orientation)
    torch.testing.assert_close(encoded.cpu(), torch.tensor(expected), rtol=0, atol=0)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('bits', [1, 2, 3])
def test_hilbert_order_has_continuous_unit_grid_steps(device, bits):
    cells = torch.tensor(list(itertools.product(range(1 << bits), repeat=3)), device=device)
    order = encode_spatial_curve(cells, bits, 'hilbert').argsort()
    step = (cells[order][1:] - cells[order][:-1]).abs().sum(1)
    assert torch.all(step == 1)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('bits', [1, 2, 8, 9, 16, 17, 21])
@pytest.mark.parametrize('orientation,axes', [('morton', (0, 1, 2)),
                                            ('morton_yzx', (1, 2, 0)),
                                            ('morton_zxy', (2, 0, 1))])
def test_morton_encoder_matches_explicit_binary_interleaving(device, bits, orientation, axes):
    if bits <= 2:
        cells = torch.tensor(list(itertools.product(range(1 << bits), repeat=3)))
    else:
        generator = torch.Generator().manual_seed(23)
        random = torch.randint(0, 1 << bits, (128, 3), generator=generator)
        high = (1 << bits) - 1
        cells = torch.cat((random, torch.tensor([[0, 0, 0], [high, high, high],
                                                [high, 0, 0], [0, high, 0], [0, 0, high]])))
    expected = [int(''.join(format(value, f'0{bits}b')[bit] for bit in range(bits) for value in cell), 2)
                for cell in cells[:, axes].tolist()]
    encoded = encode_spatial_curve(cells.to(device), bits, orientation)
    torch.testing.assert_close(encoded.cpu(), torch.tensor(expected), rtol=0, atol=0)
    # Cached LUTs must also preserve the API shape for empty/singleton input.
    empty = encode_spatial_curve(cells[:0].to(device), bits, orientation)
    assert empty.shape == (0,) and empty.dtype == torch.long and empty.device == encoded.device
    torch.testing.assert_close(encode_spatial_curve(cells[:1].to(device), bits, orientation), encoded[:1], rtol=0, atol=0)


@pytest.mark.parametrize('device', DEVICES)
def test_long_superpoints_are_neighbors_when_boundaries_touch_and_centers_are_far(device):
    first = torch.stack((torch.linspace(0, 5, 51), torch.zeros(51), torch.zeros(51)), 1)
    second = first + torch.tensor([5.05, 0., 0.])
    points = torch.cat((first, second)).to(device)
    labels = torch.tensor([4] * 51 + [17] * 51, device=device)
    assert torch.dist(first.mean(0), second.mean(0)) > 5
    adjacency = superpoint_curve_adjacency(points, labels, radius=.1, window_size=1)
    torch.testing.assert_close(adjacency[0, 1], torch.tensor(.05, device=device).neg().exp(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(adjacency, adjacency.T)
    assert not adjacency.diag().any()


@pytest.mark.parametrize('device', DEVICES)
def test_nearness_along_a_curve_cannot_create_a_distant_spatial_edge(device):
    points = torch.tensor([[0., 0., 0.], [5., 0., 0.], [6., 9., 7.]], device=device)
    labels = torch.tensor([0, 2, 9], device=device)
    adjacency = superpoint_curve_adjacency(points, labels, radius=.2, window_size=32)
    assert not adjacency.any()


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('chunk_size', [1, 2, 3, 8192])
def test_overlapping_windows_keep_contacts_across_chunk_boundaries(device, chunk_size):
    points = torch.zeros((20, 3), device=device)
    points[:, 0] = torch.arange(20, device=device) * .01
    labels = torch.tensor([0] * 10 + [1] * 10, device=device)
    adjacency = superpoint_curve_adjacency(points, labels, radius=.02, window_size=2, chunk_size=chunk_size)
    torch.testing.assert_close(adjacency[0, 1], torch.tensor(-.01, device=device).exp())


@pytest.mark.parametrize('device', DEVICES)
def test_multiple_curves_take_union_of_candidates_with_exact_boundary_distance(device):
    generator = torch.Generator().manual_seed(6)
    points = torch.rand((80, 3), generator=generator).to(device)
    labels = torch.arange(80, device=device) % 5
    combined = superpoint_curve_adjacency(points, labels, radius=.3, window_size=2)
    expected = torch.zeros_like(combined)
    for curve in DEFAULT_CURVE_ORDERS:
        single = superpoint_curve_adjacency(points, labels, radius=.3, window_size=2, curve_orders=(curve,))
        expected = torch.maximum(expected, single)
    torch.testing.assert_close(combined, expected)


@pytest.mark.parametrize('device', DEVICES)
def test_curve_proposal_is_checked_against_entire_superpoint_boundaries(device):
    # On this Morton ordering the nearest A/B points have a third label
    # between them. A/B are instead proposed by their distant endpoints.
    # Verification must discover the close contact, rather than reject A/B
    # because those window-selected endpoints are one meter apart.
    points = torch.tensor([[0., 0., 0.], [10., 0., 0.], [0., .01, 0.],
                           [11., 0., 0.], [0., .005, 0.]], device=device)
    labels = torch.tensor([0, 0, 1, 1, 2], device=device)
    adjacency = superpoint_curve_adjacency(points, labels, radius=.1, window_size=1,
                                         resolution=.001, curve_orders=('morton',))
    torch.testing.assert_close(adjacency[0, 1], torch.tensor(-.01, device=device).exp())


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('count', [0, 1, 8])
def test_empty_clouds_and_single_superpoint_have_no_edges(device, count):
    result = superpoint_curve_adjacency(torch.zeros((count, 3), device=device),
                                       torch.zeros(count, dtype=torch.long, device=device))
    assert result.shape == (min(count, 1), min(count, 1))
    assert not result.any()


@pytest.mark.parametrize('kwargs', [{'radius': 0}, {'resolution': 0}, {'window_size': 0},
                                  {'chunk_size': 0}, {'curve_orders': ()},
                                  {'curve_orders': ('bad_curve',)},
                                  {'curve_orders': ('hilbert', 'hilbert')}])
def test_invalid_curve_search_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        superpoint_curve_adjacency(torch.zeros((2, 3)), torch.arange(2), **kwargs)


def test_nonfinite_points_and_coordinate_overflow_are_rejected():
    with pytest.raises(ValueError, match='finite'):
        superpoint_curve_adjacency(torch.tensor([[0., 0., 0.], [float('nan'), 0., 0.]]), torch.arange(2))
    with pytest.raises(ValueError, match='21-bit'):
        superpoint_curve_adjacency(torch.tensor([[0., 0., 0.], [1e6, 0., 0.]]), torch.arange(2))
