"""ATen parity and ordered pullbacks for the shared material texture sampler."""
from __future__ import annotations

import gc
from types import SimpleNamespace
import weakref

import pytest
import torch
import torch.nn.functional as F

import witwin.maxwell as mw
from witwin.maxwell.compiler.materials import (
    _sample_box_parameter_field,
    _sample_material_region_density,
    _sample_trilinear_border,
)
from witwin.maxwell.compiler.trilinear import _trilinear_border_plan


_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
_DEVICES = ["cpu", pytest.param("cuda", marks=_CUDA)]
_DTYPES = [torch.float32, torch.float64]


def _oracle(texture, query):
    return F.grid_sample(texture[None, None], query[None], mode="bilinear",
                         padding_mode="border", align_corners=False)[0, 0]


def _queries(shape, device, dtype):
    points = torch.rand((24, 3), device=device, dtype=dtype) * 3.6 - 1.8
    special = [[-1., -1., -1.], [1., 1., 1.], [0., 0., 0.],
               [-2., 0., 2.], [1., -1., 1.], [0.125, -0.25, 0.375]]
    for axis, size in enumerate(reversed(shape)):
        for i in range(size + 1):
            point = [0., 0., 0.]
            point[axis] = 2 * i / size - 1  # Voxel faces.
            special.append(point)
        for i in range(size):
            point = [0., 0., 0.]
            point[axis] = (2 * i + 1) / size - 1  # Voxel centers.
            special.append(point)
    points = torch.cat((points, torch.tensor(special, device=device, dtype=dtype),
                        points[:1].expand(7, 3)))
    return torch.stack((points, points.flip(0)), dim=0).unsqueeze(1)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 3, 4), (7, 5, 9), (1, 3, 1)])
@pytest.mark.parametrize("strided", [False, True])
def test_forward_and_backward_match_aten_and_repeat(device, dtype, shape, strided):
    torch.manual_seed(1401)
    texture = torch.randn(shape, device=device, dtype=dtype)
    query = _queries(shape, device, dtype)
    if strided:
        texture = texture.transpose(0, 2).contiguous().transpose(0, 2)
        query = query.transpose(-1, -2).contiguous().transpose(-1, -2)
    texture.requires_grad_()
    query.requires_grad_()
    expected = _oracle(texture, query)
    grad_out = torch.randn_like(expected)
    reference_grads = torch.autograd.grad(expected, (texture, query), grad_out)
    gradients = []
    for _ in range(3):
        actual = _sample_trilinear_border(texture, query)
        assert actual.shape == query.shape[:-1]
        assert torch.equal(actual, expected), (actual - expected).abs().max().item()
        grads = torch.autograd.grad(actual, (texture, query), grad_out)
        tol = 1e-6 if dtype == torch.float32 else 1e-12
        assert torch.allclose(grads[0], reference_grads[0], rtol=tol, atol=tol)
        assert torch.equal(grads[1], reference_grads[1])
        gradients.append(grads)
    for repeat in gradients[1:]:
        assert all(torch.equal(a, b) for a, b in zip(gradients[0], repeat))


@_CUDA
def test_csr_cache_reuse_mutation_shape_change_and_lifetime():
    query = torch.rand((1, 2, 13, 3), device="cuda")
    first = torch.randn((2, 3, 4), device="cuda", requires_grad=True)
    second = torch.randn_like(first, requires_grad=True)
    a = _sample_trilinear_border(first, query)
    plan = _trilinear_border_plan(first, query)
    b = _sample_trilinear_border(second, query)
    assert _trilinear_border_plan(second, query) is plan
    (a.sum() + b.sum()).backward()
    assert torch.equal(first.grad, second.grad)
    with torch.no_grad():
        query.add_(0.125)
    mutated = _trilinear_border_plan(first, query)
    assert mutated is not plan
    reshaped = _trilinear_border_plan(first[:1], query)
    assert reshaped is not mutated
    rows_ref = weakref.ref(reshaped[0])
    query_ref = weakref.ref(query)
    del a, b, query, plan, mutated, reshaped
    gc.collect()
    assert query_ref() is None
    assert rows_ref() is None


@_CUDA
def test_trainable_query_bypasses_cache_and_double_backward_raises():
    texture = torch.randn((2, 3, 4), device="cuda", requires_grad=True)
    query = torch.rand((1, 2, 5, 3), device="cuda", requires_grad=True)
    assert _trilinear_border_plan(texture, query) is not _trilinear_border_plan(texture, query)
    assert not hasattr(query, "_maxwell_trilinear_plan")
    out = _sample_trilinear_border(texture, query)
    with pytest.raises(RuntimeError, match="does not support double backward"):
        torch.autograd.grad(out.sum(), (texture, query), create_graph=True)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_empty_query(device, dtype):
    texture = torch.randn((2, 1, 4), device=device, dtype=dtype, requires_grad=True)
    query = torch.empty((1, 0, 3, 3), device=device, dtype=dtype, requires_grad=True)
    actual = _sample_trilinear_border(texture, query)
    assert torch.equal(actual, _oracle(texture, query))
    grad_texture, grad_query = torch.autograd.grad(actual.sum(), (texture, query))
    assert torch.equal(grad_texture, torch.zeros_like(texture))
    assert grad_query.shape == query.shape


@_CUDA
@pytest.mark.parametrize("dtype", _DTYPES)
def test_nonfinite_queries(dtype):
    texture = torch.randn((2, 3, 4), device="cuda", dtype=dtype, requires_grad=True)
    query = torch.zeros((1, 1, 6, 3), device="cuda", dtype=dtype)
    for axis in range(3):
        query[0, 0, 2*axis, axis] = float("inf")
        query[0, 0, 2*axis+1, axis] = -float("inf")
    query.requires_grad_()
    actual = _sample_trilinear_border(texture, query)
    expected = _oracle(texture, query)
    assert torch.equal(actual, expected)
    actual_grads = torch.autograd.grad(actual.sum(), (texture, query))
    expected_grads = torch.autograd.grad(expected.sum(), (texture, query))
    assert all(torch.equal(a, b) for a, b in zip(actual_grads, expected_grads))
    with torch.no_grad():
        query[0, 0, 0, 0] = float("nan")
    actual = _sample_trilinear_border(texture, query)
    expected = _oracle(texture, query)
    assert torch.equal(actual, expected)
    actual_grads = torch.autograd.grad(actual.sum(), (texture, query))
    expected_grads = torch.autograd.grad(expected.sum(), (texture, query))
    assert torch.equal(actual_grads[0], expected_grads[0])
    assert torch.equal(actual_grads[1].nan_to_num(), expected_grads[1].nan_to_num())


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("kind", ["parameter", "density"])
def test_both_call_sites_preserve_box_geometry_gradients(device, kind):
    torch.manual_seed(310)
    position = torch.nn.Parameter(torch.tensor([0.03, -0.02, 0.04], device=device))
    size = torch.nn.Parameter(torch.tensor([0.6, 0.7, 0.8], device=device))
    geometry = mw.Box(position=position, size=size, device=device)
    values = torch.rand((3, 4, 2), device=device, requires_grad=True)
    scene = SimpleNamespace(device=torch.device(device))
    coords = torch.meshgrid(*(torch.linspace(-0.2, 0.2, 3, device=device)
                              for _ in range(3)), indexing="ij")
    query = torch.stack(tuple(2 * (c - geometry.position[i]) / geometry.size[i]
                              for i, c in enumerate(coords)), dim=-1)
    expected = _oracle(values.permute(2, 1, 0), query)
    if kind == "parameter":
        actual = _sample_box_parameter_field(scene, geometry, values, coords)
    else:
        actual = _sample_material_region_density(
            scene, SimpleNamespace(geometry=geometry), values, coords=coords)
    assert torch.equal(actual, expected)
    actual_grads = torch.autograd.grad(actual.sum(), (values, position, size))
    expected_grads = torch.autograd.grad(expected.sum(), (values, position, size))
    for i, (a, b) in enumerate(zip(actual_grads, expected_grads)):
        if i == 0:
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
        else:
            assert torch.equal(a, b)
            assert torch.count_nonzero(a)



@_CUDA
@pytest.mark.parametrize("dtype", _DTYPES)
def test_texture_gradient_matches_aten_bitwise_with_at_most_two_contributions(dtype):
    # Consecutive queries along x share one voxel column: every voxel receives at most two
    # corner terms, so ATen's atomic sum is order independent and must match exactly.
    texture = torch.randn((3, 4, 9), device="cuda", dtype=dtype, requires_grad=True)
    w = texture.shape[2]
    centers = (torch.arange(0, w - 1, device="cuda", dtype=dtype) + 0.5 + 0.25) * 2 / w - 1
    query = torch.zeros((1, 1, centers.numel(), 3), device="cuda", dtype=dtype)
    query[0, 0, :, 0] = centers
    query[0, 0, :, 1] = (1.5 + 0.5) * 2 / texture.shape[1] - 1
    query[0, 0, :, 2] = (0.3 + 0.5) * 2 / texture.shape[0] - 1
    grad_out = torch.randn(query.shape[:-1], device="cuda", dtype=dtype)
    actual, = torch.autograd.grad(_sample_trilinear_border(texture, query), texture, grad_out)
    expected, = torch.autograd.grad(_oracle(texture, query), texture, grad_out)
    assert torch.equal(actual, expected)
