"""Raw scalar, density and parameter-field gradient repeatability."""

from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from tests.gradients.scalar_permittivity_scene import ScalarPermittivityPointScene
from tests.gradients.test_fdtd_adjoint_bridge import _DensityTFSFScene, _build_simulation


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class _DeterminismScene(ScalarPermittivityPointScene):
    def __init__(self, kind, material_kind):
        super().__init__()
        self.kind = kind
        self.material_kind = material_kind
        if material_kind != "scalar":
            self.eps_r.requires_grad_(False)
            self.values = torch.nn.Parameter(torch.linspace(
                0.2, 0.8, 24, device="cuda").reshape(2, 3, 4))

    def to_scene(self):
        scene = super().to_scene()
        if self.kind == "tfsf":
            scene = mw.Scene(
                subpixel_samples=scene.subpixel,
                domain=mw.Domain(bounds=((-1.2, 1.2),) * 3),
                grid=mw.GridSpec.uniform(0.12),
                boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
                structures=scene.structures,
                monitors=scene.monitors,
                device="cuda",
            )
            scene.add_source(_DensityTFSFScene._build_source(
                SimpleNamespace(source_kind="plane_wave")))
        elif self.kind == "flux":
            scene.add_monitor(mw.FluxMonitor("flux", axis="z", position=0.24))
        else:
            # Distinct physical coordinates snap to the same Ez cell as probe.
            scene.add_monitor(mw.PointMonitor(
                "coincident", (0.001, 0.001, 0.181), fields=("Ez",)))
        if self.material_kind != "scalar":
            geometry = scene.structures.pop().geometry
            scene.subpixel = mw.SubpixelSpec(samples=1, averaging="arithmetic")
            if self.material_kind == "density":
                scene.add_material_region(mw.MaterialRegion(
                    name="design", geometry=geometry, density=self.values,
                    eps_bounds=(1.0, 6.0)))
            else:
                scene.add_structure(mw.Structure(geometry=geometry,
                    material=mw.PerturbationMedium(
                        mw.Material(eps_r=3.0), perturbation=self.values,
                        eps_sensitivity=0.2)))
        return scene


@pytest.mark.parametrize("kind", ["tfsf", "flux", "coincident_point"])
@pytest.mark.parametrize("material_kind", ["scalar", "density", "parameter"])
def test_adjoint_gradients_are_bitwise_repeatable(kind, material_kind):
    gradients = []
    for _ in range(3):
        model = _DeterminismScene(kind, material_kind)
        result = _build_simulation(model, time_steps=32).run()
        if kind == "flux":
            loss = result.monitor("flux")["flux"].sum()
        else:
            data = result.monitor("probe")["data"]
            loss = data.abs().square().sum()
            if kind == "coincident_point":
                other = result.monitor("coincident")["data"]
                assert torch.equal(data, other), "monitors must snap to the same Yee cell"
                loss = loss + 0.375 * other.abs().square().sum()
        loss.backward()
        parameter = model.eps_r if material_kind == "scalar" else model.values
        gradient = parameter.grad.detach().clone()
        assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
        gradients.append(gradient)
    assert torch.equal(gradients[0], gradients[1])
    assert torch.equal(gradients[0], gradients[2])


@pytest.mark.parametrize("seed_dtype", [torch.float32, torch.float64])
def test_coincident_point_seed_preserves_point_order(seed_dtype):
    from witwin.maxwell.fdtd.adjoint.accumulation import build_point_cell_plan
    from witwin.maxwell.fdtd.cuda.backend import _seed_inject_point, get_compiled_extension

    get_compiled_extension()
    field = torch.zeros((2, 2, 2), device="cuda")
    point_i = torch.tensor([1, 0, 1, 1, 0], device="cuda")
    zero = torch.zeros_like(point_i)
    cells, offsets, order = build_point_cell_plan(point_i, zero, zero, field.shape)
    # Cancellation distinguishes ascending point order from regrouping terms.
    real = torch.tensor([[1e8, 3, -1e8, 1, 2]], dtype=seed_dtype, device="cuda")
    imag = torch.zeros_like(real)
    cos = torch.ones((1, 1), dtype=seed_dtype, device="cuda")
    for _ in range(3):
        field.zero_()
        _seed_inject_point(
            AdjField=field, GradReal=real, GradImag=imag, Cells=cells,
            CellOffsets=offsets, PointOrder=order, CosPack=cos,
            SinPack=torch.zeros_like(cos), step=0)
        assert field[1, 0, 0].item() == 1.0
        assert field[0, 0, 0].item() == 5.0


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 3, 4)])
def test_collocation_gather_matches_autograd_transpose(shape):
    from witwin.maxwell.fdtd.adjoint.core import _collocated_field_square
    from witwin.maxwell.fdtd.cuda.backend import _collocation_transpose, get_compiled_extension

    get_compiled_extension()
    nx, ny, nz = shape
    torch.manual_seed(120)
    fields = tuple(torch.randn(s, device="cuda", requires_grad=True) for s in (
        (nx, ny + 1, nz + 1), (nx + 1, ny, nz + 1), (nx + 1, ny + 1, nz)))
    grads = tuple(torch.randn_like(f) for f in fields)
    squares = _collocated_field_square(*fields)
    expected = torch.autograd.grad(tuple(squares.values()), fields, grad_outputs=grads)
    initial = tuple(torch.randn_like(f) for f in fields)
    outputs = []
    for _ in range(3):
        actual = tuple(f.clone() for f in initial)
        _collocation_transpose(
            AdjEx=actual[0], AdjEy=actual[1], AdjEz=actual[2],
            GEx=grads[0], GEy=grads[1], GEz=grads[2],
            Ex=fields[0], Ey=fields[1], Ez=fields[2])
        for value, seed, reference in zip(actual, initial, expected):
            torch.testing.assert_close(value, seed + reference, rtol=2e-6, atol=2e-6)
        outputs.append(actual)
    assert all(torch.equal(a, b) for a, b in zip(outputs[0], outputs[1]))
    assert all(torch.equal(a, b) for a, b in zip(outputs[0], outputs[2]))


@pytest.mark.parametrize("kind", ["scalar", "x", "y", "z", "interpolated", "single_sample"])
def test_tfsf_sample_gather_matches_ordered_reference(kind):
    from witwin.maxwell.fdtd.adjoint.accumulation import build_tfsf_sample_plan
    from witwin.maxwell.fdtd.cuda.backend import (
        _accumulate_tfsf_sample_adjoint, get_compiled_extension,
    )

    get_compiled_extension()
    shape = (2, 3, 4)
    torch.manual_seed(33)
    coeff = torch.randn(shape)
    adj = torch.randn(shape)
    term = {"coeff_patch": coeff, "scalar_sample_index": None}
    count = 1 if kind == "single_sample" else 5
    if kind in {"interpolated", "single_sample"}:
        term["sample_positions"] = torch.linspace(-0.5, 3.5, coeff.numel()).reshape(shape)
    elif kind == "scalar":
        term["scalar_sample_index"] = 2
    else:
        axis = "xyz".index(kind)
        term["sample_view"] = tuple(shape[d] if d == axis else 1 for d in range(3))
        # Duplicate destinations exercise the case the old line atomics raced.
        term["sample_indices"] = torch.tensor([1, 1, 3, 2][:shape[axis]])
    origin, ds = (0.1, 0.3) if kind == "interpolated" else (0.25, 0.5)
    cpu_plan = build_tfsf_sample_plan(term, sample_count=count, origin=origin, ds=ds)
    gpu_term = {key: value.cuda() if isinstance(value, torch.Tensor) else value
                for key, value in term.items()}
    gpu_plan = build_tfsf_sample_plan(gpu_term, sample_count=count, origin=origin, ds=ds)
    for cpu, gpu in zip(cpu_plan, gpu_plan):
        assert torch.equal(cpu, gpu.cpu())
    initial = torch.randn(count)
    expected = initial.clone()
    offsets, cells, weights = cpu_plan
    values = (-0.75 * adj * coeff).flatten()
    for sample in range(count):
        for entry in range(int(offsets[sample]), int(offsets[sample + 1])):
            expected[sample] += values[cells[entry]] * weights[entry]
    gpu_adj, gpu_coeff = adj.cuda(), coeff.cuda()
    for _ in range(3):
        actual = initial.cuda()
        _accumulate_tfsf_sample_adjoint(
            AdjAuxField=actual, AdjFieldPatch=gpu_adj, CoeffPatch=gpu_coeff,
            RowOffsets=gpu_plan[0], CellIndices=gpu_plan[1], Weights=gpu_plan[2],
            componentScale=-0.75)
        assert torch.equal(actual.cpu(), expected)
