from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")

_FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _boundary_spec(boundary_kind):
    if boundary_kind in ("pml", "pml_slab"):
        return mw.BoundarySpec.pml(num_layers=4, strength=1.0)
    if boundary_kind == "pml_mixed":
        # PEC x with PML y/z forces the boundary-active full CPML electric
        # kernels instead of the all-faces-PML interior fast path.
        return mw.BoundarySpec.faces(default="pml", x="pec", num_layers=4, strength=1.0)
    # The bloch case uses a zero wavevector: the Bloch phase derives from the
    # domain span, which is dl longer on a uniform scene (arange end-exclusive
    # nodes) than the custom scene built from its node extents, so any nonzero
    # wavevector describes different physics. Zero phase keeps the runs
    # bitwise comparable while still exercising the complex Bloch kernels.
    # (A mixed x/y-Bloch + z-PML case is not constructible here: mixed specs
    # require a nonzero wavevector, which breaks the span equivalence above.)
    if boundary_kind == "bloch":
        return mw.BoundarySpec.bloch((0.0, 0.0, 0.0))
    return mw.BoundarySpec(kind=boundary_kind)


def _cpml_config(boundary_kind):
    if boundary_kind == "pml_slab":
        return {"memory_mode": "slab"}
    return None


def _dipole_scene(domain_bounds, grid, boundary_kind):
    scene = mw.Scene(
        domain=mw.Domain(bounds=domain_bounds),
        grid=grid,
        boundary=_boundary_spec(boundary_kind),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.1,
            source_time=mw.CW(frequency=1.0e9, amplitude=1.0),
            name="src",
        )
    )
    return scene


def _run_fields(scene, cpml_config=None):
    prepared = mw.Simulation.fdtd(
        scene,
        frequency=1.0e9,
        cpml_config=cpml_config,
        run_time=mw.TimeConfig(time_steps=40),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).prepare()
    prepared.solver.solve(time_steps=40, dft_frequency=None, dft_window="none", full_field_dft=False)
    fields = {}
    for name in _FIELDS:
        fields[name] = getattr(prepared.solver, name).clone()
        imag = getattr(prepared.solver, f"{name}_imag", None)
        if imag is not None:
            fields[f"{name}_imag"] = imag.clone()
    return fields


@pytest.mark.parametrize(
    "boundary_kind",
    ["periodic", "pec", "pml", "pml_mixed", "pml_slab", "bloch"],
)
def test_custom_constant_grid_matches_uniform_bitwise(boundary_kind):
    # A uniform grid expressed as GridSpec.custom with the uniform scene's own
    # float64 node masters must reproduce the GridSpec.uniform run bitwise:
    # the kernels load the identical constant spacing arrays, dt and source
    # placement derive from the same masters through a single code path.
    dl = 0.05
    bounds = ((-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4))
    uniform_scene = _dipole_scene(bounds, mw.GridSpec.uniform(dl), boundary_kind)
    prepared_uniform = prepare_scene(uniform_scene)

    custom_grid = mw.GridSpec.custom(
        prepared_uniform.x_nodes64,
        prepared_uniform.y_nodes64,
        prepared_uniform.z_nodes64,
    )
    custom_bounds = tuple(
        (float(nodes[0]), float(nodes[-1]))
        for nodes in (
            prepared_uniform.x_nodes64,
            prepared_uniform.y_nodes64,
            prepared_uniform.z_nodes64,
        )
    )
    custom_scene = _dipole_scene(custom_bounds, custom_grid, boundary_kind)

    cpml_config = _cpml_config(boundary_kind)
    uniform_fields = _run_fields(uniform_scene, cpml_config)
    custom_fields = _run_fields(custom_scene, cpml_config)

    assert set(uniform_fields) == set(custom_fields)
    for name, uniform_value in uniform_fields.items():
        assert torch.isfinite(uniform_value).all(), name
        assert torch.equal(uniform_value, custom_fields[name]), name
    for name in _FIELDS:
        assert uniform_fields[name].abs().sum() > 0.0, name


class _DesignScene(mw.SceneModule):
    """Point-dipole scene with a (2, 2, 2) design region for gradient checks."""

    def __init__(self, grid, bounds):
        super().__init__()
        self._grid = grid
        self._bounds = bounds
        self.logits = torch.nn.Parameter(torch.zeros((2, 2, 2), device="cuda"))

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=self._bounds),
            grid=self._grid,
            boundary=mw.BoundarySpec(kind="pec"),
            device="cuda",
        )
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(0.06, 0.06, 0.18), size=(0.24, 0.24, 0.24)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(0.0, 0.0, -0.18),
                polarization="Ez",
                width=0.04,
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=50.0),
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.18), fields=("Ez",)))
        return scene


def _adjoint_gradient(model):
    result = mw.Simulation.fdtd(
        model,
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=80),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).run()
    data = result.monitor("probe")["data"]
    loss = (data * data.conj()).real if data.is_complex() else data * data
    loss.backward()
    return model.logits.grad.detach().clone()


def test_custom_constant_grid_matches_uniform_adjoint_gradient():
    # The adjoint reverse pass must consume the same per-axis spacing arrays as
    # the forward pass: a custom grid built from the uniform scene's masters
    # must reproduce the uniform design gradient.
    dl = 0.12
    bounds = ((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))
    uniform_model = _DesignScene(mw.GridSpec.uniform(dl), bounds).cuda()
    prepared_uniform = prepare_scene(uniform_model.to_scene())

    custom_grid = mw.GridSpec.custom(
        prepared_uniform.x_nodes64,
        prepared_uniform.y_nodes64,
        prepared_uniform.z_nodes64,
    )
    custom_bounds = tuple(
        (float(nodes[0]), float(nodes[-1]))
        for nodes in (
            prepared_uniform.x_nodes64,
            prepared_uniform.y_nodes64,
            prepared_uniform.z_nodes64,
        )
    )
    custom_model = _DesignScene(custom_grid, custom_bounds).cuda()

    grad_uniform = _adjoint_gradient(uniform_model)
    grad_custom = _adjoint_gradient(custom_model)

    assert grad_uniform.abs().max() > 0.0
    torch.testing.assert_close(grad_custom, grad_uniform, rtol=1.0e-6, atol=0.0)
