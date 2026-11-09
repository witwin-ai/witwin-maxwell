"""Perfect magnetic conductor (PMC) structures: the magnetic dual of PEC."""
import cmath
import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box
from witwin.maxwell.compiler.materials import _pmc_occupancy
from witwin.maxwell.fdtd.adjoint.bridge import _unsupported_adjoint_medium
from witwin.maxwell.fdtd.runtime.materials import average_node_to_magnetic_component
from witwin.maxwell.scene import prepare_scene

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD/FDFD")


def _cpu_scene(*structures):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.3, 0.3), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.05),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cpu",
    )
    for structure in structures:
        scene.add_structure(structure)
    return scene


def test_pmc_material_construction_and_validation():
    pmc = mw.Material.pmc(name="wall")
    assert pmc.is_pmc and not pmc.is_pec and pmc.name == "wall"
    assert not mw.Material.pec().is_pmc
    with pytest.raises(ValueError, match="both a perfect electric and a perfect magnetic"):
        mw.Material(pec=True, pmc=True)
    with pytest.raises(ValueError, match="PMC Material must not carry"):
        mw.Material(pmc=True, eps_r=2.0)
    with pytest.raises(ValueError, match="PMC Material must not carry"):
        mw.Material(pmc=True, sigma_m=1.0)
    with pytest.raises(ValueError, match="cannot wrap a PMC"):
        mw.PerturbationMedium(pmc, perturbation=torch.zeros(2, 2, 2))


def test_pmc_structure_compiles_to_magnetic_occupancy_not_bulk_material():
    box = Box(position=(0.075, 0.0, 0.0), size=(0.15, 0.2, 0.2))
    vacuum = prepare_scene(_cpu_scene()).compile_materials()
    model = prepare_scene(_cpu_scene(mw.Structure(geometry=box, material=mw.Material.pmc()))).compile_materials()
    assert vacuum["pmc_occupancy"] is None
    assert model["pec_occupancy"] is None
    occupancy = model["pmc_occupancy"]
    assert occupancy is not None and occupancy.shape == model["eps_r"].shape
    # A PMC marker is not a bulk dielectric: the electric/magnetic tensors stay vacuum.
    assert torch.equal(model["eps_r"], vacuum["eps_r"])
    assert torch.equal(model["mu_r"], vacuum["mu_r"])
    assert not model["lorentz_poles"]


def test_pmc_staircase_holds_h_components_inside_the_box():
    # The box face x = 0.0 sits on a node plane; the H component half a cell
    # inside (x = 0.025) is covered, the one half a cell outside (x = -0.025) is open.
    box = Box(position=(0.075, 0.0, 0.0), size=(0.15, 0.2, 0.2))
    prepared = prepare_scene(_cpu_scene(mw.Structure(geometry=box, material=mw.Material.pmc())))
    occupancy = _pmc_occupancy(prepared)
    hy = average_node_to_magnetic_component(None, occupancy, "Hy") >= 0.5
    x_half = 0.5 * (prepared.x[:-1] + prepared.x[1:])
    y_index = int(torch.argmin(prepared.y.abs()))
    z_half = 0.5 * (prepared.z[:-1] + prepared.z[1:])
    z_index = int(torch.argmin(z_half.abs()))
    inside = int(torch.argmin((x_half - 0.025).abs()))
    outside = int(torch.argmin((x_half + 0.025).abs()))
    assert bool(hy[inside, y_index, z_index])
    assert not bool(hy[outside, y_index, z_index])
    assert not bool(hy[int(torch.argmin((x_half - 0.175).abs())), y_index, z_index])


def test_fdtd_adjoint_rejects_trainable_pmc_geometry():
    position = torch.tensor([0.075, 0.0, 0.0], requires_grad=True)
    box = Box(position=position, size=(0.15, 0.2, 0.2))
    scene = _cpu_scene(mw.Structure(geometry=box, material=mw.Material.pmc()))
    scene.subpixel = mw.SubpixelSpec(samples=2, averaging="polarized")
    reason = _unsupported_adjoint_medium(scene)
    assert reason is not None and "PMC" in reason
    fixed = _cpu_scene(mw.Structure(geometry=Box(position=(0.075, 0.0, 0.0), size=(0.15, 0.2, 0.2)),
                                    material=mw.Material.pmc()))
    assert _unsupported_adjoint_medium(fixed) is None


# ---------------------------------------------------------------------------
# GPU physics checks (small: < 1 GB, a few tens of seconds)
# ---------------------------------------------------------------------------

_F = 1.0e9
_DX = 0.0075
_WALL = 0.15
_PROBES = (-0.15, -0.1125, -0.075)


def _plane_wave_run(material):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.45, 0.45), (-0.015, 0.015), (-0.015, 0.015))),
        grid=mw.GridSpec.uniform(_DX),
        device="cuda",
        boundary=mw.BoundarySpec.faces(default="none", x="pml", y="periodic", z="periodic", num_layers=12),
    )
    if material is not None:
        scene.add_structure(mw.Structure(geometry=Box(position=(_WALL + 0.25, 0.0, 0.0), size=(0.5, 0.1, 0.1)),
                                         material=material))
    scene.add_source(mw.PlaneWave(direction=(1.0, 0.0, 0.0), polarization=(0.0, 0.0, 1.0),
                                  source_time=mw.CW(_F, amplitude=1.0)))
    positions = (*_PROBES, _WALL, _WALL + 0.5 * _DX)
    for index, x in enumerate(positions):
        scene.add_monitor(mw.PointMonitor(f"p{index}", (x, 0.0, 0.0), fields=("Ez", "Hy")))
    result = mw.Simulation.fdtd(
        scene, absorber="cpml", frequency=_F, full_field_dft=False,
        run_time=mw.TimeConfig.auto(transient_cycles=14, steady_cycles=8),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
    ).run()

    def value(index, component):
        return complex(torch.as_tensor(result.monitor(f"p{index}")[component]).reshape(-1)[0])

    ez = [value(i, "Ez") for i in range(len(positions))]
    hy = [value(i, "Hy") for i in range(len(positions))]
    return ez, hy


@_CUDA
def test_fdtd_pmc_slab_is_the_pec_dual_under_plane_wave():
    incident, incident_h = _plane_wave_run(None)
    pec, pec_h = _plane_wave_run(mw.Material.pec())
    pmc, pmc_h = _plane_wave_run(mw.Material.pmc())
    k = cmath.phase(incident[1] / incident[0]) / (_PROBES[1] - _PROBES[0])

    def reflection(total, wall):
        return [
            (total[j] - incident[j]) / incident[j] * cmath.exp(-2j * k * (wall - x))
            for j, x in enumerate(_PROBES)
        ]

    # PEC shorts tangential E on the face node plane; PMC holds tangential H on the
    # first dual plane inside the face, half a cell deeper.
    r_pec = reflection(pec, _WALL)
    r_pmc = reflection(pmc, _WALL + 0.5 * _DX)
    for a, b in zip(r_pec, r_pmc):
        assert abs(abs(a) - 1.0) < 1e-3
        assert abs(abs(b) - 1.0) < 1e-3
        # Dual reflection: r_PMC = -r_PEC (a pi phase flip).
        assert abs(b / a + 1.0) < 1e-2

    # Dual wall fields: E_t = 0 on the PEC wall and H_t = 0 on the PMC wall, while
    # the free component doubles (standing-wave antinode).
    assert pec[3] == 0j
    assert pmc_h[4] == 0j
    assert abs(abs(pmc[3]) / abs(incident[3]) - 2.0) < 1e-2
    assert abs(abs(pec_h[3]) / abs(incident_h[3]) - 2.0) < 2e-2


def _dipole_scene(material):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.24, 0.24), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=4.0),
        device="cuda",
    )
    if material is not None:
        scene.add_structure(mw.Structure(geometry=Box(position=(0.12, 0.0, 0.0), size=(0.16, 0.2, 0.2)),
                                         material=material))
    scene.add_source(mw.PointDipole(position=(-0.04, 0.0, 0.0), polarization="Ez", width=0.02,
                                    source_time=mw.CW(1.5e9, amplitude=1.0)))
    return scene


_DIPOLE_PROBES = ((-0.08, 0.04, 0.0), (0.02, -0.06, 0.02), (-0.12, 0.0, 0.06))


def _probe_values(result, scene):
    prepared = prepare_scene(scene)
    ez = torch.as_tensor(result.tensor("Ez"))

    def index(nodes, value):
        return int(torch.argmin((torch.as_tensor(nodes).cpu().double() - value).abs()))

    return [complex(ez[index(prepared.x, x), index(prepared.y, y), index(prepared.z, z)])
            for x, y, z in _DIPOLE_PROBES]


def _fdtd_dipole(material):
    scene = _dipole_scene(material)
    result = mw.Simulation.fdtd(
        scene, frequency=1.5e9, full_field_dft=True,
        run_time=mw.TimeConfig.auto(transient_cycles=14, steady_cycles=8),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
    ).run()
    return _probe_values(result, scene)


def _fdfd_dipole(material):
    scene = _dipole_scene(material)
    result = mw.Simulation.fdfd(scene, frequency=1.5e9,
                                solver=mw.GMRES(solver_type="direct", precision="double")).run()
    return _probe_values(result, scene)


@_CUDA
def test_fdtd_pmc_matches_good_magnetic_conductor_limit():
    # A volumetric magnetic conductivity far above the skin-depth limit is the
    # physical PMC limit, just as a large sigma_e approaches PEC.
    vacuum = _fdtd_dipole(None)
    pmc = [a / b for a, b in zip(_fdtd_dipole(mw.Material.pmc()), vacuum)]
    eta0_sq = 376.730313668 ** 2
    good = [a / b for a, b in zip(_fdtd_dipole(mw.Material(sigma_m=2.0e4 * eta0_sq)), vacuum)]
    for a, b in zip(pmc, good):
        assert abs(a - b) / abs(b) < 1e-3
    pec = [a / b for a, b in zip(_fdtd_dipole(mw.Material.pec()), vacuum)]
    assert max(abs(a - b) / abs(b) for a, b in zip(pmc, pec)) > 0.2


@_CUDA
def test_fdfd_pmc_matches_fdtd_pmc():
    fdtd_vacuum = _fdtd_dipole(None)
    fdfd_vacuum = _fdfd_dipole(None)
    for material, tolerance in ((mw.Material.pec(), 5e-2), (mw.Material.pmc(), 5e-2)):
        fdtd = [a / b for a, b in zip(_fdtd_dipole(material), fdtd_vacuum)]
        fdfd = [a / b for a, b in zip(_fdfd_dipole(material), fdfd_vacuum)]
        for a, b in zip(fdfd, fdtd):
            assert abs(a - b) / abs(b) < tolerance
