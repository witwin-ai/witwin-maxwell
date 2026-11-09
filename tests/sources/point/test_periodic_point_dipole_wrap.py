"""Point dipoles whose profile wraps across periodic and Bloch faces.

A finite-support dipole profile in a periodic domain is the periodic sum of its
images. The compile folds that sum onto the unique in-domain Yee samples, so a
narrow period yields one patch per component (two for a complex Bloch profile)
instead of one patch per image inside a fixed metric cutoff, and the ``ideal``
delta lands on its periodic representative instead of being clamped onto the
boundary cell.
"""
import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.distributed.source_rebuild import rebuild_ideal_ex_interface_plane


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")

PERIOD = 1.0e-3
STEP = 5.0e-5  # 20 cells per period
FREQUENCY = 60.0e9
CORNER = (0.98e-3, 0.01e-3, 0.0)  # inside the last x cell and the first y cell


def _narrow_scene(source, *, boundary=None, pml_layers=40):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, PERIOD), (0.0, PERIOD), (-2.0e-3, 2.0e-3))),
        grid=mw.GridSpec.uniform(STEP),
        boundary=boundary or mw.BoundarySpec.faces(default="pml", x="periodic", y="periodic", num_layers=pml_layers),
        device="cuda",
    )
    scene.add_source(source)
    return scene


def _dipole(profile, polarization, position=CORNER, amplitude=1.0):
    extra = {"profile": "ideal"} if profile == "ideal" else {"width": 1.0e-4}
    return mw.PointDipole(
        position=position,
        polarization=polarization,
        source_time=mw.GaussianPulse(FREQUENCY, fwidth=30.0e9, amplitude=amplitude),
        name="dipole",
        **extra,
    )


@pytest.mark.parametrize("profile", ("ideal", "gaussian"))
@pytest.mark.parametrize("polarization", ("Ex", "Ez"))
def test_narrow_period_dipole_compiles_one_patch_with_unit_moment(profile, polarization):
    scene = _narrow_scene(_dipole(profile, polarization))
    solver = mw.Simulation.fdtd(scene, frequencies=[FREQUENCY], run_time=mw.TimeConfig(time_steps=1)).prepare().solver
    terms = solver._source_terms
    assert len(terms) == 1
    term = terms[0]
    field = getattr(solver, polarization)
    unique = (field.shape[0] - (polarization != "Ex"), field.shape[1] - (polarization != "Ey"))
    patch = term["patch"].double()
    # Periodic patches stay on unique samples; the source kernel copies node 0
    # to its periodic partner, so the last node is never written directly.
    for axis in range(2):
        assert term["offsets"][axis] + patch.shape[axis] <= unique[axis]
    eps = getattr(solver, "eps_" + polarization)[
        tuple(slice(o, o + n) for o, n in zip(term["offsets"], patch.shape))
    ].double()
    moment = float((patch * eps).sum()) * STEP**3 / (-solver.dt)
    assert moment == pytest.approx(1.0, rel=1e-5)
    if profile == "ideal":
        # x = 0.98 mm lies in the staggered Ex seam gap (and in the last Ez node
        # cell); every bracketing pair is split across samples, never clamped.
        assert torch.count_nonzero(patch).item() == (4 if polarization == "Ex" else 8)


def _sheet_or_dipole_plane(source, *, steps=1600):
    scene = _narrow_scene(source, pml_layers=10)
    scene.add_monitor(mw.PlaneMonitor("far", axis="z", position=1.8e-3, fields=("Ex",)))
    result = mw.Simulation.fdtd(
        scene, absorber="cpml",
        frequencies=[FREQUENCY],
        run_time=mw.TimeConfig(time_steps=steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    ).run()
    return result.monitor("far")["data"].detach().to(torch.complex128).cpu()


@pytest.mark.parametrize("profile", ("ideal", "gaussian"))
def test_narrow_period_dipole_matches_uniform_current_sheet(profile, record_property):
    # A sub-wavelength periodic dipole array radiates only its zeroth Floquet
    # order away from the source plane: the field of a uniform current sheet
    # carrying the same moment per unit area. The evanescent orders decay as
    # exp(-2 pi z / L) ~ 1e-5 at the 1.8 mm probe plane; the plane average is
    # the zeroth order alone.
    dipole = _sheet_or_dipole_plane(_dipole(profile, "Ex"))
    sheet = _sheet_or_dipole_plane(
        mw.UniformCurrentSource(
            size=(PERIOD, PERIOD, STEP),
            center=(0.5 * PERIOD, 0.5 * PERIOD, 0.0),
            polarization="Ex",
            source_time=mw.GaussianPulse(FREQUENCY, fwidth=30.0e9, amplitude=1.0 / (PERIOD * PERIOD * STEP)),
            name="sheet",
        )
    )
    if profile == "gaussian":
        # The Gaussian's spread along z filters the zeroth order by its
        # spectrum at the free-space wavenumber (about 0.8 % for w = 0.1 mm).
        nodes = np.arange(-40, 41) * STEP
        weights = np.exp(-(nodes**2) / (2.0 * 1.0e-4**2))
        wavenumber = 2.0 * math.pi * FREQUENCY / 299792458.0
        sheet = sheet * float(np.sum(weights * np.cos(wavenumber * nodes)) / np.sum(weights))
    reference = sheet.abs().mean().item()
    error = (dipole - sheet).abs().max().item() / reference
    mean_error = abs(dipole.mean().item() - sheet.mean().item()) / reference
    record_property("relative_error", repr(error))
    record_property("mean_relative_error", repr(mean_error))
    print(f"profile={profile} sheet |Ex|={reference:.4e} max rel err={error:.3e} mean rel err={mean_error:.3e}")
    assert reference > 0.0
    assert error < 2e-3
    assert mean_error < 1e-4


SUPERCELL_PERIOD = 0.5
SUPERCELL_STEP = 0.03125
SUPERCELL_TRANSVERSE = ((0.0, 0.25), (0.0, 0.25))
SUPERCELL_FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _supercell_fields(boundary, cells, weights, *, profile, polarization, offset):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, cells * SUPERCELL_PERIOD), *SUPERCELL_TRANSVERSE)),
        grid=mw.GridSpec.uniform(SUPERCELL_STEP),
        boundary=boundary,
        device="cuda",
    )
    extra = {"profile": "ideal"} if profile == "ideal" else {"width": 0.03}
    for index, weight in enumerate(weights):
        if weight == 0.0:
            continue
        x, y, z = offset
        scene.add_source(
            mw.PointDipole(
                position=(x + index * SUPERCELL_PERIOD, y, z),
                polarization=polarization,
                source_time=mw.GaussianPulse(1.0e9, fwidth=0.6e9, amplitude=float(weight)),
                name=f"src{index}",
                **extra,
            )
        )
    scene.add_monitor(mw.PointMonitor("probe", (0.4, 0.125, 0.125), fields=("Ez",)))
    result = mw.Simulation.fdtd(
        scene,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=300),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).run()
    solver = result.solver
    fields = {name: getattr(solver, name).detach().double() for name in SUPERCELL_FIELDS}
    if solver.complex_fields_enabled:
        fields.update({name + "_imag": getattr(solver, name + "_imag").detach().double() for name in SUPERCELL_FIELDS})
    return fields, solver


@pytest.mark.parametrize(
    ("profile", "polarization", "offset"),
    (
        ("gaussian", "Ez", (0.02, 0.11, 0.135)),  # Gaussian window crosses x_lo
        ("gaussian", "Ey", (0.49, 0.11, 0.135)),  # Gaussian window crosses x_hi
        ("ideal", "Ex", (0.005, 0.11, 0.135)),  # delta in the staggered Ex seam gap
        ("ideal", "Ez", (0.49, 0.11, 0.135)),  # delta in the last x cell
    ),
)
def test_wrapped_bloch_dipole_matches_periodic_supercell(profile, polarization, offset, record_property):
    phase = 2.0 * math.pi / 3.0
    cells = 3
    kwargs = {"profile": profile, "polarization": polarization, "offset": offset}
    bloch, solver = _supercell_fields(
        mw.BoundarySpec.faces(default="pec", x="bloch", bloch_wavevector=(phase / SUPERCELL_PERIOD, 0.0, 0.0)),
        1,
        (1.0,),
        **kwargs,
    )
    assert solver.complex_fields_enabled
    assert len(solver._source_terms) == 2  # real and imaginary folded patches
    supercell = mw.BoundarySpec.faces(default="pec", x="periodic")
    cosine, _ = _supercell_fields(supercell, cells, tuple(math.cos(phase * n) for n in range(cells)), **kwargs)
    sine, _ = _supercell_fields(supercell, cells, tuple(math.sin(phase * n) for n in range(cells)), **kwargs)

    error_sq = 0.0
    reference_sq = 0.0
    for name in SUPERCELL_FIELDS:
        actual = torch.complex(bloch[name], bloch[name + "_imag"])
        length = actual.shape[0]
        expected = torch.complex(cosine[name][:length], sine[name][:length])
        error_sq += float((actual - expected).abs().square().sum())
        reference_sq += float(expected.abs().square().sum())
    relative_error = math.sqrt(error_sq / reference_sq)
    record_property("relative_error", repr(relative_error))
    print(f"{profile} {polarization} offset={offset} relative_error={relative_error:.3e}")
    assert reference_sq > 0.0
    assert relative_error < 2e-5


def test_shard_interface_rebuild_keeps_periodic_wrap():
    # On one shard the interface rebuild with global == local indices must
    # reproduce the compiled plane, including the folded periodic y/z factors.
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-2.0e-3, 2.0e-3), (0.0, PERIOD), (0.0, PERIOD))),
        grid=mw.GridSpec.uniform(STEP),
        boundary=mw.BoundarySpec.faces(default="pml", y="periodic", z="periodic", num_layers=40),
        device="cuda",
    )
    scene.add_source(_dipole("ideal", "Ex", position=(0.01e-3, 0.98e-3, 0.99e-3)))
    solver = mw.Simulation.fdtd(scene, frequencies=[FREQUENCY], run_time=mw.TimeConfig(time_steps=1)).prepare().solver
    (term,) = solver._source_terms
    assert term.get("axis_factors") is not None
    source = solver._compiled_sources[0]
    before = term["patch"].clone()
    x_half = np.asarray(solver.scene.x_half64, dtype=np.float64)
    for relative in range(before.shape[0]):
        index = int(term["offsets"][0]) + relative
        if before[relative].abs().max() == 0.0:
            continue
        rebuild_ideal_ex_interface_plane(
            term, source, solver, local_index=index, global_index=index, global_x_half64=x_half
        )
        assert torch.allclose(term["patch"][relative], before[relative], rtol=1e-5, atol=0.0)
