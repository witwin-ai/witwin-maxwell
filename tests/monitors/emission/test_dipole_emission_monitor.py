"""DipoleEmissionMonitor: Purcell factor / local density of states.

Tier A covers construction, export, and compilation (dipole cell + polarization
resolution) on CPU without a solver. Tier B validates the physics on CUDA.

Normalization choice: the Purcell factor is obtained by dividing the delivered
power of a dipole run by the delivered power of the *same* dipole in vacuum on
the *same* grid. The discrete Yee-grid effective source volume sets the absolute
scale of ``power_delivered`` and has no reliable closed form (the reference
Gaussian source is far narrower than a coarse cell), but it cancels exactly in
the vacuum-normalized ratio. This is more robust than an analytic free-space
formula and is the standard FDTD LDOS approach.
"""

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.monitors import compile_fdtd_observers

F = 1e9
C = 299792458.0
K = 2.0 * np.pi * F / C


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


# ---------------------------------------------------------------------------
# Tier A: construction, export, compilation (CPU, no solver).
# ---------------------------------------------------------------------------


def test_monitor_construction_and_export():
    monitor = mw.monitors.DipoleEmissionMonitor("emit", source_name="d0", frequencies=[F])
    assert monitor.kind == "dipole_emission"
    assert monitor.source_name == "d0"
    assert monitor.frequencies == (F,)
    assert "DipoleEmissionMonitor" in mw.monitors.__all__
    assert mw.monitors.DipoleEmissionMonitor is monitor.__class__


def _emission_payload(power, frequencies=(F, 2 * F)):
    """The FluxData a DipoleEmissionMonitor returns for this delivered power."""
    power = torch.as_tensor(power)
    return mw.FluxData(values=power, dims=("f",), coords={"f": frequencies},
                       attrs={"current_spectrum": torch.ones_like(power)})


def test_purcell_factor_returns_differentiable_tensors():
    structured = torch.tensor([2.0, 0.5], dtype=torch.float64, requires_grad=True)
    vacuum = torch.tensor([1.25, 0.8], dtype=torch.float32)
    data = mw.postprocess.purcell_factor(_emission_payload(structured), _emission_payload(vacuum), "emit")
    for key in ("purcell_factor", "power_delivered", "power_delivered_vacuum"):
        assert isinstance(data[key], torch.Tensor) and data[key].dtype == torch.float64
    assert data["purcell_factor"].requires_grad
    assert data["frequencies"] == (F, 2 * F)
    assert torch.autograd.gradcheck(
        lambda s, v: mw.postprocess.purcell_factor(_emission_payload(s), _emission_payload(v), "emit")["purcell_factor"],
        (structured, vacuum.to(torch.float64).requires_grad_()),
    )


def test_single_frequency_purcell_factor_is_zero_dim_tensor():
    structured = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    data = mw.postprocess.purcell_factor(
        _emission_payload(structured, (F,)), _emission_payload([1.5], (F,)), "emit")
    factor = data["purcell_factor"]
    assert factor.shape == () and factor.item() == 2.0
    factor.backward()
    assert structured.grad.item() == 1.0 / 1.5


def _cpu_scene(pol="Ez", position=(0.1, 0.0, -0.1)):
    scene = mw.Scene(
                domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
                grid=mw.GridSpec.uniform(0.25),
                boundary=mw.BoundarySpec.pml(num_layers=2),
                sources=(
            mw.PointDipole(position=position, polarization=pol, source_time=mw.CW(frequency=F), name="d0"),
            ),
            )
    return scene


def test_compiler_resolves_dipole_cell_and_polarization():
    scene = _cpu_scene(pol="Ez", position=(0.1, 0.0, -0.1))
    scene = scene.updated(monitors=(*scene.monitors, mw.monitors.DipoleEmissionMonitor("emit", source_name="d0")))

    records = compile_fdtd_observers(scene)
    assert len(records) == 1
    record = records[0]
    assert record["monitor_type"] == "dipole_emission"
    assert record["monitor_name"] == "emit"
    assert record["component"] == "ez"
    # The observer is co-located at the dipole cell.
    assert record["position"] == (0.1, 0.0, -0.1)
    assert record["dipole_polarization"] == (0.0, 0.0, 1.0)
    assert record["source_name"] == "d0"


def test_compiler_maps_transverse_polarization():
    scene = _cpu_scene(pol="Ex", position=(0.0, 0.0, 0.0))
    scene = scene.updated(monitors=(*scene.monitors, mw.monitors.DipoleEmissionMonitor("emit", source_name="d0")))

    records = compile_fdtd_observers(scene)
    assert [r["component"] for r in records] == ["ex"]
    assert records[0]["dipole_polarization"] == (1.0, 0.0, 0.0)


def test_compiler_errors_on_missing_source():
    scene = _cpu_scene()
    scene = scene.updated(monitors=(
        *scene.monitors,
        mw.monitors.DipoleEmissionMonitor("emit", source_name="does_not_exist"),
    ))
    with pytest.raises(ValueError, match="does_not_exist"):
        compile_fdtd_observers(scene)


# ---------------------------------------------------------------------------
# Tier B: physics on CUDA.
# ---------------------------------------------------------------------------

_BOUNDS = ((-0.48, 0.48), (-0.48, 0.48), (-0.48, 0.48))
_SPACING = 0.03


def _run_dipole(position, pol, *, pec_z_low=False):
    if pec_z_low:
        boundary = mw.BoundarySpec.faces(default="pml", num_layers=8, strength=1.0, z_low="pec")
    else:
        boundary = mw.BoundarySpec.pml(num_layers=8, strength=1.0)
    scene = mw.Scene(
                domain=mw.Domain(bounds=_BOUNDS),
                grid=mw.GridSpec.uniform(_SPACING),
                boundary=boundary,
                sources=(
            mw.PointDipole(
                position=position,
                polarization=pol,
                source_time=mw.CW(frequency=F, amplitude=1.0),
                name="d",
            ),
            ),
                monitors=(mw.monitors.DipoleEmissionMonitor("emit", source_name="d"),),
            )
    return mw.Simulation.fdtd(
        scene,
        frequencies=[F],
        config=mw.FDTDConfig(
            run_time=mw.TimeConfig.auto(steady_cycles=8, transient_cycles=18),
            spectral_sampler=mw.fdtd.SpectralSampler(window="none"),
            full_field_dft=False,
        ),
    ).run()


def _power(result):
    data = result.monitor("emit")
    assert isinstance(data, mw.FluxData) and data.dims == ("f",)
    power = _to_numpy(data.values).reshape(-1)
    assert power.size == 1
    return float(power[0])


def _perpendicular_analytic(x):
    return 1.0 + 3.0 * (np.sin(x) / x**3 - np.cos(x) / x**2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_vacuum_dipole_purcell_is_unity():
    # Two vacuum runs at different dipole positions must deliver the same power
    # (translational invariance in homogeneous space), so the vacuum-normalized
    # Purcell factor is ~1. This is non-tautological: the reference dipole sits
    # at a different distance from the absorbing boundaries.
    reference = _run_dipole((0.0, 0.0, 0.0), "Ez")
    structured = _run_dipole((2 * _SPACING, _SPACING, -_SPACING), "Ez")

    assert _power(reference) > 0.0
    assert _power(structured) > 0.0

    purcell = mw.postprocess.purcell_factor(structured, reference, "emit")
    assert purcell["frequencies"] == (F,)
    assert isinstance(purcell["purcell_factor"], torch.Tensor)
    assert purcell["purcell_factor"].shape == ()
    assert float(purcell["purcell_factor"]) == pytest.approx(1.0, abs=0.12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_dipole_near_pec_mirror_matches_analytic():
    # Perpendicular dipole above a PEC mirror: the Purcell factor follows the
    # image-dipole curve F = 1 + 3[sin(x)/x^3 - cos(x)/x^2], x = 2 k d. Two
    # separations straddle unity (enhancement then near-free-space).
    vacuum = _run_dipole((0.0, 0.0, 0.0), "Ez")
    z_mirror = _BOUNDS[2][0]

    for separation in (0.06, 0.12):
        mirror = _run_dipole((0.0, 0.0, z_mirror + separation), "Ez", pec_z_low=True)
        purcell = mw.postprocess.purcell_factor(mirror, vacuum, "emit")["purcell_factor"]
        analytic = _perpendicular_analytic(2.0 * K * separation)
        # Discretization-limited: soft-source width, coarse grid, image method.
        assert float(purcell) == pytest.approx(analytic, rel=0.15)

    # The enhancement at the closer separation must exceed free space.
    close = mw.postprocess.purcell_factor(
        _run_dipole((0.0, 0.0, z_mirror + 0.06), "Ez", pec_z_low=True), vacuum, "emit"
    )["purcell_factor"]
    assert close > 1.15
