"""Independent PEC plane reflection and lossless closed-cavity gates."""
import numpy as np
import pytest
import torch
import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Planar PEC physics requires CUDA")
DX = .6e-3
FREQUENCY = 2.45e9


def _fit(x, field, low, high, k):
    chosen = (x >= low) & (x <= high)
    matrix = np.column_stack((np.exp(1j*k*x[chosen]), np.exp(-1j*k*x[chosen])))
    waves = np.linalg.lstsq(matrix, field[chosen], rcond=None)[0]
    norm = np.linalg.norm(field[chosen])
    return waves, float(np.linalg.norm(matrix @ waves-field[chosen])/norm) if norm else 0.


def reflection(normal="x", offset=0.0, mode="staircase"):
    normal_index = "xyz".index(normal)
    electric_index = (normal_index + 2) % 3
    electric_axis = "xyz"[electric_index]
    axes = [np.arange(-2, 3, dtype=np.float64)*DX for _ in range(3)]
    axes[normal_index] = np.unique(np.r_[np.arange(-50, 51, dtype=np.float64)*DX, offset])
    boundaries = {"xyz"[d]: ("pec" if d == electric_index else "pmc") for d in range(3) if d != normal_index}
    direction, polarization = [0, 0, 0], [0, 0, 0]
    direction[normal_index], polarization[electric_index] = 1, 1
    scene = mw.Scene(
        domain=mw.Domain(bounds=tuple((a[0], a[-1]) for a in axes)),
        grid=mw.GridSpec.custom(*axes),
        boundary=mw.BoundarySpec.faces(default="pml", num_layers=10, **boundaries),
        subpixel_samples=mw.SubpixelSpec(pec=mode, averaging="arithmetic", samples=1), device="cuda",
        structures=[mw.Structure(mw.PlanarPEC.rectangle(((-16*DX, 16*DX),)*2,
                              normal_axis=normal, coordinate=offset, physical_thickness=35e-6),
                              mw.Material.pec(), name="foil")],
        sources=[mw.PlaneWave(direction=direction, polarization=polarization,
                             source_time=mw.CW(frequency=FREQUENCY))])
    prepared = mw.Simulation.fdtd(scene, absorber="cpml", frequency=FREQUENCY,
        run_time=mw.TimeConfig.auto(transient_cycles=12, steady_cycles=12),
        spectral_sampler=mw.SpectralSampler(window="hanning"), full_field_dft=True).prepare()
    result = prepared.run()
    field = getattr(result.E, electric_axis)
    if field.ndim == 4:
        field = field[0]
    selected = [n//2 for n in field.shape]
    selected[normal_index] = slice(None)
    profile = field[tuple(selected)].detach().cpu().numpy()
    x = getattr(prepared.solver.scene, f"{normal}_nodes64")-offset
    dt = prepared.solver.dt
    k = 2/DX*np.arcsin(np.sin(np.pi*FREQUENCY*dt)*DX/(299792458*dt))
    (a, b), left_residual = _fit(x, profile, -.022, -.005, k)
    (c, d), right_residual = _fit(x, profile, .005, .022, k)
    denominator = a*a-d*d
    transmission = (c*a-b*d)/denominator
    reflection = (b*a-c*d)/denominator
    return {"normal": normal, "offset_m": offset, "pec_mode": mode,
            "dt_s": float(dt),
            "r": [float(reflection.real), float(reflection.imag)],
            "t": [float(transmission.real), float(transmission.imag)],
            "abs_t": float(abs(transmission)), "abs_r_plus_one": float(abs(reflection+1)),
            "one_minus_R_minus_T": float(1-abs(reflection)**2-abs(transmission)**2),
            "fit_residual": [left_residual, right_residual],
            "audit": prepared.solver._compiled_material_model["pec_sheet_audit"]}


@pytest.mark.parametrize("normal", ["x", "y", "z"])
@pytest.mark.parametrize("mode", ["staircase", "conformal"])
@pytest.mark.parametrize("offset", [0., DX/2])
def test_planar_pec_reflection_matches_analytic_boundary(normal, offset, mode):
    row = reflection(normal, offset, mode)
    assert row["abs_t"] <= 1e-4, row
    assert row["abs_r_plus_one"] <= .005, row


def cavity_energy(mode, steps):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.2, .2),)*3), grid=mw.GridSpec.uniform(.02),
        boundary=mw.BoundarySpec.pec(), device="cuda",
        subpixel_samples=mw.SubpixelSpec(pec=mode, averaging="arithmetic", samples=1),
        structures=[mw.Structure(mw.PlanarPEC.rectangle(((-.2, .2),)*2, normal_axis="x"),
                                mw.Material.pec())],
        monitors=[mw.FieldTimeMonitor("probe", components=("Ez",), position=(.1, .08, .06))],
        sources=[mw.PointDipole(position=(.12, .09, .11), polarization=(0, 0, 1),
                    source_time=mw.RickerWavelet(frequency=4e9))])
    prepared = mw.Simulation.fdtd(scene, frequency=4e9, run_time=mw.TimeConfig(time_steps=steps)).prepare()
    prepared.run()
    solver = prepared.solver
    electric = sum(float((getattr(solver, c).double()**2).sum()) for c in ("Ex", "Ey", "Ez"))
    # Yee's fields are staggered in time. The conserved leapfrog quadratic uses
    # H[n-1/2]*H[n+1/2], rather than E[n]^2 + H[n-1/2]^2, whose bounded
    # oscillation is not absorption. Evaluate one source-free native field step.
    from witwin.maxwell.fdtd.runtime.stepping import _make_field_update_runner
    previous_h = {c: getattr(solver, c).clone() for c in ("Hx", "Hy", "Hz")}
    _make_field_update_runner(solver, False)(steps * solver.dt)
    magnetic = sum(float((previous_h[c].double()*getattr(solver, c).double()).sum()) for c in previous_h)
    return .5*8.8541878128e-12*electric+.5*1.25663706212e-6*magnetic


@pytest.mark.parametrize("mode", ["staircase", "conformal"])
def test_planar_pec_closed_cavity_has_no_artificial_absorption(mode):
    early, late = cavity_energy(mode, 800), cavity_energy(mode, 6000)
    assert early > 0
    assert abs(late/early-1) < 1e-4
