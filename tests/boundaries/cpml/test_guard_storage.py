"""CPU storage and time-level controls for the CPML energy guard."""
from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.runtime.passivity import (
    SAMPLE_INTERVAL, WINDOW_SAMPLES, make_passivity_state, total_energy, uncounted_storage,
)
from witwin.maxwell.fdtd.copper_foil import advance_copper_electric
from witwin.maxwell.fdtd.runtime.stepping import _field_update_block, enforce_pec_boundaries
from witwin.maxwell.fdtd.thin_sheet import (
    _copper_rhs_torch, _chord_forcing_torch, _copper_state_torch,
    _sheet_flux, _prepare_sheet_flux, _SheetRuntime, thin_sheet_energy, thin_sheet_energy_device,
)
from tests.materials.thin_sheet import test_thin_sheet_runtime as cavity
from tests.materials.thin_sheet.harness import rectangle_mesh
from tests.boundaries.cpml.test_auto_passivity import host_preparation
from witwin.maxwell.fdtd.runtime import passivity


@pytest.fixture(autouse=True)
def _sample_deferred_storage(monkeypatch):
    # v0.6.0 skips guarded copper/sheet scenes; this module qualifies the
    # counted functional that dense sampling will use once it is affordable.
    monkeypatch.setattr(passivity, "DEFERRED_STORAGE", ())


def _flux_plan(matrix):
    import scipy.sparse as sp
    runtime = _SheetRuntime.__new__(_SheetRuntime)
    runtime.device = torch.device("cpu")
    return _prepare_sheet_flux(runtime, sp.csr_matrix(matrix))


@pytest.mark.parametrize("complex_fields", [False, True])
def test_foil_guard_counts_modes_and_both_traces(host_preparation, complex_fields, monkeypatch):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.1, .1),)*3), grid=mw.GridSpec.uniform(.025), device="cpu",
        boundary=mw.BoundarySpec.faces(x="bloch" if complex_fields else "pec", y="pec", z="pml",
                                      num_layers=2, bloch_wavevector=(1., 0., 0.) if complex_fields else (0., 0., 0.)),
        structures=[mw.Structure(geometry=mw.Box(position=(0., 0., 0.), size=(.1, 0., .1)),
                                 material=mw.CopperFoil(thickness=2e-6, frequency_range=(.5e9, 2e9)))])
    solver = mw.Simulation.fdtd(scene, frequency=1e9, absorber="cpml",
                                run_time=mw.TimeConfig(time_steps=1)).prepare().solver
    assert solver._copper_foils
    assert uncounted_storage(solver) == []
    assert make_passivity_state(solver) is not None
    assert solver.passivity_guard_status == "active"
    expected = 0.
    for i, entry in enumerate(solver._copper_foils):
        entry["state"].fill_(i + 1)
        entry["right_e"].fill_(i + 2)
        getattr(solver, entry["electric_name"])[entry["index"]] = i + 3
        expected += .5*((entry["state"].square()*entry["area"]*entry["admittance_scale"]).sum()
                         + (entry["area"]*(entry["capacity"][0]*(i+3)**2
                                           + entry["capacity"][1]*entry["right_e"].double().square())).sum())

    def host_read(*args, **kwargs):
        raise AssertionError("foil energy reduction read device data on the host")

    with monkeypatch.context() as patch:
        for name in ("cpu", "numpy", "item", "__float__", "__bool__"):
            patch.setattr(torch.Tensor, name, host_read)
        energy = total_energy(solver)
    assert energy.dtype == torch.float64 and energy.device.type == "cpu"
    torch.testing.assert_close(energy, expected, rtol=3e-15, atol=0)
    assert any(e["imaginary"] for e in solver._copper_foils) == complex_fields


@pytest.mark.parametrize("material", ["pec", "copper"])
@pytest.mark.parametrize("modified", [False, True])
def test_device_sheet_functional_matches_qualified_reference(material, modified, monkeypatch):
    foil = (mw.Material.pec() if material == "pec" else
            mw.CopperFoil(thickness=2e-6, frequency_range=(.5e9, 4e9)))
    kwargs = dict(sheet_model="conformal_two_sided") if material == "pec" else {}
    solver = cavity._run(cavity._cavity(cavity._plate(foil, **kwargs)), 80).solver
    expected = thin_sheet_energy(solver, modified=modified)
    assert expected > 0
    assert uncounted_storage(solver) == []
    # First use prepares the diagnostic operators; steady sampling is device-only.
    thin_sheet_energy_device(solver, modified=modified)

    def host_read(*args, **kwargs):
        raise AssertionError("energy reduction read device data on the host")

    # Exercise the complete reduction, including sparse flux recovery, under a
    # host-read prohibition. Setup and the independent reference precede it.
    with monkeypatch.context() as patch:
        for name in ("cpu", "numpy", "item", "__float__", "__bool__"):
            patch.setattr(torch.Tensor, name, host_read)
        actual = thin_sheet_energy_device(solver, modified=modified)
        guarded = total_energy(solver) if modified else actual
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64), rtol=4e-14, atol=0)
    torch.testing.assert_close(guarded, actual, rtol=0, atol=0)


def test_explicit_cpml_copper_sheet_guard_is_active():
    scene = cavity._cavity(cavity._plate(mw.CopperFoil(thickness=2e-6, frequency_range=(.5e9, 4e9))))
    scene.boundary = mw.BoundarySpec.faces(x="pec", y="pec", z="pml", num_layers=2)
    solver = mw.Simulation.fdtd(scene, frequency=2e9, backend="cpu_reference", absorber="cpml",
                                run_time=mw.TimeConfig(time_steps=80)).run().solver
    assert solver.passivity_guard_status == "active"
    assert solver._passivity_state is not None
    assert uncounted_storage(solver) == []
    torch.testing.assert_close(total_energy(solver),
                               torch.tensor(thin_sheet_energy(solver, modified=True), dtype=torch.float64),
                               rtol=4e-14, atol=0)


def _mixed_copper_scene(boundary):
    copper = mw.CopperFoil(thickness=2e-6, frequency_range=(.5e9, 4e9))
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.09, .09),)*3), grid=mw.GridSpec.uniform(.01),
        boundary=boundary, device="cpu", structures=[
            mw.Structure(name="planar", geometry=mw.Box(position=(-.04, 0., 0.), size=(0., .04, .04)),
                         material=copper),
            mw.Structure(name="sheet", geometry=rectangle_mesh((.04, .003, .002), (1., .4, .25), .012, .01),
                         material=copper)])


def test_mixed_guard_keeps_native_and_sheet_widths_independent(monkeypatch):
    scene = _mixed_copper_scene(mw.BoundarySpec.faces(x="pec", y="pec", z="pml", num_layers=2))
    solver = mw.Simulation.fdtd(scene, frequency=2e9, backend="cpu_reference", absorber="cpml",
                                run_time=mw.TimeConfig(time_steps=1)).prepare().solver
    assert solver._copper_foils and solver._thin_sheets
    assert make_passivity_state(solver) is not None
    assert solver.passivity_guard_status == "active"
    assert uncounted_storage(solver) == []
    native, sheet = solver._passivity_widths, solver._thin_sheet_energy_widths
    assert native is not sheet
    for axis in "xyz":
        torch.testing.assert_close(sheet[axis, "dual"][[0, -1]], native[axis, "dual"][[0, -1]]/2)
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        getattr(solver, name).fill_(.25)
    for entry in solver._copper_foils:
        entry["right_e"].fill_(.75)
    expected = total_energy(solver)
    # A mixed scene's base field integral and foil mass/cross replacements
    # must all use the sheet widths, even if the native cache changes.
    solver._passivity_widths = {key: value*3 for key, value in native.items()}

    def host_read(*args, **kwargs):
        raise AssertionError("mixed energy reduction read device data on the host")

    with monkeypatch.context() as patch:
        for name in ("cpu", "numpy", "item", "__float__", "__bool__"):
            patch.setattr(torch.Tensor, name, host_read)
        actual = total_energy(solver)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert make_passivity_state(solver) is not None
    assert solver._thin_sheet_energy_widths is sheet
    torch.testing.assert_close(total_energy(solver), expected, rtol=0, atol=0)


@pytest.mark.parametrize("loss_rate", [0., 1e9], ids=["lossless", "lossy"])
def test_mixed_copper_cavity_energy_plus_heat_is_constant(loss_rate, monkeypatch):
    """Two separate conductor owners in one closed, source-free CPU cavity.

    Use two passive canonical copper modes, as in the chord balance control,
    and float64 prepared tensors to isolate the discrete work identity from
    float32 material rounding. Native FP64 admission is unchanged.
    """
    from witwin.maxwell.scene import prepare_scene

    def prepare_double(scene, **kwargs):
        return prepare_scene(scene, **{**kwargs, "material_dtype": torch.float64})

    response = SimpleNamespace(
        rates=torch.tensor([0., loss_rate], dtype=torch.float64),
        state_space=SimpleNamespace(B=torch.tensor([[50., 50.], [35., -35.]], dtype=torch.float64),
                                    D=torch.zeros((2, 2), dtype=torch.float64)))
    monkeypatch.setattr("witwin.maxwell.simulation.prepare_scene", prepare_double)
    monkeypatch.setattr("witwin.maxwell.fdtd.copper_foil.compile_copper_response", lambda _: response)
    monkeypatch.setattr("witwin.maxwell.compiler.copper_foil.compile_copper_response", lambda _: response)
    solver = mw.Simulation.fdtd(_mixed_copper_scene(mw.BoundarySpec.pec()), frequency=2e9,
                                backend="cpu_reference", run_time=mw.TimeConfig(time_steps=300)).prepare().solver
    assert solver.Ex.dtype == torch.float64
    assert len(solver._thin_sheets) == 1 and solver._copper_foils
    generator = torch.Generator().manual_seed(901)
    for name in ("Ex", "Ey", "Ez"):
        getattr(solver, name).normal_(generator=generator)
    runtime = solver._thin_sheets[0]
    runtime.scatter(solver, runtime.plan_all_e, runtime.e_extra,
                    torch.zeros(runtime.plan_all_e["size"], dtype=torch.float64))
    enforce_pec_boundaries(solver)
    for entry in solver._copper_foils:
        entry["right_e"].copy_(getattr(solver, entry["electric_name"])[entry["index"]])
    initial = total_energy(solver)
    balances = []
    for _ in range(300):
        _field_update_block(solver, 0.)
        enforce_pec_boundaries(solver)
        foil_heat = sum(entry["cumulative_loss"] for entry in solver._copper_foils)
        balances.append(total_energy(solver) + runtime.loss + foil_heat)
    residual = (torch.stack(balances) - initial).abs().max()/initial
    assert residual < 1e-13
    assert abs(balances[-1]-balances[99])/initial < 1e-13
    assert runtime.copper["state"].abs().max() > 0
    assert all(entry["state"].abs().max() > 0 for entry in solver._copper_foils)
    if loss_rate:
        assert runtime.loss > 0 and foil_heat > 0
    else:
        assert runtime.loss == 0 and foil_heat == 0
    print(dict(mixed_loss_rate=loss_rate, relative_energy_heat_residual=float(residual),
               sheet_heat=float(runtime.loss), foil_heat=float(foil_heat)))


def _resonator(loss_rate):
    """One two-sided chord, one sheet face, and two copper diffusion modes.

    This is the zero-bulk-curl subproblem of the production Newmark update.
    All masses are normalized to isolate float64 temporal round-off from
    native float32 material coefficients and spatial discretization.
    """
    tensor = lambda x: torch.tensor(x, dtype=torch.float64)
    dt = .08
    rates, B, D = tensor([0., loss_rate]), tensor([[.8, .8], [.6, -.6]]), torch.zeros((2, 2), dtype=torch.float64)
    decay = (1 - dt*rates/2)/(1 + dt*rates/2)
    drive = dt/2*B/(1 + dt*rates/2)[:, None]
    effective = D + B.T@drive
    Ks = tensor([[1., -1.]])
    empty = torch.zeros(0, dtype=torch.float64)
    rt = SimpleNamespace(
        copper=dict(count=1, mass=tensor([[1., 1.]]), conductance=tensor([[0., 0.]]),
                    area=tensor([1.]), history=B.T*((1+decay)/2)[None, :], effective=effective,
                    Acx=torch.zeros((2, 0), dtype=torch.float64), Acc=Ks.T@Ks,
                    state=tensor([[.3], [-.2]]), decay=decay, drive=drive, rates=rates, D=D,
                    promoted_position=torch.zeros(0, dtype=torch.int64),
                    promoted_chords=torch.zeros(0, dtype=torch.int64)),
        chord_e=tensor([[1., -.4]]), chord_e_prev=tensor([[1., -.4]]),
        chord_f=tensor([[.2, -.2]]), x_prev=empty.clone(), e_old=empty.clone(), e_new=empty.clone(),
        e_extra=empty.clone(), h_extra=empty.clone(), feeds=[], loss=tensor(0.),
        plan_all_e=None, plan_all_h={"native": []},
        energy=dict(e_mass=empty, h_mass=empty, dual=empty, lh=empty,
                    K=torch.zeros((0, 0), dtype=torch.float64), K_patch=torch.zeros((0, 0), dtype=torch.float64),
                    Kc=torch.zeros((0, 2), dtype=torch.float64), KcT=torch.zeros((2, 0), dtype=torch.float64),
                    Ks=Ks, flux=_flux_plan(Ks.T.numpy()), root_lh=tensor([1.])))
    rt.gather = lambda *args: empty
    solver = SimpleNamespace(device="cpu", dt=dt, mu0=1., complex_fields_enabled=False, _thin_sheets=[rt],
                             _passivity_widths={(a, k): tensor([1.]) for a in "xyz" for k in ("primal", "dual")})
    # Zero native fields with actual Yee staggering permit the global modified
    # cross term to run, while only the independent chord carries energy.
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        axis = "xyz".index(name[1].lower())
        shape = [2 if (a != axis) == name.startswith("E") else 1 for a in range(3)]
        setattr(solver, name, torch.zeros(shape, dtype=torch.float64))
        setattr(solver, ("eps_" if name.startswith("E") else "mu_")+name, 1.)
    solver._passivity_widths.update({(a, "dual"): tensor([.5, .5]) for a in "xyz"})
    solver._thin_sheet_energy_widths = solver._passivity_widths.copy()
    return solver, rt


@pytest.mark.parametrize("loss_rate", [0., .15])
def test_modified_copper_resonance_energy_plus_heat_is_constant(loss_rate):
    solver, rt = _resonator(loss_rate)
    cu, dt = rt.copper, solver.dt
    S = torch.eye(2, dtype=torch.float64)/dt + cu["effective"]/2 + dt/4*cu["Acc"]
    initial = total_energy(solver)
    values, plain, energies, heats = [], [], [], []
    for _ in range(600):
        _, rhs = _copper_rhs_torch(rt, solver, rt.e_old, rt.e_new)
        updated = torch.linalg.solve(S, rhs)
        _chord_forcing_torch(rt, solver, rt.e_old, rt.e_new, updated)
        _copper_state_torch(rt, solver, rt.e_old, rt.e_new, updated,
                            torch.empty((0, 2), dtype=torch.float64), torch.empty((0, 2), dtype=torch.float64))
        energy = total_energy(solver)
        energies.append(energy)
        heats.append(rt.loss.clone())
        values.append(energy + rt.loss)
        plain.append(thin_sheet_energy_device(solver) + rt.loss)
    residual = (torch.stack(values) - initial).abs().max()/initial
    # A long record accumulates a few ulps per local linear solve.
    assert residual < 600*torch.finfo(torch.float64).eps
    assert abs(energies[-1]-energies[199] + heats[-1]-heats[199]) < 600*torch.finfo(torch.float64).eps*initial
    assert rt.loss > 0 if loss_rate else rt.loss == 0
    # Falsification: using the unshifted flux would not satisfy this identity.
    assert (torch.stack(plain) - initial).abs().max() > 1e-4*initial
    print(dict(loss_rate=loss_rate, relative_energy_heat_residual=float(residual)))


def test_sparse_flux_recovery_handles_dependent_rows():
    # A closed face loop has dependent generated-curl rows. Compare the
    # factored solve with an independent minimum-norm dense oracle.
    B = torch.tensor([[1., -1., 0.], [0., 1., -1.], [-1., 0., 1.], [1., -1., 0.]], dtype=torch.float64)
    forcing = B@torch.tensor([.3, .6, -.2], dtype=torch.float64)
    plan = _flux_plan(B.numpy())
    actual = _sheet_flux(plan, forcing)
    expected = torch.linalg.pinv(B)@forcing
    torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-15)
    assert torch.count_nonzero(_sheet_flux(plan, forcing*0)) == 0


def test_plain_foil_resonance_has_bounded_oscillations_and_decaying_window_maxima():
    """Production two-trace CN copper coupled to one leapfrog magnetic mode.

    The plain completed-step norm is appropriate to the guard's envelope fit,
    but only its modified counterpart has an exact heat-balance identity.
    """
    solver, _ = _resonator(0.)
    solver._thin_sheets = ()
    dt = solver.dt
    t = lambda value: torch.tensor(value, dtype=torch.float64)
    index = (torch.tensor([0]),)*3
    upper = (torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
    B, rate = t([[.6, .6]]), t([.04])
    decay = (1-dt*rate/2)/(1+dt*rate/2)
    drive = dt/2*B/(1+dt*rate/2)[:, None]
    effective = B.T@drive
    lhs = effective[None]/2 + torch.eye(2, dtype=torch.float64)[None]*(.5/dt)
    rhs_matrix = -effective[None]/2 + torch.eye(2, dtype=torch.float64)[None]*(.5/dt)
    entry = dict(electric_name="Ex", magnetic_name="Hz", normal_magnetic_name="Hy", index=index,
                 indices=torch.tensor([[0, 0, 0]]), low_h=index, high_h=upper, low_normal_h=index,
                 high_normal_h=index, low_valid=t([0.]), high_valid=t([0.]), paired=False, phase=0.,
                 sign=1., inv_tangent_dual=t([1.]), left_width=.5, right_width=.5, material_ade=False,
                 B=B, D=torch.zeros((2, 2), dtype=torch.float64), rates=rate, decay=decay, drive=drive,
                 state=t([[0.]]), area=t([1.]), admittance_scale=t([1.]), capacity=t([[.5], [.5]]),
                 inverse=torch.linalg.inv(lhs), rhs_matrix=rhs_matrix, old_e=t([[1.], [1.]]),
                 right_e=t([1.]), cumulative_loss=t(0.))
    solver._copper_foils = [entry]
    solver.Ex[index] = 1.
    solver.eps_Ex = torch.full_like(solver.Ex, 4.)
    solver.mu_Hz = torch.full_like(solver.Hz, 4.)
    h = t(0.)
    initial = total_energy(solver)
    energies, modified_plus_heat = [], []
    for _ in range(4096):
        entry["old_e"].copy_(torch.stack((solver.Ex[index], entry["right_e"])))
        h = h - dt*solver.Ex[index][0]
        solver.Hz[index], solver.Hz[upper] = h/2, -h/2
        advance_copper_electric(solver)
        energy = total_energy(solver)
        energies.append(energy)
        modified = energy - dt/2*h*solver.Ex[index][0]
        modified_plus_heat.append(modified + entry["cumulative_loss"])
        assert energy <= initial/(1-dt/2)*(1+1e-13)
        assert energy >= (initial-entry["cumulative_loss"])/(1+dt/2)*(1-1e-13)
    energies = torch.stack(energies)
    maxima = energies.reshape(-1, 256).amax(1)
    assert torch.all(maxima[1:] <= maxima[:-1])
    guard_maxima = energies[SAMPLE_INTERVAL-1::SAMPLE_INTERVAL].reshape(-1, WINDOW_SAMPLES).amax(1)
    assert torch.all(guard_maxima[1:] <= guard_maxima[:-1])
    assert torch.count_nonzero(torch.diff(energies) > 0) > 100
    assert energies[:256].max()-energies[:256].min() > .03*initial
    torch.testing.assert_close(torch.stack(modified_plus_heat), initial.expand(4096), rtol=2e-13, atol=0)
    print(dict(plain_relative_band=float((energies[:256].max()-energies[:256].min())/initial),
               first_window_max=float(maxima[0]), last_window_max=float(maxima[-1]),
               relative_modified_heat_residual=float((torch.stack(modified_plus_heat)-initial).abs().max()/initial)))
