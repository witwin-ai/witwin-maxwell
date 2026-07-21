"""Circuit-driven ESD (Phase-3 gap closure): the standard-network MNA topology.

An ESD discharge is driven through a source-impedance NETWORK instead of the
ideal current injection of :class:`ESDCurrentSource`. The versioned ESD current
waveform drives a time-dependent voltage source inside an MNA circuit whose
Thevenin source impedance is the standard 330 ohm discharge resistor shunted by
the 150 pF storage capacitance (:class:`ESDVoltageSource`); the network output
node is bound to a scene ``LumpedPort`` exactly as the plan-04 strong FDTD+MNA
coupling does.

Headline gate (a) -- RC-load analytic cross-check (reuses the F1b cross-check
harness *pattern*; no shared runtime code): the coupled FDTD+MNA run's port
voltage/current, with the ESD waveform driving the 330 ohm/150 pF network into a
resistive load, is reproduced by a fully independent scipy state-equation
integration of the same series-loop ODE. The EM one-port admittance is measured
and fit (data, not a shared code path), realized by hand into a modal
state-space, and integrated with :func:`scipy.integrate.solve_ivp` for the ESD
generator drive.

Falsification (gate a): perturbing the storage capacitance in a fresh coupled run
shifts its port voltage away from the (unperturbed) independent prediction beyond
the gate tolerance.

The unit-level tests also cover the ESD-waveform -> circuit-source resample, the
standard-network topology, and the generator-provenance ride-through to the
result (gate c).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Circuit-driven ESD FDTD/MNA coupling requires CUDA.",
)

# ---------------------------------------------------------------------------
# Pre-registered tolerances (frozen before measurement; see acceptance doc).
# The field one-port is a ~sub-pF gap capacitance shunting the slow (tens-of-ns)
# RC node, so its effect on the port voltage is a ~1% correction that the
# characterized modal model removes; the residual is integrator/tabulation floor.
# ---------------------------------------------------------------------------
_FIT_TOL = 1.0e-3        # rational fit rel. error vs measured Y over the band
_VOLTAGE_TOL = 1.0e-2    # |v_ode - v_fdtd|_inf / peak(|v_fdtd|)   (headline gate)
_CURRENT_TOL = 3.0e-2    # |i_ode - i_fdtd|_inf / peak(|i_fdtd|)   (source current)
_FIT_ORDER = 4
_FALSIFY_CS_SCALE = 1.06  # storage-capacitance perturbation for the falsification

# EM-coupling load-bearing variant (audit-minor H4a). The standard 150 pF network
# shunts the ~0.13 pF field one-port so heavily that dropping the EM coupling from
# the prediction changes nothing -- gate (a) alone does NOT make EM coupling
# load-bearing. This variant drives a small storage cap into a high-impedance load
# so the field one-port materially shifts the port voltage. Tolerances are
# pre-registered from the reference host (observed rel_true 2.0e-3, rel_zero 2.5e-2,
# ~12.7x margin); the gates below sit well inside those with slack for CUDA
# reduction/integration noise.
_EM_VARIANT_CS = 1.0e-12       # small storage capacitance (vs 150 pF standard)
_EM_VARIANT_RLOAD = 2000.0     # high-impedance load
_EM_MATERIAL_TOL = 1.0e-2      # zeroed-EM prediction must MISS by more than this
_EM_VOLTAGE_TOL = 6.0e-3       # with-EM prediction still reproduces the coupled run
_EM_IMPROVEMENT_FACTOR = 4.0   # with-EM must beat zeroed-EM by at least this factor

_DX = 0.005
_BOUNDS = ((-0.02, 0.02),) * 3
_BAND = np.linspace(1.5e9, 5.5e9, 25)

_LEVEL_VOLTAGE = 8_000.0
_RD = 330.0            # standard discharge resistor
_CS = 150.0e-12       # standard storage capacitance
_RLOAD = 470.0        # resistive device-under-test load
_R_CHAR = 50.0        # characterization series resistor
_CHAR_DRIVE = dict(t0=0.4e-9, tau=0.12e-9, fc=3.5e9, amp=1.0)
_CHAR_STEPS = 1200
_TEST_STEPS = 1500


def _port() -> mw.LumpedPort:
    return mw.LumpedPort(
        name="feed",
        positive=(0.0, 0.0, 0.005),
        negative=(0.0, 0.0, -0.005),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(position=(0.0, 0.0, -0.0025), size=(0.015, 0.015, 0.0)),
        reference_impedance=50.0,
    )


def _scene(circuit: mw.Circuit) -> mw.Scene:
    return mw.Scene(
        domain=mw.Domain(bounds=_BOUNDS),
        grid=mw.GridSpec.uniform(_DX),
        boundary=mw.BoundarySpec.pml(num_layers=8),
        ports=(_port(),),
        circuits=(circuit,),
        device="cuda",
    )


def _waveform() -> mw.ESDWaveform:
    return mw.ESDWaveform.iec_61000_4_2(_LEVEL_VOLTAGE)


def _prepared_dt() -> float:
    generator = mw.ESDVoltageSource("gun", port="feed", waveform=_waveform())
    solver = mw.Simulation.fdtd(
        _scene(generator.build_circuit(t_end=1.0e-9)), frequency=1.0e9
    ).prepare().solver
    return float(solver.dt)


# ---------------------------------------------------------------------------
# Characterization drive (broadband, F1b style) -- measures the EM one-port.
# ---------------------------------------------------------------------------
def _modulated_gaussian(times, *, t0, tau, fc, amp):
    envelope = np.exp(-(((times - t0) / tau) ** 2))
    return amp * envelope * np.sin(2.0 * np.pi * fc * (times - t0))


def _char_waveform():
    times = np.linspace(0.0, 3.0e-9, 4000)
    values = _modulated_gaussian(times, **_CHAR_DRIVE)
    return mw.PiecewiseLinearWaveform(
        torch.tensor(times, dtype=torch.float64),
        torch.tensor(values, dtype=torch.float64),
    )


def _char_circuit():
    circuit = mw.Circuit("loop")
    drive = circuit.node("drive")
    pnode = circuit.node("pnode")
    circuit.add(mw.VoltageSource("V1", drive, circuit.ground, 0.0, waveform=_char_waveform()))
    circuit.add(mw.Resistor("R1", drive, pnode, _R_CHAR))
    circuit.bind_port("feed", positive=pnode, negative=circuit.ground)
    return circuit


def _run(circuit, *, steps):
    return mw.Simulation.fdtd(
        _scene(circuit),
        frequencies=tuple(float(f) for f in _BAND),
        run_time=mw.TimeConfig(time_steps=steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    ).run()


def _measure_admittance(result):
    port = result.port("feed")
    v = port.voltage.detach().cpu().numpy().reshape(-1)
    i = port.current.detach().cpu().numpy().reshape(-1)
    return i / v


def _fit_admittance(admittance):
    config = mw.RationalFitConfig(
        order=_FIT_ORDER, iterations=10, proportional=True, enforce_stability=True
    )
    return mw.fit_rational(
        torch.tensor(_BAND, dtype=torch.float64),
        torch.tensor(admittance, dtype=torch.complex128),
        config,
        representation="Y",
    )


# ---------------------------------------------------------------------------
# Hand-built modal realization of Y_em(s) (NO framework realization helper).
# ---------------------------------------------------------------------------
def _realize_admittance(model):
    poles = model.poles.detach().cpu().numpy()
    residues = model.residues.detach().cpu().numpy().reshape(-1)
    D = float(model.direct.detach().cpu().numpy().reshape(-1)[0].real)
    Cp = float(np.asarray(model.proportional.detach().cpu().numpy()).reshape(-1)[0].real)

    used = np.zeros(len(poles), dtype=bool)
    a_blocks, b_blocks, c_blocks = [], [], []
    for k in range(len(poles)):
        if used[k]:
            continue
        p = poles[k]
        r = residues[k]
        if abs(p.imag) < 1.0e-3 * abs(p.real):
            used[k] = True
            a_blocks.append(np.array([[p.real]]))
            b_blocks.append(np.array([1.0]))
            c_blocks.append(np.array([r.real]))
        else:
            partner = int(
                np.argmin(
                    [
                        abs(poles[m] - np.conj(p)) + (1.0e30 if used[m] or m == k else 0.0)
                        for m in range(len(poles))
                    ]
                )
            )
            used[k] = used[partner] = True
            a1 = -2.0 * p.real
            a0 = float(abs(p) ** 2)
            b1 = 2.0 * r.real
            b0 = -2.0 * (r * np.conj(p)).real
            a_blocks.append(np.array([[0.0, 1.0], [-a0, -a1]]))
            b_blocks.append(np.array([0.0, 1.0]))
            c_blocks.append(np.array([b0, b1]))

    A = block_diag(*a_blocks)
    B = np.concatenate(b_blocks)
    C = np.concatenate(c_blocks)
    return A, B, C, D, Cp


@dataclass(frozen=True)
class _PortTrace:
    times: np.ndarray
    voltage: np.ndarray
    current: np.ndarray

    @property
    def peak_voltage(self):
        return float(np.abs(self.voltage).max())

    @property
    def peak_current(self):
        return float(np.abs(self.current).max())


def _generator():
    return mw.ESDVoltageSource("gun", port="feed", waveform=_waveform())


def _test_circuit(*, storage_capacitance=_CS, rload=_RLOAD):
    generator = mw.ESDVoltageSource(
        "gun", port="feed", waveform=_waveform(),
        discharge_resistance=_RD, storage_capacitance=storage_capacitance,
    )
    t_end = _TEST_STEPS * _prepared_dt()
    circuit = generator.build_circuit(t_end=t_end)
    circuit.add(mw.Resistor("Rload", circuit.node("tip"), circuit.ground, rload))
    return circuit


def _generator_drive_table():
    """The exact PWL generator drive V_gen(t) = R_d * i_esd(t) (shared input)."""

    t_end = _TEST_STEPS * _prepared_dt()
    pwl = _generator().circuit_waveform(t_end=t_end)
    return pwl.times.detach().cpu().numpy(), pwl.values.detach().cpu().numpy()


def _predict(model, times, *, zero_em=False, storage_capacitance=_CS, rload=_RLOAD) -> _PortTrace:
    """Independent scipy integration of the hand-derived ESD-network loop ODE.

        node 'tip':  (V_gen - v)/R_d = C_s dv/dt + v/R_load + I_field
        I_field    = C x + D v + C_p dv/dt      (measured EM one-port)
        dx/dt      = A x + B v

    ``zero_em=True`` drops the EM one-port entirely (``I_field = 0``, ``C_p = 0``):
    the prediction then models the bare R_d / C_s network into R_load with no field
    coupling. Used to prove the EM one-port is load-bearing (the with-EM vs
    zeroed-EM contrast). ``storage_capacitance`` / ``rload`` let the load-bearing
    variant use a small storage cap into a high-impedance load (where the field
    one-port materially shifts the port voltage).
    """

    A, B, C, D, Cp = _realize_admittance(model)
    if zero_em:
        C = np.zeros_like(C)
        D = 0.0
        Cp = 0.0
    n = A.shape[0]
    drive_t, drive_v = _generator_drive_table()

    def vgen(t):
        # Clamp to the table endpoints (the drive starts at 0 and is smooth at the
        # tail); a hard right=0.0 would spuriously zero the final sample when the
        # run's last float32 time rounds just past the float64 table end.
        return np.interp(t, drive_t, drive_v)

    def rhs(t, z):
        x = z[:n]
        v = z[n]
        dv = ((vgen(t) - v) / _RD - v / rload - (C @ x + D * v)) / (storage_capacitance + Cp)
        dx = A @ x + B * v
        return np.concatenate([dx, [dv]])

    dt = float(times[1] - times[0])
    solution = solve_ivp(
        rhs,
        (float(times[0]), float(times[-1])),
        np.zeros(n + 1),
        t_eval=times,
        rtol=1.0e-9,
        atol=1.0e-12,
        method="RK45",
        max_step=2.0 * dt,
    )
    assert solution.success, solution.message
    v = solution.y[n]
    current = (vgen(times) - v) / _RD
    return _PortTrace(times=times, voltage=v, current=current)


def _coupled_trace(result, circuit_name) -> _PortTrace:
    data = result.circuit(circuit_name)
    times = data.times.detach().cpu().numpy()
    v = data.node_voltage("tip").detach().cpu().numpy()
    # Source-branch current == -(current the generator pushes into R_d/the port).
    current = -data.branch_current("Vgen_gun").detach().cpu().numpy()
    return _PortTrace(times=times, voltage=v, current=current)


# ---------------------------------------------------------------------------
# Module fixtures: characterize once, run the ESD-network scene once, predict.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def characterization():
    result = _run(_char_circuit(), steps=_CHAR_STEPS)
    admittance = _measure_admittance(result)
    return admittance, _fit_admittance(admittance)


@pytest.fixture(scope="module")
def coupled():
    circuit = _test_circuit()
    result = _run(circuit, steps=_TEST_STEPS)
    return circuit.name, _coupled_trace(result, circuit.name)


@pytest.fixture(scope="module")
def predicted(characterization, coupled):
    _, model = characterization
    _, trace = coupled
    return _predict(model, trace.times)


# ---------------------------------------------------------------------------
# Gate (a): RC-load analytic cross-check.
# ---------------------------------------------------------------------------
def test_measured_admittance_fits_low_order_rational(characterization):
    admittance, model = characterization
    assert model.is_stable
    fit = model.evaluate(torch.tensor(_BAND, dtype=torch.float64)).detach().cpu().numpy().reshape(-1)
    rel = float(np.abs(fit - admittance).max() / np.abs(admittance).max())
    assert rel < _FIT_TOL, f"rational fit rel. error {rel:.3e} exceeds {_FIT_TOL:.1e}"


def test_circuit_driven_esd_matches_independent_port_voltage(coupled, predicted):
    """Headline gate (a): independent scipy ODE reproduces the coupled port voltage."""

    _, trace = coupled
    peak = trace.peak_voltage
    assert peak > 0.0
    # Non-vacuous: the discharge genuinely rings the standard network up.
    assert peak > 100.0
    rel = float(np.abs(predicted.voltage - trace.voltage).max() / peak)
    assert rel < _VOLTAGE_TOL, f"port-voltage cross-check rel. error {rel:.3e} exceeds {_VOLTAGE_TOL:.1e}"


def test_circuit_driven_esd_matches_independent_source_current(coupled, predicted):
    """Corroboration (a): the generator source current also agrees."""

    _, trace = coupled
    peak = trace.peak_current
    assert peak > 0.0
    rel = float(np.abs(predicted.current - trace.current).max() / peak)
    assert rel < _CURRENT_TOL, f"source-current cross-check rel. error {rel:.3e} exceeds {_CURRENT_TOL:.1e}"


def test_crosscheck_rejects_perturbed_storage_capacitance(characterization, predicted):
    """Falsification (a): a perturbed 150 pF storage cap is caught by the gate.

    The independent prediction uses the nominal 150 pF network; a fresh coupled run
    with the storage capacitance scaled by ``_FALSIFY_CS_SCALE`` shifts its port
    voltage away from that prediction beyond the gate tolerance.
    """

    circuit = _test_circuit(storage_capacitance=_CS * _FALSIFY_CS_SCALE)
    result = _run(circuit, steps=_TEST_STEPS)
    perturbed = _coupled_trace(result, circuit.name)
    peak = perturbed.peak_voltage
    rel = float(np.abs(predicted.voltage - perturbed.voltage).max() / peak)
    assert rel > _VOLTAGE_TOL, (
        f"perturbed storage cap not detected: rel. error {rel:.3e} within tol {_VOLTAGE_TOL:.1e}"
    )


@pytest.fixture(scope="module")
def coupled_em_variant():
    circuit = _test_circuit(storage_capacitance=_EM_VARIANT_CS, rload=_EM_VARIANT_RLOAD)
    result = _run(circuit, steps=_TEST_STEPS)
    return circuit.name, _coupled_trace(result, circuit.name)


def test_em_one_port_is_load_bearing(characterization, coupled_em_variant):
    """Gate (a) companion: the measured EM field one-port materially shifts v_port.

    Gate (a)'s standard 150 pF network so heavily shunts the ~0.13 pF field one-port
    that the zeroed-EM prediction matches the coupled run just as well -- i.e. gate
    (a) alone is EM-insensitive. This variant (small storage cap into a
    high-impedance load) brings the field one-port into play. Load-bearing evidence:
    the with-EM prediction reproduces the coupled FDTD, while dropping the EM
    one-port (``zero_em=True``) misses by more than the material tolerance and is
    beaten by the pre-registered factor. The zeroed-EM branch here is the committed
    falsification of "the EM one-port is load-bearing": removing the field coupling
    from the prediction reddens the reproduction against the same coupled run.
    """
    _, model = characterization
    _, trace = coupled_em_variant
    peak = trace.peak_voltage
    # Non-vacuous: the discharge rings the high-impedance variant network up.
    assert peak > 100.0
    kwargs = dict(storage_capacitance=_EM_VARIANT_CS, rload=_EM_VARIANT_RLOAD)
    pred_true = _predict(model, trace.times, zero_em=False, **kwargs)
    pred_zero = _predict(model, trace.times, zero_em=True, **kwargs)
    rel_true = float(np.abs(pred_true.voltage - trace.voltage).max() / peak)
    rel_zero = float(np.abs(pred_zero.voltage - trace.voltage).max() / peak)
    # (1) EM coupling materially shifts the port voltage: dropping it misses badly.
    assert rel_zero > _EM_MATERIAL_TOL, (
        f"EM one-port not load-bearing in this variant: zeroed-EM rel {rel_zero:.3e} "
        f"within material tol {_EM_MATERIAL_TOL:.1e}"
    )
    # (2) Only the full model (with the measured EM one-port) reproduces the run.
    assert rel_true < _EM_VOLTAGE_TOL, (
        f"with-EM prediction rel {rel_true:.3e} exceeds {_EM_VOLTAGE_TOL:.1e}"
    )
    # (3) ... and it beats the zeroed-EM prediction by the pre-registered factor.
    assert rel_true < rel_zero / _EM_IMPROVEMENT_FACTOR, (
        f"with-EM ({rel_true:.3e}) does not beat zeroed-EM ({rel_zero:.3e}) by "
        f"{_EM_IMPROVEMENT_FACTOR}x"
    )


# ---------------------------------------------------------------------------
# Waveform -> circuit-source resample (unit).
# ---------------------------------------------------------------------------
def test_circuit_waveform_conserves_impulse_and_scales():
    waveform = _waveform()
    t_end = 200.0e-9
    pwl = waveform.to_circuit_waveform(t_end=t_end, scale=_RD)
    times = pwl.times
    values = pwl.values
    # The standard contact current is non-negative, so the scaled drive is too.
    assert bool(torch.all(values >= 0.0))
    # The trapezoidal impulse of R_d * i(t) matches R_d * (analytic charge over [0, t_end]).
    trap = float(torch.trapezoid(values, times))
    dense = torch.linspace(0.0, t_end, 400_001, dtype=torch.float64)
    analytic = _RD * float(torch.trapezoid(waveform.current(dense), dense))
    assert abs(trap - analytic) < 1.0e-3 * abs(analytic)
    # Unit scale reproduces the raw current table.
    raw = waveform.to_circuit_waveform(t_end=t_end, scale=1.0)
    torch.testing.assert_close(raw.values * _RD, values)


def test_circuit_waveform_rejects_bad_arguments():
    waveform = _waveform()
    with pytest.raises(ValueError):
        waveform.to_circuit_waveform(t_end=-1.0e-9)
    with pytest.raises(ValueError):
        waveform.to_circuit_waveform(samples=1)


# ---------------------------------------------------------------------------
# Standard-network topology and provenance (units / gate c).
# ---------------------------------------------------------------------------
def test_standard_network_topology_and_binding():
    generator = _generator()
    circuit = generator.build_circuit(t_end=5.0e-9)
    devices = {d.name: d for d in circuit.devices}
    assert set(devices) == {"Vgen_gun", "Rd_gun", "Cs_gun"}
    assert float(devices["Rd_gun"].resistance) == pytest.approx(_RD)
    assert float(devices["Cs_gun"].capacitance) == pytest.approx(_CS)
    assert devices["Vgen_gun"].waveform is not None
    assert [b.port_name for b in circuit.bindings] == ["feed"]
    assert "esd_generator" in circuit.metadata


def test_esd_voltage_source_validates_inputs():
    with pytest.raises(TypeError):
        mw.ESDVoltageSource("gun", port="feed", waveform=object())
    with pytest.raises(ValueError):
        mw.ESDVoltageSource("gun", port="feed", waveform=_waveform(), discharge_resistance=0.0)
    with pytest.raises(ValueError):
        mw.ESDVoltageSource("gun", port="feed", waveform=_waveform(), storage_capacitance=-1.0)


def test_generator_provenance_rides_through_to_result(coupled):
    """Gate (c): waveform provenance (revision, level voltage) reaches the Result."""

    circuit_name, _ = coupled
    circuit = _test_circuit()
    result = _run(circuit, steps=64)
    assert circuit.name in result.esd_generator_names()
    provenance = result.esd_generator(circuit.name)
    assert provenance["capability_level"] == "stress-only"
    assert provenance["injection"] == "source_impedance_network"
    assert provenance["discharge_resistance"] == pytest.approx(_RD)
    assert provenance["storage_capacitance"] == pytest.approx(_CS)
    waveform = provenance["waveform"]
    assert waveform["standard"] == "IEC 61000-4-2"
    assert waveform["standard_revision"] == "ed2-contact"
    assert waveform["level_voltage"] == pytest.approx(_LEVEL_VOLTAGE)
    with pytest.raises(KeyError):
        result.esd_generator("does-not-exist")
