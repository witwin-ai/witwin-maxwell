"""Shared physical fixture for the coordinator's GPU-only acceptance runs."""
import sysconfig
import os

os.environ["CUDA_HOME"] = os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu13")
os.environ["WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR"] = os.path.expanduser("~/.cache/witwin-bc-fdtd-build-v6")

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import torch
import witwin.maxwell as mw

F0 = 3.0e9
C0 = 299792458.0
FREQUENCIES = (2.7e9, 2.85e9, F0, 3.15e9, 3.3e9)
FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
SPACING = .0005


def simulation(*, transverse="bloch", angle=0., termination=None, empty=False,
               steps=24000, graph=True, load_y=0., conductivity=False):
    """A split 14 mm square PEC patch in a 20 mm periodic cell.

    All frequencies are above the fixed-k light line and below the first
    diffracted-order cutoff. The two monitor planes are inside the total-field
    slab, 30 mm away from the patch; the downstream plane measures total flux.
    """
    theta = math.radians(angle)
    k_parallel = 2 * math.pi * F0 / C0 * math.sin(theta)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((- .01, .01), (- .01, .01), (- .06, .06))),
        grid=mw.GridSpec.uniform(SPACING),
        boundary=mw.BoundarySpec.faces(default="pml", num_layers=16,
            x=transverse, y=transverse, z="pml", bloch_wavevector=(0., k_parallel, 0.)),
        device="cuda:0",
    )
    if not empty:
        for name, bounds in (("left", (-.007, -.0005)), ("right", (.0005, .007))):
            scene.add_structure(mw.Structure(
                geometry=mw.PlanarPEC.rectangle((bounds, (-.007, .007)), normal_axis="z", coordinate=0.),
                material=mw.Material.pec(), name=name))
    if conductivity:
        # A resolved Ohmic bridge across the same gap, without circuit state.
        # sigma=L/(R*A) gives a nominal 50-ohm, one-cell-square bridge.
        scene.add_structure(mw.Structure(name="conductive_load",
            geometry=mw.Box(position=(0., load_y, 0.), size=(.001, SPACING, SPACING)),
            material=mw.Material(sigma_e=.001 / (50. * SPACING ** 2))))
    # The reference retains only a field observation contour (no termination or
    # structures), avoiding the implicit full-volume multifrequency DFT path.
    scene.add_port(mw.LumpedPort(
        name="load", negative=(-.0005, load_y, 0.), positive=(.0005, load_y, 0.),
        voltage_path=mw.AxisPath("x"),
        current_surface=mw.Box(position=(-.00025, load_y, 0.), size=(0., .0015, .0015)),
        termination=None if empty else termination,
    ))
    scene.add_source(mw.PlaneWave(
        name="incident", direction=(0., math.sin(theta), math.cos(theta)),
        polarization=(1., 0., 0.),
        source_time=mw.GaussianPulse(frequency=F0, fwidth=.9e9),
        injection=mw.TFSF.slab(axis="z", bounds=(-.045, .045)),
    ))
    for name, z in (("upstream", -.03), ("downstream", .03)):
        scene.add_monitor(mw.FluxMonitor(name, axis="z", position=z))
    return mw.Simulation.fdtd(
        scene, frequencies=FREQUENCIES, absorber="cpml", cuda_graph=graph,
        run_time=mw.TimeConfig(time_steps=steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
    )


def run(sim, *, history=False):
    prepared = sim.prepare()
    solver = prepared.solver
    samples = []
    original = solver.accumulate_time_observers
    field_weights = {}
    if history:
        for name in FIELDS:
            field = getattr(solver, name)
            mass = getattr(solver, "eps_" + name) if name.startswith("E") else solver.mu0
            volume = torch.ones(field.shape, dtype=torch.float64, device=solver.device)
            for axis, coordinate in enumerate("xyz"):
                primal = (coordinate == name[1].lower()) == name.startswith("E")
                widths = torch.as_tensor(getattr(solver.scene,
                    f"d{coordinate}_{'primal' if primal else 'dual'}64"), device=solver.device)
                shape = [1, 1, 1]
                shape[axis] = widths.numel()
                volume.mul_(widths.view(shape))
            field_weights[name] = .5 * mass * volume

    def sample(step):
        original(step)
        if step % 32 == 0 or step == sim.config.run_time.time_steps - 1:
            # Uniform vacuum cell: common cell volume cancels in the decay ratio.
            # Retain scalars on the GPU until the solve has finished.
            energy = torch.zeros((), device=solver.device, dtype=torch.float64)
            stored = energy.clone()
            for name in FIELDS:
                mass = getattr(solver, "eps_" + name) if name.startswith("E") else solver.mu0
                square = getattr(solver, name).double().square()
                energy.add_(torch.sum(mass * square))
                stored.add_(torch.sum(field_weights[name] * square))
                if solver.complex_fields_enabled:
                    square = getattr(solver, name + "_imag").double().square()
                    energy.add_(torch.sum(mass * square))
                    stored.add_(torch.sum(field_weights[name] * square))
            current = energy.new_zeros(())
            circuit = current.clone()
            voltage_real, voltage_imag = current.clone(), current.clone()
            load = solver._port_runtimes[0].lumped
            if load is not None:
                current = load.last_branch_current.double().square()
                circuit = .5 * (load.inductance * load.inductor_current.square()
                                + load.capacitance * load.capacitor_voltage.square())
                voltage_real = load.last_voltage_midpoint.double().clone()
                if solver.complex_fields_enabled:
                    current = current + load.last_branch_current_imag.double().square()
                    circuit = circuit + .5 * (load.inductance * load.inductor_current_imag.square()
                                             + load.capacitance * load.capacitor_voltage_imag.square())
                    voltage_imag = load.last_voltage_midpoint_imag.double().clone()
            samples.append(torch.stack((energy, current, stored + circuit, circuit,
                                        voltage_real, voltage_imag)))

    if history:
        solver.accumulate_time_observers = sample
    result = prepared.run()
    return result, torch.stack(samples).cpu() if history else None


def assert_decay(history, *, loaded):
    assert torch.isfinite(history).all(), "Nonfinite field energy/load current"
    channels = (0, 1) if loaded else (0,)
    for channel in channels:
        peak = history[:, channel].max()
        # Check the entire final sampling window, avoiding a current zero crossing.
        tail = history[-16:, channel].max()
        ratio = tail / peak
        print(f"decay channel={('field energy', 'current squared')[channel]} ratio={ratio.item():.9g}", flush=True)
        # Current squared is energy-like (inductor/branch energy scales as L*I^2),
        # so it shares the field-energy threshold. A 1e-12 current-squared limit
        # demanded amplitude 1e-6, three orders stricter than the field's 1e-3.
        limit = 1e-6
        assert peak > 0 and ratio < limit, (channel, ratio.item(), limit)


def assert_passivity(history):
    """Bound late total field-plus-LC energy without demanding bound-mode decay."""
    assert torch.isfinite(history).all(), "Nonfinite field/circuit state"
    # Samples are 32 steps apart. Fit maxima of complete 2048-step windows,
    # using the last half of the record after the pulse has vanished.
    count = (history.shape[0] - 1) // 64
    envelopes = history[:count * 64, 2].reshape(count, 64).amax(1)
    late = envelopes[count // 2:]
    x = torch.arange(late.numel(), dtype=torch.float64) * 2048
    y = late.log()
    centered = x - x.mean()
    slope = (centered * (y - y.mean())).sum() / centered.square().sum()
    residual = y - y.mean() - slope * centered
    stderr = (residual.square().sum() / (x.numel() - 2) / centered.square().sum()).sqrt()
    # A radius floor of about 2e-7 in float32 gives a log-energy slope of this
    # scale. This one-sided bound also admits genuine decay.
    floor = 4e-7
    upper = slope + 3 * stderr
    ratio = late[late.numel() // 2:].max() / late[:late.numel() // 2].max()
    bound = torch.exp(x[-1] * floor)
    print(f"LC total energy peak_J={float(history[:, 2].max()):.9g} "
          f"late_max_ratio={float(ratio):.9g} bound={float(bound):.9g} "
          f"log_energy_slope_per_step={float(slope):.12g} "
          f"slope_stderr={float(stderr):.9g} upper_3sigma={float(upper):.12g} "
          f"roundoff_floor={floor:.9g}", flush=True)
    print(f"LC final circuit_fraction={float(history[-1, 3] / history[-1, 2]):.9g}", flush=True)
    assert late.min() > 0 and ratio <= bound and upper <= floor, (ratio, bound, upper, floor)
