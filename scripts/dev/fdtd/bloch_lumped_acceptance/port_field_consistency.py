"""Pulse or analytic CW Bloch run: combined field spectra versus passive port V/I.

Voltage is the midpoint E line integral on the port's half-step clock. Current
into the field is displacement current plus the RF-oriented H contour integral
(Ampere's law). Finite-record endpoint terms below make these identities exact
for either finite record, including CW endpoints.
No port accumulator or circuit state is used to reconstruct either reference.
"""
import sysconfig
import os

os.environ["CUDA_HOME"] = os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu13")
os.environ["WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR"] = os.path.expanduser("~/.cache/witwin-bc-fdtd-build-v6")

import argparse
import cmath
import itertools

from common import F0, FREQUENCIES, mw, simulation, torch
from witwin.maxwell.fdtd.coords import component_coords
from witwin.maxwell.scene import prepare_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("pulse", "cw"), default="pulse")
    parser.add_argument("--steps", type=int, help="Default: 4096 for pulse, 96000 for CW")
    parser.add_argument("--normalize-source", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    steps = args.steps if args.steps is not None else (96000 if args.source == "cw" else 4096)
    target_frequencies = (F0,) if args.source == "cw" else FREQUENCIES
    problem = simulation(angle=30., load_y=.001, steps=steps,
                         termination=mw.SeriesRLC(r=50., l=5e-9, c=.5e-12))
    if args.source == "cw":
        source = problem.scene.sources[0]
        problem.scene.sources[0] = mw.PlaneWave(
            name=source.name, direction=source.direction, polarization=source.polarization,
            injection=source.injection, source_time=mw.CW(F0, phase=.47))
        problem = mw.Simulation.fdtd(problem.scene, frequencies=(F0,), run_time=problem.config.run_time)
    problem.config.spectral_sampler = mw.SpectralSampler(window="none", normalize_source=args.normalize_source)
    # A nondefault reference also catches an imaginary bank left at f_ref=1.
    problem.config.reference_frequency = 4.2e9
    planning = prepare_scene(problem.scene.clone(device="cpu"))
    geometry = planning.compile_ports(device="cpu")[0]
    rows = {"voltage": [], "contour": [], "charge": []}
    probes = {}

    def add(row, component, index, weight):
        key = (component, tuple(int(i) for i in index))
        if key not in probes:
            name = f"port_native_{len(probes)}"
            coords = component_coords(planning, component)
            point = tuple(float(axis[i]) for axis, i in zip(coords, key[1]))
            problem.scene.add_monitor(mw.PointMonitor(name, point, fields=(component,)))
            probes[key] = name
        rows[row].append((key, float(weight)))

    for index, weight in zip(geometry.voltage_indices, geometry.voltage_weights):
        add("voltage", geometry.voltage_component, index, weight)
    for component, indices, weights in zip(geometry.current_components, geometry.current_indices, geometry.current_weights):
        for index, weight in zip(indices, weights):
            add("contour", component, index, weight)

    # The fixed fixture has an x-normal current surface. Its bounded dual Yee
    # face contains the normal Ex edges needed for the displacement-flux term.
    surface = problem.scene.ports[0].current_surface
    position, size = surface.position.detach().cpu().double(), surface.size.detach().cpu().double()
    normal = int(geometry.current_indices[0][0, 0])
    nodes_y = torch.as_tensor(planning.y_nodes64)
    nodes_z = torch.as_tensor(planning.z_nodes64)
    ys = torch.nonzero((nodes_y > position[1] - size[1] / 2) &
                       (nodes_y < position[1] + size[1] / 2)).flatten().tolist()
    zs = torch.nonzero((nodes_z > position[2] - size[2] / 2) &
                       (nodes_z < position[2] + size[2] / 2)).flatten().tolist()
    for j, k in itertools.product(ys, zs):
        add("charge", "Ex", (normal, j, k), planning.dy_dual64[j] * planning.dz_dual64[k])

    prepared = problem.prepare()
    solver = prepared.solver
    # Physical epsilon on this vacuum face; independent of the lumped solve.
    charge_indices = torch.tensor([key[1] for key, _ in rows["charge"]], device=solver.device)
    eps = solver.eps_Ex[tuple(charge_indices.T)].double().cpu().tolist()
    rows["charge"] = [(key, area * epsilon) for (key, area), epsilon in zip(rows["charge"], eps)]

    def endpoint(row):
        terms = [torch.complex(getattr(solver, component)[index], getattr(solver, component + "_imag")[index])
                 .to(torch.complex128) * weight for (component, index), weight in rows[row]]
        return torch.stack(terms).sum().detach()

    initial_v, initial_q = endpoint("voltage").cpu(), endpoint("charge").cpu()
    if args.source == "cw":
        # The patch/absorber field transient outlasts the isolated 2L/R branch
        # transient. Check final adjacent four-cycle windows after settling,
        # using demodulated E/H integrals and no circuit state.
        window_steps = round(4 / (F0 * solver.dt))
        settled = [[], []]
        accumulate = solver.accumulate_time_observers

        def sample(step):
            accumulate(step)
            if step >= steps - 2 * window_steps and step % 32 == 0:
                window = (step - (steps - 2 * window_steps)) // window_steps
                settled[window].append(torch.stack((endpoint("voltage"), endpoint("contour")))
                                       * cmath.exp(2j * torch.pi * F0 * solver.dt * step))

        solver.accumulate_time_observers = sample
    result = prepared.run()
    assert solver.complex_fields_enabled and solver._port_cuda_graph_active
    assert solver._spectral_integration_dt == (None if args.source == "cw" else solver.dt)
    assert torch.count_nonzero(solver._port_runtimes[0].accumulator_imag._voltage_sum), \
        "Fixture must exercise a nonzero imaginary voltage bank"
    values = {}
    for key, name in probes.items():
        spectra = []
        for frequency in target_frequencies:
            payload = result.monitor(name, frequency=frequency)
            assert tuple(payload["field_indices"][key[0]]) == key[1], (name, key)
            # Public observers combine both field banks before normalizing.
            spectra.append(torch.as_tensor(payload[key[0]]).detach().cpu().to(torch.complex128))
        values[key] = torch.stack(spectra)

    def integral(row):
        return torch.stack([values[key] * weight for key, weight in rows[row]]).sum(0)

    dt = float(solver.dt)
    frequencies = torch.tensor(target_frequencies, dtype=torch.float64)
    omega = 2 * torch.pi * frequencies
    half_retard = torch.exp(-.5j * omega * dt)
    end_phase = torch.exp(1j * omega * dt * steps)
    # Field observers use dt*f_ref for pulses and 1/sum(weights) for analytic
    # CW (runtime/representation.py:spectral_scale). Here all N weights are 1.
    scale = 1 / steps if args.source == "cw" else dt * solver._spectral_reference_frequency
    source_spectrum = torch.ones(len(target_frequencies), dtype=torch.complex128)
    # At F0 the analytic unit-envelope reference exp(-i*w*t) has spectrum 1;
    # the physical source phase is deliberately retained by normalization.
    if args.normalize_source and args.source == "pulse":
        # Independent full-record Fourier integral of the fixture's unit pulse.
        waveform = problem.scene.sources[0].source_time
        times = torch.arange(steps, dtype=torch.float64) * dt
        tau = times - waveform.delay
        signal = torch.exp(-.5 * (2 * torch.pi * waveform.fwidth * tau).square()) \
            * torch.cos(2 * torch.pi * waveform.frequency * tau)
        source_spectrum = scale * (torch.exp(1j * omega[:, None] * times) * signal).sum(-1)

    # S_prev = initial + exp(i*w*dt)*S - final*exp(i*w*N*dt).
    # CW's final fields do not vanish: retain their 1/N endpoint corrections.
    voltage = torch.cos(.5 * omega * dt) * integral("voltage") \
        + .5 * scale * half_retard * (initial_v - endpoint("voltage").cpu() * end_phase) / source_spectrum
    displacement = -2j * torch.sin(.5 * omega * dt) / dt * integral("charge") \
        + scale / dt * half_retard * (-initial_q + endpoint("charge").cpu() * end_phase) / source_spectrum
    # Compiled current weights already reverse the physical Ampere contour
    # for RF current into the field, so add that signed integral to dQ/dt.
    current = displacement + integral("contour")
    comparisons = []
    for name, reference in (("voltage", voltage), ("current", current)):
        actual = getattr(result.ports["load"], name).detach().cpu().to(torch.complex128)
        assert torch.isfinite(reference).all() and reference.abs().max() > 0
        relative_l2 = torch.linalg.vector_norm(actual - reference) / torch.linalg.vector_norm(reference)
        print(f"{name}: relative L2={relative_l2.item():.9g}; normalize_source={args.normalize_source}", flush=True)
        max_relative = ((actual - reference).abs() / reference.abs()).max()
        print(f"{name}: max relative abs diff={max_relative.item():.9g}", flush=True)
        comparisons.append((actual, reference))
    for actual, reference in comparisons:
        # Native field DFT banks accumulate in float32; port sums use float64.
        torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-7 * reference.abs().max().item())
    if args.source == "cw":
        previous, final = (torch.stack(window).mean(0).cpu() for window in settled)
        drift = ((final - previous).abs() / final.abs()).max()
        print(f"CW adjacent-window max relative drift={drift.item():.9g}", flush=True)
        torch.testing.assert_close(previous, final, rtol=1e-4, atol=0.)
    print("PASS port/combined-field spectral consistency", flush=True)


if __name__ == "__main__":
    main()
