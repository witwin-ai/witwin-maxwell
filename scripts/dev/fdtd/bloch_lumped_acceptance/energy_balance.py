"""Pulsed source-off decay and carrier-CW power conservation of a loaded cell."""
import sysconfig
import os

os.environ["CUDA_HOME"] = os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu13")
os.environ["WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR"] = os.path.expanduser("~/.cache/witwin-bc-fdtd-build-v6")

import argparse

from common import C0, F0, FREQUENCIES, assert_decay, assert_passivity, mw, run, simulation, torch
from witwin.maxwell.fdtd.quadrature import _compute_plane_flux


def plane(result, name, frequency):
    payload = result.monitor(name, frequency=frequency)
    # Rebuild from collocated fields so subtraction never reuses cached products
    # of the original total fields.
    return {key: payload[key] for key in
            ("axis", "frequency", "frequencies", "x", "y", "cell_widths", "Ex", "Ey", "Hx", "Hy")}


def power(reference, loaded, label, frequency, *, gate):
    index = FREQUENCIES.index(frequency)
    k_parallel = loaded.solver.resolved_bloch_wavevector[1]
    assert 2 * torch.pi * frequency / C0 > abs(k_parallel), "Frequency below the fixed-k light line"
    incident_fields = plane(reference, "upstream", frequency)
    upstream = plane(loaded, "upstream", frequency)
    scattered = dict(upstream)
    for name in ("Ex", "Ey", "Hx", "Hy"):
        scattered[name] = upstream[name] - incident_fields[name]
    incident = torch.as_tensor(_compute_plane_flux(incident_fields))
    reflected = -torch.as_tensor(_compute_plane_flux(scattered))
    transmitted = torch.as_tensor(_compute_plane_flux(plane(loaded, "downstream", frequency)))
    assert incident > 0 and torch.isfinite(incident), (frequency, incident)
    residual = incident - reflected - transmitted
    # A reflecting absorber leaves reference/scattered cross terms upstream.
    # Conservation uses the actual total flux on both planes, including them.
    net = torch.as_tensor(_compute_plane_flux(upstream)) - transmitted
    port = loaded.ports["load"]
    absorbed = -.5 * torch.real(port.voltage[index] * port.current[index].conj())
    print(f"{'CW gate' if gate else 'pulse diagnostic'} {label} f={frequency / F0:.3f}f0 "
          f"R={float(reflected/incident):.9g} T={float(transmitted/incident):.9g} "
          f"Aflux={float(residual/incident):.9g} Anet={float(net/incident):.9g} "
          f"Aload={float(absorbed/incident):.9g}", flush=True)
    if gate:
        if label == "LC":
            assert abs(net / incident) <= 1e-3, (frequency, net / incident)
        else:
            assert absorbed > 0, (frequency, absorbed)
            assert abs((net - absorbed) / absorbed) <= .01, (frequency, net, absorbed)


def cw_simulation(args, *, termination=None, empty=False):
    problem = simulation(angle=args.angle, empty=empty, termination=termination, steps=args.cw_steps)
    source = problem.scene.sources[0]
    problem.scene.sources[0] = mw.PlaneWave(
        name=source.name, direction=source.direction, polarization=source.polarization,
        injection=source.injection, source_time=mw.CW(F0))
    # Resolve one fixed Bloch phase from the discrete carrier, removing the
    # continuum/numerical-wavenumber mismatch as well as the broadband defect.
    problem.scene.boundary = mw.BoundarySpec.faces(default="pml", num_layers=16,
        x="bloch", y="bloch", z="pml", bloch_wavevector="auto")
    # Keep the same requested band and band-scale alpha as the pulse.
    # A late Hann window removes switch-on transients from the carrier phasors.
    problem.config.run_time = mw.TimeConfig(time_steps=args.cw_steps,
                                          transient_cycles=args.cw_transient_cycles)
    problem.config.spectral_sampler = mw.SpectralSampler(window="hanning")
    return problem


def report_lc_tail(history, *, dt, light_line):
    # Exclude the final off-stride sample. An AR(2) fit resolves the two
    # conjugate autonomous poles without a finite-record FFT-bin ambiguity.
    voltage = torch.complex(history[:-1, 4], history[:-1, 5])
    voltage = voltage[voltage.numel() // 2:]
    design = torch.stack((voltage[1:-1], voltage[:-2]), 1)
    coefficients = torch.linalg.lstsq(design, voltage[2:]).solution
    poles = torch.linalg.eigvals(torch.stack((coefficients, coefficients.new_tensor([1., 0.]))))
    pole = poles[poles.abs().argmax()]
    frequency = pole.angle().abs() / (2 * torch.pi * 32 * dt)
    residual = torch.linalg.vector_norm(design @ coefficients - voltage[2:]) / torch.linalg.vector_norm(voltage[2:])
    print(f"LC tail frequency_GHz={float(frequency / 1e9):.12g} "
          f"light_line_GHz={light_line / 1e9:.12g} "
          f"fitted_amplitude_rho={float(pole.abs() ** (1 / 32)):.12g} "
          f"AR2_relative_residual={float(residual):.9g}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120000)
    parser.add_argument("--angle", type=float, default=30.)
    parser.add_argument("--cw-steps", type=int, default=96000)
    parser.add_argument("--cw-transient-cycles", type=int, default=160)
    parser.add_argument("--models", nargs="+", choices=("LC", "R", "RLC", "conductivity"),
                        default=("LC", "R", "RLC", "conductivity"))
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--decay-only", action="store_true")
    stages.add_argument("--cw-only", action="store_true")
    args = parser.parse_args()
    models = {"LC": mw.SeriesRLC(l=5e-9, c=.5e-12), "R": mw.SeriesRLC(r=50.),
              "RLC": mw.SeriesRLC(r=50., l=5e-9, c=.5e-12), "conductivity": None}
    if not args.cw_only:
        reference, history = run(simulation(angle=args.angle, empty=True, steps=args.steps), history=True)
        assert_decay(history, loaded=False)
        for label in args.models:
            loaded, history = run(simulation(angle=args.angle, termination=models[label], steps=args.steps,
                                            conductivity=label == "conductivity"), history=True)
            if label != "conductivity":
                assert loaded.solver._port_cuda_graph_active, "Acceptance must exercise the captured load step"
            for frequency in FREQUENCIES:
                power(reference, loaded, label, frequency, gate=False)
            if label == "LC":
                assert_passivity(history)
                report_lc_tail(history, dt=loaded.solver.dt,
                               light_line=C0 * abs(loaded.solver.resolved_bloch_wavevector[1]) / (2 * torch.pi))
            else:
                assert_decay(history, loaded=label != "conductivity")
            print(f"PASS {label} pulsed {'passivity' if label == 'LC' else 'decay'}", flush=True)
    if not args.decay_only:
        reference, _ = run(cw_simulation(args, empty=True))
        phase = complex(reference.solver.boundary_phase_cos[1], reference.solver.boundary_phase_sin[1])
        for name in ("upstream", "downstream"):
            components = reference.monitor(name, frequency=F0)["components"]
            for component in ("Ex", "Hy"):
                field = torch.as_tensor(components[component]["data"])
                defect = torch.linalg.vector_norm(field[:, -1] - phase * field[:, 0]) \
                    / torch.linalg.vector_norm(field[:, 0])
                print(f"CW fixed-k phase {name} {component} defect={float(defect):.9g}", flush=True)
        # The CW work comparison uses circuit V/I; conductivity is the
        # independent source-off EM control, with no circuit state or work stamp.
        for label in (name for name in args.models if name != "conductivity"):
            loaded, _ = run(cw_simulation(args, termination=models[label]))
            assert loaded.solver._port_cuda_graph_active
            power(reference, loaded, label, F0, gate=True)
            print(f"PASS {label} carrier-CW power balance", flush=True)


if __name__ == "__main__":
    main()
