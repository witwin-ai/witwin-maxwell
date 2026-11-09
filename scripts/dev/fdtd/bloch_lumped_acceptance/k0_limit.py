"""Zero-k imaginary-state invariance and bounded source-path gaps."""
import sysconfig
import os

os.environ["CUDA_HOME"] = os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu13")
os.environ["WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR"] = os.path.expanduser("~/.cache/witwin-bc-fdtd-build-v6")

import argparse

from common import FIELDS, FREQUENCIES, mw, simulation, torch
from witwin.maxwell.fdtd.lumped import LUMPED_STATE_NAMES


def imaginary_nonzero_counts(solver):
    load = solver._port_runtimes[0].lumped
    return torch.stack([
        *(torch.count_nonzero(getattr(solver, name + "_imag")) for name in FIELDS),
        *(torch.count_nonzero(getattr(load, name + "_imag")) for name in LUMPED_STATE_NAMES),
    ])


def run_cell(*, transverse, loaded, graph, steps):
    # "Unloaded" retains the same split patch and observation contour, with an
    # open gap (no termination). Source providers are left exactly as compiled.
    model = mw.SeriesRLC(r=50., l=5e-9, c=.5e-12) if loaded else None
    prepared = simulation(transverse=transverse, termination=model, steps=steps, graph=graph).prepare()
    solver = prepared.solver
    if transverse == "bloch" and loaded:
        saved_steps = [-1]
        counts = [imaginary_nonzero_counts(solver)]
        original = solver.accumulate_time_observers

        def sample(step):
            original(step)
            if step % 32 == 0 or step == steps - 1:
                saved_steps.append(step)
                counts.append(imaginary_nonzero_counts(solver))

        solver.accumulate_time_observers = sample
    result = prepared.run()
    if transverse == "bloch" and loaded:
        assert not torch.stack(counts).any(), f"Nonzero imaginary bank at saved steps {saved_steps}"
        assert not imaginary_nonzero_counts(solver).any(), "Nonzero imaginary bank at end"
        if graph:
            assert solver._port_cuda_graph_active, "Acceptance must exercise the captured load step"
        print(f"PASS zero imaginary fields/state at {len(counts)} saved steps and at end; graph={graph}", flush=True)
    return result


def gap(periodic, bloch, *, loaded):
    reference, candidate = [], []
    for name in ("upstream", "downstream"):
        blocks = []
        for result in (periodic, bloch):
            fields = []
            for frequency in FREQUENCIES:
                payload = result.monitor(name, frequency=frequency)
                for component in ("Ex", "Ey", "Hx", "Hy"):
                    # Put E and H in the same units before forming the L2 norm.
                    scale = 376.730313668 if component.startswith("H") else 1.
                    fields.append(torch.as_tensor(payload[component]).reshape(-1).to(torch.complex128) * scale)
            blocks.append(torch.cat(fields))
        reference.append(blocks[0])
        candidate.append(blocks[1])
    if loaded:
        for observable in ("voltage", "current"):
            reference.append(getattr(periodic.ports["load"], observable).to(torch.complex128))
            candidate.append(getattr(bloch.ports["load"], observable).to(torch.complex128))
    # Normalize each physical observable by its periodic-reference norm so V/I
    # cannot disappear numerically beneath the larger field arrays or units.
    scales = [torch.linalg.vector_norm(block) for block in reference]
    a = torch.cat([block / scale for block, scale in zip(reference, scales)])
    b = torch.cat([block / scale for block, scale in zip(candidate, scales)])
    return torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(a)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4096)
    args = parser.parse_args()
    for graph in (False, True):
        gaps = []
        for loaded in (False, True):
            periodic, bloch = (run_cell(transverse=transverse, loaded=loaded, graph=graph, steps=args.steps)
                               for transverse in ("periodic", "bloch"))
            gaps.append(gap(periodic, bloch, loaded=loaded))
        g0, g1 = gaps
        print(f"graph={graph} g0={g0.item():.9g} g1={g1.item():.9g} limit={(2 * g0 + 1e-6).item():.9g}", flush=True)
        assert torch.isfinite(g0) and torch.isfinite(g1)
        assert g1 <= 2 * g0 + 1e-6, (g0.item(), g1.item())
        print(f"PASS source-path gap graph={graph}", flush=True)


if __name__ == "__main__":
    main()
