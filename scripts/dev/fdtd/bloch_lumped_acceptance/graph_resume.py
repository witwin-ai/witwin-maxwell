"""Exact complex loaded-cell graph/eager and mid-run disk-resume comparisons."""
import sysconfig
import os

os.environ["CUDA_HOME"] = os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu13")
os.environ["WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR"] = os.path.expanduser("~/.cache/witwin-bc-fdtd-build-v6")

import argparse
from pathlib import Path
import tempfile

from common import FIELDS, FREQUENCIES, mw, simulation, torch
from witwin.maxwell.fdtd.checkpoint import capture_checkpoint_state
from witwin.maxwell.fdtd.lumped import LUMPED_STATE_NAMES


def assert_equal(actual, expected, *, steps):
    a = capture_checkpoint_state(actual.solver, steps)
    b = capture_checkpoint_state(expected.solver, steps)
    assert a.schema == b.schema
    for name in a.tensors:
        assert torch.equal(a.tensors[name], b.tensors[name]), name
    for observable in ("voltage", "current"):
        assert torch.equal(getattr(actual.ports["load"], observable),
                           getattr(expected.ports["load"], observable)), observable
    a, b = actual.solver._port_runtimes[0], expected.solver._port_runtimes[0]
    assert a.sample_index == b.sample_index == steps
    for bank in ("accumulator", "accumulator_imag"):
        for name in ("_voltage_sum", "_current_sum", "_window_weight_sum"):
            assert torch.equal(getattr(getattr(a, bank), name), getattr(getattr(b, bank), name)), (bank, name)
    for monitor in ("upstream", "downstream"):
        for frequency in FREQUENCIES:
            a = actual.monitor(monitor, frequency=frequency)
            b = expected.monitor(monitor, frequency=frequency)
            for name in ("Ex", "Ey", "Hx", "Hy", "flux"):
                assert torch.equal(torch.as_tensor(a[name]), torch.as_tensor(b[name])), (monitor, frequency, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4096)
    args = parser.parse_args()
    stop = args.steps // 2 + 1

    def problem(graph):
        # Offset the load from the phase-reference plane so imaginary circuit
        # state is nonzero, making the checkpoint test exercise both banks.
        return simulation(angle=30., load_y=.001, steps=args.steps, graph=graph,
                          termination=mw.SeriesRLC(r=50., l=5e-9, c=.5e-12))

    eager = problem(False).run()
    graph = problem(True).run()
    assert graph.solver._port_cuda_graph_active, "Imaginary apply must execute in the captured port step"
    assert not eager.solver_stats["port_cuda_graph_active"]
    assert any(torch.count_nonzero(getattr(graph.solver, name + "_imag")) for name in FIELDS)
    assert_equal(graph, eager, steps=args.steps)
    print("PASS complex graph/eager torch.equal fields, load states and V/I", flush=True)

    with tempfile.TemporaryDirectory(prefix="bloch-lumped-resume-") as directory:
        for use_graph, expected in ((False, eager), (True, graph)):
            checkpoint = problem(use_graph).prepare().run_until(stop)
            for suffix in ("", "_imag"):
                for name in LUMPED_STATE_NAMES:
                    assert "lumped_port_0_" + name + suffix in checkpoint.physics.tensors
            assert torch.count_nonzero(checkpoint.physics.tensors["lumped_port_0_inductor_current_imag"]), \
                "Resume fixture must have active imaginary circuit history"
            path = Path(directory) / f"checkpoint-{use_graph}.pt"
            checkpoint.save(path)
            restored = mw.FDTDResumeCheckpoint.load(path, map_location="cpu")
            resumed = problem(use_graph).prepare().run(resume_from=restored)
            if use_graph:
                assert resumed.solver._port_cuda_graph_active
            assert_equal(resumed, expected, steps=args.steps)
            print(f"PASS complex disk resume at step {stop}; graph={use_graph}", flush=True)


if __name__ == "__main__":
    main()
