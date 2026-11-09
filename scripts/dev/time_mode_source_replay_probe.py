"""Time the isolated bend-scene mode-plane probe and its bridge cache on CUDA.

Run from the repository root with the maxwell environment:
    python -m scripts.dev.time_mode_source_replay_probe --repeats 5
Add --backward to time the example's full backward on the same device.
"""
import argparse
import json
import time

import torch

import witwin.maxwell as mw
from examples.invdes.waveguide_bend_topology import BendDesign, FREQUENCY, simulate
from witwin.maxwell.fdtd.adjoint.bridge import _FDTDGradientBridge, _mode_source_replay_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    model = BendDesign(device="cuda")
    simulation = mw.Simulation.fdtd(model, frequencies=[FREQUENCY])
    # Initialization and the forward mode solve are outside every measurement.
    with torch.no_grad():
        solver = simulation.prepare().solver
    bridge = _FDTDGradientBridge(simulation)
    inputs = bridge.trainable_inputs

    def measure(call):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        return 1000 * (time.perf_counter() - start)

    def uncached():
        with torch.enable_grad():
            error = _mode_source_replay_error(solver, model.to_scene(), inputs)
        assert error is None, error

    uncached()  # Warm up material kernels and autograd before timing.
    timings = {"uncached_probe_ms": [measure(uncached) for _ in range(args.repeats)]}
    timings["first_cached_probe_ms"] = measure(lambda: bridge._validate_mode_source_replay(solver, inputs))
    timings["cached_probe_ms"] = [measure(lambda: bridge._validate_mode_source_replay(solver, inputs))
                                  for _ in range(args.repeats)]
    if args.backward:
        result = simulate(model)
        loss = -result.monitor("output")["flux"].real.sum()
        timings["backward_ms"] = measure(lambda: torch.autograd.grad(loss, inputs))
    print(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()
