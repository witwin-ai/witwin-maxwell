"""Bounded GPU diagnostic of original and spectrally narrowed empty-cell pulses.

E/H histories are grid-summed weighted norms, including PML, without a 1/2
or cell-volume factor. Electric energy exactly matches the solver shutoff
functional. A 1e-30 shutoff records the late tail beyond the acceptance gate.
The scalar sampling hook leaves the field update and source kernels intact.
"""

import sysconfig
import os

os.environ.setdefault("CUDA_HOME", os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu13"))

import argparse
import fcntl
import json
import math
from pathlib import Path
import signal
import time
from unittest.mock import patch

import numpy as np
import torch

from common import F0, FREQUENCIES, K0, LIGHT_LINE_MARGIN, make_pulse, make_scene
import witwin.maxwell as mw
from witwin.maxwell.constants import C_0, MU_0
from witwin.maxwell.fdtd.runtime import stepping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pulse", choices=("original", "narrow"), required=True)
    parser.add_argument("--kpar-over-k0", type=float, default=.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    signal.alarm(290)
    lane_lock = Path(os.environ["WITWIN_GPU_LOCK_DIR"]) / "cuda1b.lock".open("r")
    fcntl.flock(lane_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.cuda.set_per_process_memory_fraction(.035)
    started = time.monotonic()
    kpar = args.kpar_over_k0 * K0
    cutoff = C_0 * kpar / (2 * math.pi)
    margin = LIGHT_LINE_MARGIN
    source_time = mw.GaussianPulse(frequency=F0, fwidth=.35 * F0)
    if args.pulse == "narrow":
        source_time = make_pulse(kpar, min(FREQUENCIES), max(FREQUENCIES))
    simulation = mw.Simulation.fdtd(
        make_scene(20, kpar, "TE", 1., source_time), frequencies=FREQUENCIES,
        run_time=mw.TimeConfig.auto(transient_cycles=0, steady_cycles=80),
        spectral_sampler=mw.SpectralSampler(window="none"), absorber="cpml",
        shutoff=1.e-30, shutoff_check_interval=100,
    )
    prepared = simulation.prepare()
    history = []
    original_energy = stepping._electric_field_energy

    def sample_energy(solver):
        electric = original_energy(solver)
        magnetic = MU_0 * sum(getattr(solver, name).square().sum() for name in
                              ("Hx", "Hy", "Hz", "Hx_imag", "Hy_imag", "Hz_imag"))
        history.append(dict(step=100 * (len(history) + 1),
                            time_ns=100 * (len(history) + 1) * solver.dt * 1.e9,
                            electric=float(electric), magnetic=float(magnetic),
                            total=float(electric + magnetic)))
        return electric

    with patch.object(stepping, "_electric_field_energy", sample_energy):
        result = prepared.run()
    solver = prepared.solver
    energy = np.array([row["total"] for row in history])
    times = np.array([row["time_ns"] for row in history])
    late = times > times[-1] / 2
    slope, intercept = np.polyfit(times[late], np.log(energy[late]), 1)
    residual = np.log(energy[late]) - (slope * times[late] + intercept)
    cutoff_ratio = math.exp(-.5 * ((source_time.frequency - cutoff * (1 + margin)) / source_time.fwidth) ** 2)
    final_electric = float(original_energy(solver))
    peak_electric = float(solver._shutoff_peak)
    summary = dict(pulse=args.pulse, provider=mw.__file__, resolution=20, kpar=kpar, cutoff_hz=cutoff,
                   margin=margin, frequency=source_time.frequency, fwidth=source_time.fwidth,
                   source_delay_ns=source_time.delay * 1.e9,
                   source_off_ns=(source_time.delay + 8 * source_time.sigma_t) * 1.e9,
                   spectrum_at_margin_over_peak=cutoff_ratio, steps=result.stats()["steps_run"],
                   duration_ns=result.stats()["steps_run"] * solver.dt * 1.e9,
                   electric_tail_over_peak=final_electric / peak_electric,
                   total_tail_over_peak=float(energy[-1] / energy.max()),
                   late_log_energy_slope_per_ns=float(slope), late_log_fit_rms=float(np.sqrt(np.mean(residual ** 2))),
                   quarter_peak_energy=[float(block.max()) for block in np.array_split(energy, 4)],
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                   elapsed_seconds=time.monotonic() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=summary, history=history), indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
