"""GPU acceptance: normal limit and two settled CW-auto spot checks."""

import sysconfig
import os

os.environ.setdefault("CUDA_HOME", os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu13"))

import argparse
import math

from common import F0, K0, run_pair

# make_pulse needs a finite band: the pulsed arm runs a +/-5% band and reads its centre bin.
BAND = (.95, 1., 1.05)
from witwin.maxwell.constants import C_0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=160)
    args = parser.parse_args()
    passed = True
    for polarization in ("TE", "TM"):
        # Normal incidence has no fixed-k line (and no exported line reference),
        # so the normal-limit pair compares settled CW runs at F0.
        normal = run_pair(args.resolution, 0., polarization, 4., frequencies=(F0,), cw=True)
        near = run_pair(args.resolution, .01 * K0, polarization, 4., frequencies=(F0,), cw=True)
        error = float(abs(near["r"] - normal["r"]).max())
        print(dict(check="normal_limit", polarization=polarization, absolute_r_error=error), flush=True)
        passed &= error < 1.e-3
        for theta, frequency in ((math.radians(20), .9 * F0), (math.radians(35), 1.1 * F0)):
            kpar = 2 * math.pi * frequency * math.sin(theta) / C_0
            cw = run_pair(args.resolution, kpar, polarization, 4., frequencies=(frequency,), cw=True)
            settled = run_pair(args.resolution, kpar, polarization, 4., frequencies=(frequency,),
                               cw=True, cw_transient_cycles=600)
            # Match the discrete wavevector actually resolved by CW auto.
            pulse = run_pair(args.resolution, settled["kpar"], polarization, 4.,
                             frequencies=tuple(v * frequency for v in BAND))
            errors = [float((abs(pulse[key][1] - settled[key][0]) / abs(settled[key][0])).max())
                      for key in ("r", "t")]
            drift = [float((abs(cw[key] - settled[key]) / abs(settled[key])).max()) for key in ("r", "t")]
            print(dict(check="cw_auto", polarization=polarization, theta=theta,
                       frequency=frequency, matched_kpar=settled["kpar"], cw_horizon_drift=drift,
                       relative_r_error=errors[0], relative_t_error=errors[1]), flush=True)
            passed &= max(errors) < .01 and max(drift) < .001
    assert passed, "Continuity/CW acceptance failed"


if __name__ == "__main__":
    main()
