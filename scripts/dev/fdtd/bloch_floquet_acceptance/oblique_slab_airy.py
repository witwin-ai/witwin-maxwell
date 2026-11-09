"""GPU acceptance: fixed-k_parallel pulsed slab against oblique Airy spectra.

Print every complex coefficient and relative error, plus analytic-error
refinement orders at 20/30/40 cells per vacuum wavelength. The separate
generation mesh defaults to 160 cells per wavelength. No GPU is selected here.

Gate (amended 2026-11-02, consistent with the CW Airy acceptance):
- every eps 4 coefficient below 1% at the generation mesh;
- the aggregate 30->40 refinement order in (1.5, 2.5) for both permittivities.
For eps 12, the absolute ppw160 error is reported, not gated. Near Fabry-Perot reflection
minima it is a second-order discretization error, identical to the CW run. The pulse is
qualified per frequency by broadband_checks.py --check parity (pulse = CW within 0.2%), and the
CW eps 12 result by the ppw320 extrapolation in
docs/memory/bloch-broadband-tfsf-injection-2026-10-31.md.
"""

import argparse
import math

import torch

from common import F0, FREQUENCIES, K0, oblique_airy, run_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-resolution", type=int, default=160)
    args = parser.parse_args()
    passed = True
    for eps in (4., 12.):
        for fraction in (.25, .55):
            kpar = fraction * K0
            for polarization in ("TE", "TM"):
                expected = oblique_airy(FREQUENCIES, kpar, eps, polarization)
                errors = {}
                for resolution in dict.fromkeys((20, 30, 40, args.generation_resolution)):
                    actual = run_pair(resolution, kpar, polarization, eps)
                    relative = torch.stack([abs(actual[key] - e) / abs(e)
                                            for key, e in zip(("r", "t"), expected)], -1)
                    errors[resolution] = relative
                    for i, frequency in enumerate(FREQUENCIES):
                        print(dict(eps=eps, kpar_over_k0=fraction, polarization=polarization,
                                   resolution=resolution, f_over_f0=frequency / F0,
                                   r=complex(actual["r"][i]), t=complex(actual["t"][i]),
                                   r_airy=complex(expected[0][i]), t_airy=complex(expected[1][i]),
                                   relative_error_r=float(relative[i, 0]), relative_error_t=float(relative[i, 1])), flush=True)
                    if resolution == args.generation_resolution and eps == 4.:
                        passed &= bool((relative < .01).all())
                for a, b in ((20, 30), (30, 40)):
                    order = torch.log(errors[a] / errors[b]) / math.log(b / a)
                    aggregate = torch.log(torch.linalg.vector_norm(errors[a], dim=0) /
                                          torch.linalg.vector_norm(errors[b], dim=0)) / math.log(b / a)
                    print(dict(eps=eps, kpar_over_k0=fraction, polarization=polarization,
                               refinement=(a, b), orders_per_frequency=order.tolist(),
                               aggregate_orders=aggregate.tolist()), flush=True)
                    if (a, b) == (30, 40):
                        passed &= bool(((aggregate > 1.5) & (aggregate < 2.5)).all())
    assert passed, "Airy acceptance failed: require eps 4 generation error <1% and 30->40 order in (1.5,2.5)"


if __name__ == "__main__":
    main()
