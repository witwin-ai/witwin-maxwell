"""Regenerate tests/mom/layered/data/strata_ling_jin_30ghz.npz with the open-source Strata library.

Strata (https://github.com/modelics/strata, commit 1dda5b9) computes formulation-C layered Green
functions by direct Sommerfeld integration under exp(+j*omega*t); the stored values are conjugated
to this package's exp(-i*omega*t) convention. Build Strata and compile strata_reference.cpp against it,
then run: python scripts/dev/rfic/make_strata_reference.py /path/to/strata_reference_binary
The stack is the four-layer grounded example of Ling and Jin (2000) shipped with Strata.
"""

from pathlib import Path
import os
import subprocess
import sys
import tempfile

import numpy as np


LAYERS = ((1.1, .7, 2.1), (.8, .3, 12.5), (.3, .5, 9.8), (0., .3, 8.6))  # zmin, h (mm), eps_r
PAIRS = ((.4e-3, 1.4e-3), (1.4e-3, 1.4e-3), (.4e-3, .6e-3))
SIGMAS = (0., .05)
FREQUENCY = 30e9
# Fixed samples preserve the archived horizontal reference across library versions.
RHO = np.array((1e-05, 1.3725965927377412e-05, 1.8840214063952605e-05, 2.5860013630631017e-05,
 3.5495366597555675e-05, 4.8720819249782056e-05, 6.687403049764221e-05,
 9.179106640370349e-05, 0.0001259921049894874, 0.0001729363340204262,
 0.00023737182283699315, 0.00032581575523800383, 0.00044721359549995795,
 0.0006138438574092373, 0.0008425599871529119, 0.0011564949675432418,
 0.0015874010519682001, 0.00217886127523986, 0.002990697562442441, 0.004105021284117567,
 0.005634538227695686, 0.00773394797298565, 0.010615590636131073, 0.014570923537052197,
 0.02))


def layer_file(sigma):
    rows = "\n".join(f"    L{i + 1}:\n        {{zmin: {z}, h: {h}, epsr: {e}, mur: 1, sigma: {sigma}}}\n"
                     for i, (z, h, e) in enumerate(LAYERS))
    return ("unit: mm\n\ndielectric_layers:\n\n" + rows +
            "\ntop_halfspace:\n    {epsr: 1, mur: 1, sigma: 0}\n\n"
            "bottom_halfspace:\n    {epsr: 1, mur: 1, sigma: -1}\n")


def main(binary):
    os.nice(10)
    values = np.empty((len(SIGMAS), len(PAIRS), 2, len(RHO)), complex)
    vertical = np.empty((len(SIGMAS), len(PAIRS), 4, len(RHO)), complex)
    scratch = Path(os.environ.get("WITWIN_EVIDENCE_ROOT", "evidence")) / "rfic-mom-20261106" / "codex" / "r3a"
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        tmp = Path(tmp)
        np.savetxt(tmp / "rho.txt", RHO)
        for s, sigma in enumerate(SIGMAS):
            (tmp / "layers.yaml").write_text(layer_file(sigma))
            for p, (zs, zo) in enumerate(PAIRS):
                subprocess.run([binary, tmp / "layers.yaml", str(FREQUENCY), str(zs), str(zo),
                                tmp / "rho.txt", tmp / "out.txt"], check=True, capture_output=True,
                               env=dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8"))
                data = np.loadtxt(tmp / "out.txt")
                values[s, p] = np.conj(np.stack((data[:, 1] + 1j * data[:, 2], data[:, 3] + 1j * data[:, 4])))
                vertical[s, p] = np.conj(np.stack((data[:, 5] + 1j * data[:, 6],
                                                   data[:, 7] + 1j * data[:, 8],
                                                   data[:, 9] + 1j * data[:, 10],
                                                   data[:, 3] + 1j * data[:, 4])))
    out = Path(__file__).resolve().parents[3] / "tests/mom/layered/data/strata_ling_jin_30ghz.npz"
    np.savez_compressed(out, rho=RHO, sigma=np.array(SIGMAS), pairs=np.array(PAIRS),
                        frequency=FREQUENCY, layers=np.array(LAYERS), values=values, vertical_values=vertical)


if __name__ == "__main__":
    main(sys.argv[1])
