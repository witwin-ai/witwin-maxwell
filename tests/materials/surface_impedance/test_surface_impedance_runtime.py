"""Generic surface-impedance runtime write coverage (S1.1 sub-plane path).

Two runtime contracts that the compile-level and full-plane broadband gates do not pin:

* an exact-fill transverse box (transverse extent equal to the domain, not overflowing
  it) must have its whole surface E plane written by the generic per-edge ADE, exactly
  like the fused order-0 kernel -- the node-coverage slice under-covers the node-length
  tangential component by one row, which would leave a masked-zero PEC seam;
* a laterally finite, mid-domain block (all six faces exposed, sub-plane writes) must run
  and stay finite -- the headline S1.1 capability, otherwise validated only structurally.
"""

from __future__ import annotations

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box
from witwin.maxwell.media import RationalSurfaceImpedance, SurfaceImpedanceMedium
from witwin.maxwell.fdtd.surface_impedance_reference import good_conductor_surface_impedance

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="generic surface-impedance runtime requires CUDA.",
)

_BAND = (0.5e9, 5.0e9)


def _generic_medium(sigma=50.0, order=6):
    freqs = torch.logspace(math.log10(_BAND[0]), math.log10(_BAND[1]), 120, dtype=torch.float64)
    admittance = (1.0 / good_conductor_surface_impedance(sigma, freqs)).to(torch.complex128)
    model = RationalSurfaceImpedance.fit(freqs, admittance, order=order, band=_BAND)
    return SurfaceImpedanceMedium(impedance=model, name="coating")


def test_exact_fill_transverse_face_writes_the_whole_surface_plane():
    """A generic full-plane face owns every transverse tangential-E DOF (no PEC seam).

    The box transverse extent exactly equals the 0.04 m domain (nodes land on the
    boundary), so the node-coverage slice stops at the last cell, one short of the
    node-length tangential component. The write must still cover the full component plane;
    otherwise the last node row stays masked to zero.
    """
    dx = 0.005
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.02, 0.02),) * 3),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.faces(default="periodic", num_layers=8, strength=1.0, x=("pml", "pml")),
        device="cuda",
        sources=[
            mw.PointDipole(
                position=(-0.01, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                width=2.0 * dx,
                source_time=mw.CW(frequency=2.0e9, amplitude=40.0),
                name="pw",
            )
        ],
        structures=[
            mw.Structure(
                # Transverse size 0.04 == domain (exact fill, not overflowing); flush +x.
                geometry=Box(position=(0.0125, 0.0, 0.0), size=(0.015, 0.04, 0.04)),
                material=_generic_medium(),
            )
        ],
    )
    solver = mw.Simulation.fdtd(
        scene, frequencies=[2.0e9], run_time=mw.TimeConfig(time_steps=1)
    ).prepare().solver

    writes = solver._surface_impedance["writes"]
    full_generic = [w for w in writes if w.get("ade") is not None and w["full_plane"]]
    assert full_generic, "expected generic full-plane surface writes on the exact-fill face"
    for write in full_generic:
        electric = getattr(solver, write["e_name"])
        plane = electric.select(write["axis"], int(write["electric_index"]))
        # Every tangential-E DOF on the surface plane is an ADE edge (full coverage).
        assert write["ade"]["state"].shape[1] == plane.numel(), (
            f"{write['e_name']} face under-covers its surface plane: "
            f"{write['ade']['state'].shape[1]} edges vs {plane.numel()} DOFs"
        )


def test_finite_mid_domain_block_runs_finite_on_all_six_faces():
    """A laterally finite, mid-domain generic block (six sub-plane faces) stays finite."""
    dx = 0.004
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.03, 0.03),) * 3),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.pml(num_layers=8),
        device="cuda",
        sources=[
            mw.PointDipole(
                position=(-0.012, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                width=2.0 * dx,
                source_time=mw.CW(frequency=2.0e9, amplitude=40.0),
                name="pw",
            )
        ],
        structures=[
            mw.Structure(
                # Fully interior finite block: every face is an illuminated sub-plane.
                geometry=Box(position=(0.008, 0.0, 0.0), size=(0.012, 0.02, 0.02)),
                material=_generic_medium(),
            )
        ],
    )
    prepared = mw.Simulation.fdtd(
        scene, frequencies=[2.0e9], run_time=mw.TimeConfig(time_steps=200)
    ).prepare()
    solver = prepared.solver

    writes = solver._surface_impedance["writes"]
    exposed_faces = {(w["axis"], w["electric_index"]) for w in writes}
    # Six illuminated faces -> six distinct (axis, surface_node) planes, all sub-plane.
    assert len(exposed_faces) == 6
    assert all(w.get("ade") is not None and not w["full_plane"] for w in writes)

    # Advance the full run through the six-face sub-plane write path (time domain only).
    prepared.run_until(199)
    for name in ("Ex", "Ey", "Ez"):
        field = getattr(solver, name)
        assert torch.isfinite(field).all(), (
            f"finite-block generic surface diverged (non-finite {name})"
        )
    assert float(solver.Ez.abs().max()) > 0.0
