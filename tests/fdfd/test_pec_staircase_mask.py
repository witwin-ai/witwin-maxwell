"""FDFD eliminates exactly the FDTD staircase PEC edge set.

Both solvers short an E edge iff both endpoints are conductor nodes (inside or on
the PEC union) or a PlanarPEC sheet owns it. The former FDFD rule (edge coverage
>= 0.5, i.e. the edge midpoint inside) shorted surface-straddling edges with an
exterior endpoint, the same defect that slowed the FDTD round coax.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.materials import _pec_edge_fill, _pec_signed_distance
from witwin.maxwell.fdtd.runtime.materials import _pec_edge_open_fractions

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FDFD requires CUDA")

H = 0.25e-3
_ENDPOINTS = {
    "Ex": ((slice(None, -1), slice(None), slice(None)), (slice(1, None), slice(None), slice(None))),
    "Ey": ((slice(None), slice(None, -1), slice(None)), (slice(None), slice(1, None), slice(None))),
    "Ez": ((slice(None), slice(None), slice(None, -1)), (slice(None), slice(None), slice(1, None))),
}


def _node_centred_boxes():
    n = 4
    return [mw.Box(position=(i * H, j * H, 0.0), size=(H, H, 4.0e-3))
            for i in range(-n, n + 1) for j in range(-n, n + 1) if (i * H) ** 2 + (j * H) ** 2 <= (1.0e-3) ** 2]


@pytest.mark.parametrize("geometries", [
    lambda: [mw.Cylinder(position=(0.0, 0.0, 0.0), radius=1.0e-3, height=4.0e-3, axis="z")],
    _node_centred_boxes,
], ids=["cylinder", "node_centred_boxes"])
def test_fdfd_pec_mask_equals_the_fdtd_staircase_short_set(geometries):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-2.0e-3, 2.0e-3), (-2.0e-3, 2.0e-3), (-1.0e-3, 1.0e-3))),
        grid=mw.GridSpec.uniform(H),
        boundary=mw.BoundarySpec.pml(num_layers=4),
        device="cuda",
    )
    for index, geometry in enumerate(geometries()):
        scene.add_structure(mw.Structure(geometry, mw.Material.pec(), name=f"pec{index}"))
    scene.add_source(mw.PointDipole(position=(1.5e-3, 0.0, 0.0), width=2 * H, polarization=(0, 0, 1),
                                    source_time=mw.CW(frequency=10e9, amplitude=1.0), name="src"))

    fdtd = mw.Simulation.fdtd(scene, frequencies=[10e9], run_time=mw.TimeConfig(time_steps=1)).prepare()
    fdtd_open = _pec_edge_open_fractions(fdtd.solver)
    fdfd = mw.Simulation.fdfd(scene, frequency=10e9, solver=mw.GMRES(max_iter=1)).prepare()
    fdfd_scene = fdfd.solver.scene
    mask = torch.as_tensor(fdfd.solver._ensure_pec_mask(), device="cuda")
    conductor = _pec_signed_distance(fdfd_scene) <= 1.0e-5 * H
    former = _pec_edge_fill(fdfd_scene)

    offset = 0
    former_extra = 0
    for component, (low, high) in _ENDPOINTS.items():
        shape = tuple(getattr(fdfd_scene, f"N{axis}_{component.lower()}") for axis in "xyz")
        count = math.prod(shape)
        fdfd_shorted = mask[offset:offset + count].reshape(shape)
        offset += count
        fdtd_shorted = fdtd_open[component] == 0.0
        assert fdtd_shorted.shape == fdfd_shorted.shape, component
        assert torch.equal(fdfd_shorted, fdtd_shorted.to(fdfd_shorted.device)), component
        both = (conductor[low] & conductor[high]).to(fdfd_shorted.device)
        assert torch.equal(fdfd_shorted, both), component
        former_extra += int(((former[component] >= 0.5).to(both.device) & ~both).sum())
    assert offset == mask.numel()
    # The former coverage >= 0.5 rule shorted many surface-straddling edges here.
    assert former_extra > 20
