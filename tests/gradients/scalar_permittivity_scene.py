"""Shared scalar-permittivity point scene for exact FDTD gradient comparisons.

The trainable is one scalar ``eps_r`` on a Box. Density and parameter-field
repeatability are covered separately in the adjoint determinism tests.
"""

from __future__ import annotations

import torch

import witwin.maxwell as mw


class ScalarPermittivityPointScene(mw.SceneModule):
    def __init__(self):
        super().__init__()
        self.eps_r = torch.nn.Parameter(torch.tensor(3.5, device="cuda"))

    def to_scene(self):
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
            grid=mw.GridSpec.uniform(0.12),
            boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
            device="cuda",
        )
        scene.add_structure(
            mw.Structure(
                geometry=mw.Box(position=(0.06, 0.06, 0.18), size=(0.24, 0.24, 0.24)),
                material=mw.Material(eps_r=self.eps_r),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(0.0, 0.0, -0.18),
                polarization="Ez",
                width=0.04,
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=50.0),
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.18), fields=("Ez",)))
        return scene
