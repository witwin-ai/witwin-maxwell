"""Topology optimization of a 90-degree silicon waveguide bend with the native FDTD adjoint.

STATUS: NOT RUN. This example was written while the development GPUs were busy;
only its scene construction and parameterization are exercised by the CPU test
``tests/invdes/test_invdes_example.py``. Expect to tune ``--lr``, the time
configuration and the iteration count on the first GPU run.

Workflow (plain PyTorch, ``Scene -> Simulation -> Result``):

* ``BendDesign`` is a ``SceneModule`` whose trainable ``DensityParameterization``
  (diagonal-symmetric parameters -> conic filter -> tanh projection) supplies the
  ``MaterialRegion`` density of a 1.6 um x 1.6 um design slab.
* A ``ModeSource`` launches the fundamental mode of the input guide along +x; a
  ``FluxMonitor`` measures the power leaving through the output guide along -y.
* Each iteration runs ``Simulation.fdtd(model, ...).run()``; ``loss.backward()``
  drives the native FDTD adjoint back to the region density and autograd carries
  it through projection, filter and symmetry to the raw parameters.
* ``DesignLoop`` steps Adam, clips the parameters to [0, 1], follows a beta
  continuation schedule and records history. Once beta is large, the
  Zhou et al. minimum line-width / gap constraints join the loss as penalties.

Run on a free GPU, for example:
  CUDA_VISIBLE_DEVICES=1 conda run -n maxwell python examples/invdes/waveguide_bend_topology.py --iterations 80
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import witwin.maxwell as mw
from witwin.maxwell import invdes

UM = 1.0e-6
WAVELENGTH = 1.55 * UM
FREQUENCY = 299_792_458.0 / WAVELENGTH
CELL = 0.04 * UM
CORE_EPS = 12.11
WG_WIDTH = 0.50 * UM
WG_THICKNESS = 0.22 * UM
DESIGN_SIZE = 1.60 * UM
FILTER_RADIUS = 0.12 * UM


class BendDesign(mw.SceneModule):
    """Input guide along +x, output guide along -y, square design region between them.

    The bend maps the input port ``(-L/2, 0)`` onto the output port ``(0, -L/2)``
    under the ``x <-> y`` diagonal mirror, so that symmetry is imposed on the
    parameters.
    """

    def __init__(self, *, device="cuda", init=0.5):
        super().__init__()
        cells = round(DESIGN_SIZE / CELL)
        self.device_name = device
        self.design = invdes.DensityParameterization(
            (cells, cells, 1),
            (DESIGN_SIZE, DESIGN_SIZE, WG_THICKNESS),
            radius=FILTER_RADIUS,
            filter_kind="conic",
            padding="edge",
            beta=1.0,
            eta=0.5,
            diagonal=(0, 1),
            init=init,
            device=device,
        )

    def to_scene(self) -> mw.Scene:
        half = DESIGN_SIZE / 2.0
        scene = mw.Scene(
            subpixel_samples=mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic"),
            domain=mw.Domain(bounds=((-2.0 * UM, 1.2 * UM), (-2.0 * UM, 1.2 * UM), (-0.6 * UM, 0.6 * UM))),
            grid=mw.GridSpec.uniform(CELL),
            boundary=mw.BoundarySpec.pml(num_layers=10),
            device=self.device_name,
        )
        core = mw.Material(eps_r=CORE_EPS)
        guide_length = 2.0 * UM - half + 0.4 * UM  # reaches through the PML
        scene.add_structure(
            mw.Structure(
                name="input_guide",
                geometry=mw.Box(position=(-half - guide_length / 2.0, 0.0, 0.0), size=(guide_length, WG_WIDTH, WG_THICKNESS)),
                material=core,
            )
        )
        scene.add_structure(
            mw.Structure(
                name="output_guide",
                geometry=mw.Box(position=(0.0, -half - guide_length / 2.0, 0.0), size=(WG_WIDTH, guide_length, WG_THICKNESS)),
                material=core,
            )
        )
        scene.add_material_region(
            mw.MaterialRegion(
                name="bend",
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(DESIGN_SIZE, DESIGN_SIZE, WG_THICKNESS)),
                density=self.design(),
                eps_bounds=(1.0, CORE_EPS),
            )
        )
        scene.add_source(
            mw.ModeSource(
                position=(-1.4 * UM, 0.0, 0.0),
                size=(0.0, 1.2 * UM, 0.8 * UM),
                polarization="Ey",
                direction="+",
                source_time=mw.CW(frequency=FREQUENCY, amplitude=1.0),
                name="input",
            )
        )
        scene.add_monitor(mw.FluxMonitor("output", axis="y", position=-1.4 * UM, normal_direction="-"))
        return scene


def simulate(model: BendDesign):
    return mw.Simulation.fdtd(
        model,
        frequencies=[FREQUENCY],
        run_time=mw.TimeConfig.auto(steady_cycles=20, transient_cycles=25),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    ).run()


def length_scale_penalty(model: BendDesign) -> torch.Tensor:
    filtered = model.design.filtered()
    projected = model.design.project(filtered)
    spacing = model.design.spacing
    return invdes.minimum_linewidth_constraint(
        filtered, projected, spacing=spacing, radius=FILTER_RADIUS
    ) + invdes.minimum_gap_constraint(filtered, projected, spacing=spacing, radius=FILTER_RADIUS)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--length-scale-weight", type=float, default=1.0e3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("waveguide_bend_topology_out"))
    args = parser.parse_args()

    model = BendDesign(device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    schedule = invdes.BetaSchedule(initial=2.0, factor=2.0, interval=20, maximum=64.0)
    loop = invdes.DesignLoop(optimizer, bounds=(0.0, 1.0), beta_schedule=schedule)

    reference = None
    for _ in range(args.iterations):
        model.design.beta = loop.beta
        loop.zero_grad()
        flux = simulate(model).monitor("output")["flux"].real.sum()
        if reference is None:
            reference = flux.detach().abs()
        transmission = flux / reference
        penalty = length_scale_penalty(model)
        weight = args.length_scale_weight if loop.beta >= 16.0 else 0.0
        loss = -transmission + weight * penalty
        loss.backward()
        with torch.no_grad():
            gray = invdes.gray_level_penalty(model.design())
        record = loop.step(loss, transmission=transmission, length_scale=penalty, gray=gray)
        print(
            f"iter {record['iteration']:3d}  beta {record['beta']:5.1f}  "
            f"T/T0 {record['transmission']:.4f}  gray {record['gray']:.3f}  "
            f"length-scale {record['length_scale']:.2e}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "history.json").write_text(json.dumps(loop.history.to_dict(), indent=2))
    with torch.no_grad():
        torch.save(model.design().cpu(), args.out / "density.pt")


if __name__ == "__main__":
    main()
