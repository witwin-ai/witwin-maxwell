"""Reproduce the uniform-magnetic routing evidence, end to end.

The Yee-staggered transverse operator eliminates ``Hz`` assuming ``mu = 1``, so the
production routing (``_assemble_vector_mode_data``) requires ``mu = 1`` before sending
a homogeneous aperture to the Yee solver; a uniformly magnetic aperture
(``mu_r != 1``) stays on the legacy diagonal-anisotropic operator, which threads
``mu`` through the eliminated longitudinal fields.

This probe drives the real ``solve_mode_source_profile`` entry on two otherwise
identical guides and prints the routed solver kind and the solved ``beta``:

  * ``mu_r = 1`` -> homogeneous non-magnetic branch -> Yee-staggered operator,
  * ``mu_r = 2`` -> magnetic aperture -> legacy diagonal-anisotropic operator.

Both dense paths report the same ``mode_solver_kind`` label (``vector_dense``), so the
routing is evidenced by the solved ``beta``, not the label: the ``mu_r = 2`` guide
returns a ``mu``-dependent ``beta`` (scaling with ``sqrt(eps_r mu_r)``) that does NOT
collapse onto the ``mu = 1`` value. If the routing dropped the ``nonmagnetic`` conjunct
from the branch guard, the ``mu_r = 2`` guide would be (wrongly) sent to the ``mu = 1``
Yee operator and return the ``mu = 1`` beta -- the falsification recorded in the
acceptance doc.

Run (from the worktree root, PYTHONPATH set to it):
    conda run -n maxwell --no-capture-output python \
        docs/assessments/e1-rf-mode-operator-probes/repro_uniform_mu_e2e.py
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from witwin.core.material import VACUUM_PERMITTIVITY

import witwin.maxwell as mw
from witwin.maxwell.compiler.materials import evaluate_material_permittivity
from witwin.maxwell.compiler.sources import _compile_mode_source
from witwin.maxwell.fdtd.excitation.modes import solve_mode_source_profile
from witwin.maxwell.scene import prepare_scene

FREQUENCY = 1.0e9


def _uniform_guide_scene(mu_r: float) -> mw.Scene:
    """A uniformly filled aperture (single material spanning the mode plane)."""
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.8, 0.8), (-0.8, 0.8), (-0.8, 0.8))),
        grid=mw.GridSpec.uniform(0.08),
        boundary=mw.BoundarySpec.none(),
        device="cpu",
    )
    scene.add_structure(
        mw.Box(position=(0.0, 0.0, 0.0), size=(1.28, 0.72, 0.72)).with_material(
            mw.Material(eps_r=4.0, mu_r=float(mu_r)),
            name="core",
        )
    )
    scene.add_source(
        mw.ModeSource(
            position=(0.0, 0.0, 0.0),
            size=(0.0, 0.56, 0.56),
            polarization="Ez",
            source_time=mw.CW(frequency=FREQUENCY, amplitude=1.0),
            name="port0",
        )
    )
    return scene


def _mode_context(scene: mw.Scene) -> SimpleNamespace:
    prepared_scene = prepare_scene(scene)
    model = prepared_scene.compile_materials()
    eps_r = evaluate_material_permittivity(model, FREQUENCY).to(dtype=torch.float32)
    eps_ex = (0.5 * (eps_r[:-1, :, :] + eps_r[1:, :, :]) * VACUUM_PERMITTIVITY).contiguous()
    eps_ey = (0.5 * (eps_r[:, :-1, :] + eps_r[:, 1:, :]) * VACUUM_PERMITTIVITY).contiguous()
    eps_ez = (0.5 * (eps_r[:, :, :-1] + eps_r[:, :, 1:]) * VACUUM_PERMITTIVITY).contiguous()
    return SimpleNamespace(
        scene=prepared_scene,
        dx=prepared_scene.dx,
        dy=prepared_scene.dy,
        dz=prepared_scene.dz,
        Ex=torch.empty((1,), device=prepared_scene.device, dtype=torch.float32),
        c=299792458.0,
        eps0=VACUUM_PERMITTIVITY,
        boundary_kind=prepared_scene.boundary.kind,
        _compiled_material_model=model,
        _mode_source_rebuild_from_fields=False,
        eps_Ex=eps_ex,
        eps_Ey=eps_ey,
        eps_Ez=eps_ez,
    )


def main() -> None:
    print(f"# uniform guide, eps_r=4, freq={FREQUENCY:g}")
    print(f"# {'mu_r':>5} {'solver_kind':>22} {'beta':>12}")
    for mu_r in (1.0, 2.0):
        scene = _uniform_guide_scene(mu_r)
        compiled_source = _compile_mode_source(scene.sources[0], default_frequency=FREQUENCY)
        result = solve_mode_source_profile(_mode_context(scene), compiled_source)
        beta = float(result["beta"])
        print(f"  {mu_r:5.1f} {result['mode_solver_kind']:>22} {beta:12.6f}")


if __name__ == "__main__":
    main()
