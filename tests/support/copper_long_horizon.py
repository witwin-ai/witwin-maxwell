"""Long-horizon passive boundedness run shared by the batch and milestone copper tests."""
import math

import torch

import witwin.maxwell as mw


def run_passive_full_field_steps(thickness, replays):
    """Random passive fields and both copper traces stay bounded over replays x 100 graph steps."""
    from witwin.maxwell.fdtd.checkpoint import graph_state_tensors
    from witwin.maxwell.fdtd.cuda.runtime.graph import CudaGraphRunner
    from witwin.maxwell.fdtd.runtime.stepping import _field_update_block, enforce_pec_boundaries
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.02, .02),) * 3),
        grid=mw.GridSpec.uniform(.01), boundary=mw.BoundarySpec.pec(), device="cuda",
        structures=[mw.Structure(geometry=mw.Box(position=torch.zeros(3, dtype=torch.float64),
            size=torch.tensor([0., .08, .08], dtype=torch.float64)), name="foil",
            material=mw.CopperFoil(thickness=thickness, frequency_range=(.5e9, 2e9)))])
    solver = mw.Simulation.fdtd(scene, frequency=2e9, run_time=mw.TimeConfig(time_steps=1)).prepare().solver
    generator = torch.Generator(device="cuda").manual_seed(911)
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        getattr(solver, name).normal_(generator=generator)
        getattr(solver, name).mul_(.1 if name.startswith("E") else .1 / 376.730313668)
    enforce_pec_boundaries(solver)
    for entry in solver._copper_foils:
        entry["right_e"].copy_(getattr(solver, entry["electric_name"])[entry["index"]])
    def energy_scale():
        value = torch.zeros((), device="cuda", dtype=torch.float64)
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            coefficient = getattr(solver, ("eps_" if component.startswith("E") else "mu_") + component)
            value += .5 * (coefficient.double() * getattr(solver, component).double().square()).sum() * .01**3
        for entry in solver._copper_foils:
            value += .5 * (entry["state"].square().sum(0) * entry["area"]).sum()
            value += .5 * (entry["capacity"][1] * entry["right_e"].double().square() * entry["area"]).sum()
        return value
    initial = float(energy_scale())
    live = graph_state_tensors(solver)
    saved = {key: value.clone() for key, value in live.items()}
    def block():
        for _ in range(100):
            _field_update_block(solver, 0.)
            enforce_pec_boundaries(solver)
    replay = CudaGraphRunner(device=solver.device, enabled=True, warmup_steps=1).capture(block)
    for key, value in saved.items():
        live[key].copy_(value)
    for _ in range(replays):
        replay()
    torch.cuda.synchronize()
    final = float(energy_scale())
    assert math.isfinite(final) and final < 8 * initial
    assert all(bool(torch.isfinite(getattr(solver, name)).all()) for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"))
    assert all(bool(torch.isfinite(entry["state"]).all()) for entry in solver._copper_foils)
    assert sum(float(entry["cumulative_loss"]) for entry in solver._copper_foils) >= 0
