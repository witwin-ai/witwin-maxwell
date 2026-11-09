"""Sheet-native transverse geometry and CUDA electrostatic line-mode oracles."""
import math
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.fdtd.excitation.modes.common import _mode_source_pec_slice
from witwin.maxwell.fdtd.excitation.modes.tem import _quasistatic_laplace_energy, _quasistatic_laplace_energy_torch


@pytest.mark.parametrize("dtype", [torch.bool, torch.float32])
def test_longitudinal_sheet_edges_supply_mode_plane_conductor_nodes(dtype):
    occupancy = torch.zeros(4, 5, 6)
    edges = torch.zeros(3, 5, 6, dtype=dtype)
    edges[:, 1:4, 2] = True
    solver = SimpleNamespace(_compiled_material_model={"pec_occupancy": occupancy, "pec_sheet_edges": {"Ex": edges}})
    actual = _mode_source_pec_slice(solver, normal_axis="x", plane_index=2, tangential_bounds=((0, 4), (0, 5)))
    assert actual[:, 2].sum() == 3
    assert not occupancy.any()
    edges[2] = False
    terminal = _mode_source_pec_slice(solver, normal_axis="x", plane_index=2, tangential_bounds=((0, 4), (0, 5)))
    assert not terminal.any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("ratio", [1., 4.4, 12.])
def test_cuda_potential_and_energy_match_independent_sparse_cpu_oracle(ratio):
    shape = (27, 31)
    eps = np.ones(shape)
    eps[:, :12] = ratio
    mask = np.zeros(shape, dtype=bool)
    mask[[0, -1], :] = True
    mask[:, [0, -1]] = True
    mask[8:19, 12] = True
    values = np.zeros(shape)
    values[8:19, 12] = 1
    expected_phi, expected_energy = _quasistatic_laplace_energy(eps, mask, values, du=.0004, dv=.0006)
    phi, energy = _quasistatic_laplace_energy_torch(torch.tensor(eps, device="cuda"), mask, values, du=.0004, dv=.0006)
    assert phi.is_cuda and energy.is_cuda
    torch.testing.assert_close(phi.cpu(), torch.from_numpy(expected_phi), rtol=2e-9, atol=2e-11)
    assert float(energy) == pytest.approx(expected_energy, rel=2e-11)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("contrast", [1., 4.4])
def test_native_mode_voltage_current_and_power_obey_capacitance_identity(contrast):
    from witwin.maxwell.fdtd.excitation.modes.tem import _solve_quasistatic_line_modes
    from witwin.maxwell.fdtd.excitation.modes.common import _discrete_mode_profile_power
    from witwin.maxwell.constants import ETA_0

    shape = (33, 35)
    eps = torch.ones(shape, dtype=torch.float64, device="cuda")
    eps[:, :17] = contrast
    occupancy = torch.zeros_like(eps)
    occupancy[14:19, 17] = 1
    line = _solve_quasistatic_line_modes((eps, eps, eps), occupancy, k0=1., du=.001, dv=.001,
        field_names=("Ey", "Ez", "Hy", "Hz"), signal_potentials=[1.])
    profiles = line["component_profiles"]
    voltage = -profiles["Ez"][16, :17].sum() * .001
    assert float(voltage) == pytest.approx(1., rel=2e-11)
    currents = []
    for low, high, bottom, top in ((11, 21, 13, 21), (8, 24, 9, 25)):
        hu, hv = profiles["Hy"], profiles["Hz"]
        current = ((hu[low+1:high+1, bottom] - hu[low+1:high+1, top]).sum()
                   + (hv[high, bottom+1:top+1] - hv[low, bottom+1:top+1]).sum()) * .001
        currents.append(current)
    expected_current = line["vacuum_energy"] * math.sqrt(line["eps_eff"]) / ETA_0
    for current in currents:
        torch.testing.assert_close(current, expected_current, rtol=2e-9, atol=2e-13)
    profiles.update(Ex=torch.zeros_like(eps), Hx=torch.zeros_like(eps))
    power = _discrete_mode_profile_power(profiles, coords_u=torch.arange(shape[0], device="cuda") * .001,
        coords_v=torch.arange(shape[1], device="cuda") * .001, normal_axis="x", staggered=True)
    torch.testing.assert_close(power, .5 * voltage * expected_current, rtol=2e-6, atol=2e-12)
