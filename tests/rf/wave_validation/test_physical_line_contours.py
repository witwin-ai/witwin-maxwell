"""Independent physical V/I transfer extraction uses no modal impedance."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

path=Path(__file__).resolve().parents[3]/"docs/assessments/printed-lines-2026-10-22/physical_contours.py"
if not path.exists():
    pytest.skip("repository evidence not in this checkout: docs/assessments/printed-lines-2026-10-22/physical_contours.py", allow_module_level=True)
spec=importlib.util.spec_from_file_location("physical_line_contours",path)
driver=importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


@pytest.mark.parametrize("impedance,alpha",((43.5+0j,0.),(76.2-1.9j,.013),(50-5j,1.2)))
def test_transfer_eigenvectors_recover_physical_z_with_independent_standing_waves(impedance,alpha):
    positions=np.array((-.015,0.,.015));beta=31.4
    # Distinct forward/backward mixtures model two arbitrary independent drives.
    # No source matching or guessed modal impedance enters the extraction.
    forward=np.array((1+.2j,.11-.07j));backward=np.array((.17+.09j,1-.3j))
    traveling=np.exp((-alpha+1j*beta)*positions[:,None])
    voltage=forward*traveling+backward/traveling
    current=(forward*traveling-backward/traveling)/impedance
    extracted=driver.state_transfer(voltage,current,positions.tolist())
    for item in extracted:
        np.testing.assert_allclose(item["forward_characteristic_impedance_ohm"],impedance,rtol=2e-13,atol=1e-12)
        assert item["alpha_np_per_m"]==pytest.approx(alpha,abs=1e-12)
        assert item["beta_rad_per_m"]==pytest.approx(beta,abs=1e-12)
        assert max(item["state_condition_number_50ohm"])<3
