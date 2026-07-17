from types import SimpleNamespace

import numpy as np
import torch

from witwin.maxwell.fdtd.observers import _exact_cell_center_widths
from witwin.maxwell.postprocess.stratton_chu import (
    _equivalent_surface_currents_from_payload,
)


def test_nonuniform_yee_centers_resolve_exact_primal_widths():
    scene = SimpleNamespace(
        x_half64=np.asarray([0.5, 2.5]),
        y_half64=np.asarray([1.0]),
        z_half64=np.asarray([0.25]),
        dx_primal64=np.asarray([1.0, 3.0]),
        dy_primal64=np.asarray([2.0]),
        dz_primal64=np.asarray([0.5]),
    )
    solver = SimpleNamespace(scene=scene)

    np.testing.assert_allclose(
        _exact_cell_center_widths(solver, "x", [0.5, 2.5]),
        [1.0, 3.0],
    )
    np.testing.assert_allclose(
        _exact_cell_center_widths(solver, "y", [1.0]),
        [2.0],
    )


def test_monitor_payload_propagates_exact_area_weights_for_single_cell_axis():
    x = torch.tensor([0.5, 2.5], dtype=torch.float64)
    y = torch.tensor([1.0], dtype=torch.float64)
    field = torch.zeros((2, 1), dtype=torch.complex128)
    payload = {
        "kind": "plane",
        "axis": "z",
        "position": 0.0,
        "normal_direction": "+",
        "frequency": 1.0e9,
        "x": x,
        "y": y,
        "Ex": field,
        "Ey": field,
        "Hx": field,
        "Hy": field,
        "cell_widths": {
            "x": np.asarray([1.0, 3.0]),
            "y": np.asarray([2.0]),
        },
    }

    currents = _equivalent_surface_currents_from_payload(payload)

    torch.testing.assert_close(
        currents.weights_2d(),
        torch.tensor([[2.0], [6.0]], dtype=torch.float64),
    )
    assert currents.quadrature_rule == "cell_centered"
