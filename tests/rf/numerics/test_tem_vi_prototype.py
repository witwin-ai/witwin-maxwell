import torch

import witwin.maxwell as mw
from witwin.core import Box
from witwin.maxwell.compiler.ports import compile_port_geometry
from witwin.maxwell.ports import AxisPath, LumpedPort
from witwin.maxwell.scene import prepare_scene


def _prepared_scene():
    return prepare_scene(
        mw.Scene(
            domain=mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
            grid=mw.GridSpec.uniform(0.1),
            boundary=mw.BoundarySpec.none(),
            device="cpu",
        )
    )


def _port(*, reverse=False):
    negative = (0.5, 0.5, 0.2)
    positive = (0.5, 0.5, 0.8)
    if reverse:
        negative, positive = positive, negative
    return LumpedPort(
        name="feed",
        negative=negative,
        positive=positive,
        voltage_path=AxisPath("z"),
        current_surface=Box(position=(0.5, 0.5, 0.45), size=(0.5, 0.5, 0.0)),
        reference_impedance=50.0,
    )


def _analytic_tem_fields(scene, *, voltage=3.0, current=3.0):
    fields = {
        "Ez": torch.full(
            (scene.Nx, scene.Ny, scene.Nz - 1),
            voltage / 0.6,
            dtype=torch.float64,
        )
    }

    x = torch.as_tensor(scene.x_nodes64, dtype=torch.float64)
    x_half = torch.as_tensor(scene.x_half64, dtype=torch.float64)
    y = torch.as_tensor(scene.y_nodes64, dtype=torch.float64)
    y_half = torch.as_tensor(scene.y_half64, dtype=torch.float64)
    z_half = torch.as_tensor(scene.z_half64, dtype=torch.float64)
    coefficient = current / (2.0 * 0.5 * 0.5)
    fields["Hx"] = (-coefficient * (y_half - 0.5))[None, :, None].expand(
        x.numel(), y_half.numel(), z_half.numel()
    ).clone()
    fields["Hy"] = (coefficient * (x_half - 0.5))[:, None, None].expand(
        x_half.numel(), y.numel(), z_half.numel()
    ).clone()
    return fields


def test_uniform_tem_vi_prototype_matches_the_analytic_integrals_below_one_percent():
    scene = _prepared_scene()
    compiled = compile_port_geometry(scene, _port())
    fields = _analytic_tem_fields(scene)

    voltage = compiled.integrate_voltage(fields)
    current = compiled.integrate_current(fields)

    assert torch.abs(voltage - 3.0) / 3.0 < 0.01
    assert torch.abs(current - 3.0) / 3.0 < 0.01


def test_reversing_terminals_flips_v_and_i_but_preserves_power():
    scene = _prepared_scene()
    forward = compile_port_geometry(scene, _port())
    reverse = compile_port_geometry(scene, _port(reverse=True))
    fields = _analytic_tem_fields(scene)

    forward_v = forward.integrate_voltage(fields)
    forward_i = forward.integrate_current(fields)
    reverse_v = reverse.integrate_voltage(fields)
    reverse_i = reverse.integrate_current(fields)

    torch.testing.assert_close(reverse_v, -forward_v)
    torch.testing.assert_close(reverse_i, -forward_i)
    torch.testing.assert_close(
        reverse.average_power(fields),
        forward.average_power(fields),
    )
