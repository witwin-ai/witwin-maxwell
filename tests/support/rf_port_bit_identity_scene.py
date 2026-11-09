"""Run fixed LumpedPort/TerminalPort/WavePort scenes and save every port output.

Executed in a fresh interpreter by ``tests/rf/lumped/test_rf_port_bit_identity.py``
once against a frozen source tree and once against the working tree; the two
payloads must be bitwise equal. Only APIs present in both trees are used.
"""
from __future__ import annotations

import math
import sys

import torch

import witwin.maxwell as mw
from witwin.core import GeometryBase


def _terminal_lumped_scene():
    structures = (
        mw.Structure(mw.Box(position=(0.5, 0.5, 0.8), size=(0.5, 0.5, 0.2)), mw.Material.pec(), name="signal"),
        mw.Structure(mw.Box(position=(0.5, 0.5, 0.2), size=(0.5, 0.5, 0.2)), mw.Material.pec(), name="ground"),
    )
    ports = (
        mw.TerminalPort(
            "terminal",
            positive_terminal=mw.TerminalRef("signal"),
            negative_terminal=mw.TerminalRef("ground"),
            integration_path=mw.AxisPath("z"),
            reference_plane=0.45,
            reference_impedance=50.0,
        ),
        mw.LumpedPort(
            "lumped",
            positive=(1.2, 0.5, 0.6),
            negative=(1.2, 0.5, 0.4),
            voltage_path=mw.AxisPath("z"),
            current_surface=mw.Box(position=(1.2, 0.5, 0.45), size=(0.1, 0.1, 0.0)),
            reference_impedance=75.0,
        ),
    )
    return mw.Scene(
        domain=mw.Domain(bounds=((0.0, 1.6), (0.0, 1.0), (0.0, 1.0))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.pml(num_layers=6),
        structures=structures,
        ports=ports,
        lumped_elements=(mw.Resistor("shunt", positive=(1.4, 0.5, 0.6), negative=(1.4, 0.5, 0.4), resistance=100.0),),
        device="cuda",
    )


class _Tube(GeometryBase):
    kind = "bit_identity_tube"

    def __init__(self, *, inner_radius, outer_radius, length, device=None):
        super().__init__(position=(0.0, 0.0, 0.0), device=device)
        self.inner_radius, self.outer_radius, self.length = inner_radius, outer_radius, length

    def signed_distance(self, xx, yy, zz):
        dx, dy, dz = self._local_coords(xx, yy, zz)
        radial = torch.sqrt(dy.square() + dz.square())
        axial = torch.abs(dx) - 0.5 * self.length
        return torch.maximum(torch.maximum(radial - self.outer_radius, axial), -torch.maximum(radial - self.inner_radius, axial))


def _wave_scene(dx=0.01):
    def port(name, x, direction):
        contour_x = x + (0.5 * dx if direction == "+" else -0.5 * dx)
        mode = mw.WaveModeSpec(
            "tem",
            polarization="Ey",
            voltage_path=((x, 0.04, 0.0), (x, 0.16, 0.0)),
            current_contour=mw.Box(position=(contour_x, 0.0, 0.0), size=(0.0, 0.21, 0.21)),
        )
        return mw.WavePort(name, position=(x, 0.0, 0.0), size=(0.0, 0.40, 0.40), direction=direction,
                           reference_plane=x, modes=(mode,))

    length = 2.0 * (0.12 + 6 * dx) + 8.0 * dx
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.12, 0.12), (-0.21, 0.21), (-0.21, 0.21))),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.pml(num_layers=6),
        ports=(port("left", -0.02, "+"), port("right", 0.02, "-")),
        device="cuda",
    )
    scene.add_structure(mw.Structure(mw.Cylinder(position=(0.0, 0.0, 0.0), radius=0.04, height=length, axis="x"),
                                     mw.Material.pec(), name="inner_conductor"))
    scene.add_structure(mw.Structure(_Tube(inner_radius=0.16, outer_radius=0.20, length=length),
                                     mw.Material.pec(), name="outer_conductor"))
    return scene


def _record(prefix, result, payload):
    for name, data in result.ports.items():
        payload[f"{prefix}/{name}/voltage"] = data.voltage.detach().cpu()
        payload[f"{prefix}/{name}/current"] = data.current.detach().cpu()
        if data.available_power is not None:
            payload[f"{prefix}/{name}/available_power"] = data.available_power.detach().cpu()
    if result.network is not None:
        payload[f"{prefix}/network/s"] = result.network.s.detach().cpu()
    solver = result.solver
    if solver is not None:
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            payload[f"{prefix}/field/{component}"] = getattr(solver, component).detach().cpu()


def main(output):
    payload = {}
    frequencies = (0.4e9, 0.6e9)
    pulse = mw.GaussianPulse(frequency=0.5e9, fwidth=0.3e9)
    direct = mw.Simulation.fdtd(
        _terminal_lumped_scene(),
        frequencies=frequencies,
        excitations=mw.PortExcitation("terminal", source_time=pulse),
        run_time=mw.TimeConfig(time_steps=400),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).run()
    _record("direct", direct, payload)
    sweep = mw.Simulation.fdtd(
        _terminal_lumped_scene(),
        frequencies=frequencies,
        excitations=mw.PortSweep(source_time=pulse),
        run_time=mw.TimeConfig(time_steps=400),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).run()
    _record("sweep", sweep, payload)
    wave = mw.Simulation.fdtd(
        _wave_scene(),
        frequencies=(1.0e9,),
        excitations=mw.PortExcitation("left"),
        run_time=mw.TimeConfig(time_steps=1500),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()
    _record("wave", wave, payload)
    payload["module"] = mw.__file__
    torch.save(payload, output)


if __name__ == "__main__":
    main(sys.argv[1])
