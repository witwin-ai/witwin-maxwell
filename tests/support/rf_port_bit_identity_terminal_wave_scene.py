"""Existing port numerics plus a two-mode coupled-pair WavePort, saved for bitwise comparison.

Executed in fresh interpreters by ``tests/rf/terminal_wave/test_terminal_wave_bit_identity.py``
against the frozen pre-TerminalWavePort source and the working tree. Only APIs present in
both trees are used: the shared LumpedPort/TerminalPort/coax WavePort scenes of
``rf_port_bit_identity_scene.py`` and a shielded coupled pair whose WavePort carries the
even (mode 0) and odd (mode 1) quasi-TEM modes through the existing mode solver,
ModeSource/ModeMonitor compilation and modal overlap.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

import witwin.maxwell as mw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rf_port_bit_identity_scene import _record, main as shared_main  # noqa: E402


def _pair_scene(dx=0.01):
    def mode(name, index, x, direction):
        contour_x = x + (0.5 * dx if direction == "+" else -0.5 * dx)
        return mw.WaveModeSpec(
            "tem", index, name=name, polarization="Ez",
            voltage_path=((x, 0.05, 0.02), (x, 0.05, 0.16)),
            current_contour=mw.Box(position=(contour_x, 0.05, 0.0), size=(0.0, 0.07, 0.07)),
        )

    def port(name, x, direction):
        return mw.WavePort(name, position=(x, 0.0, 0.0), size=(0.0, 0.34, 0.34), direction=direction,
                           reference_plane=x, modes=(mode("even", 0, x, direction), mode("odd", 1, x, direction)))

    length = 2.0 * (0.12 + 6 * dx) + 8.0 * dx
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.12, 0.12), (-0.21, 0.21), (-0.21, 0.21))),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.pml(num_layers=6),
        ports=(port("left", -0.02, "+"), port("right", 0.02, "-")),
        device="cuda",
    )
    for index, (center, size) in enumerate((
        ((0.0, 0.0, 0.17), (length, 0.36, 0.02)), ((0.0, 0.0, -0.17), (length, 0.36, 0.02)),
        ((0.0, 0.17, 0.0), (length, 0.02, 0.36)), ((0.0, -0.17, 0.0), (length, 0.02, 0.36)),
    )):
        scene.add_structure(mw.Structure(mw.Box(position=center, size=size), mw.Material.pec(), name=f"shield{index}"))
    for name, y in (("line_a", 0.05), ("line_b", -0.05)):
        scene.add_structure(mw.Structure(mw.Box(position=(0.0, y, 0.0), size=(length, 0.04, 0.04)),
                                         mw.Material.pec(), name=name))
    return scene


def main(output):
    shared = Path(output).with_suffix(".shared.pt")
    shared_main(str(shared))
    payload = torch.load(shared, weights_only=False)
    module = payload.pop("module")
    pair = mw.Simulation.fdtd(
        _pair_scene(),
        frequencies=(0.8e9,),
        excitations=mw.PortSweep(),
        run_time=mw.TimeConfig(time_steps=1200),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()
    _record("pair", pair, payload)
    for name, data in pair.ports.items():
        payload[f"pair/{name}/characteristic_impedance"] = data.characteristic_impedance.detach().cpu()
        payload[f"pair/{name}/beta"] = data.beta.detach().cpu()
    payload["module"] = module
    torch.save(payload, output)


if __name__ == "__main__":
    main(sys.argv[1])
