"""Quasi-TEM terminal basis of the TerminalWavePort against closed forms (GPU, mode solve only).

* Coupled stripline (homogeneous, exact Cohn conformal map): even/odd Z0 converge
  at first order (zero-thickness strip edges) and the Richardson extrapolation of
  two levels lands within 0.5 % of the exact values.
* Coupled microstrip (inhomogeneous): even/odd eps_eff and Z0 against the
  Kirschning-Jansen static formulas (quoted accuracy ~1 %). The discrete substrate
  interface under-loads eps_eff at feasible grids (first-order, from below), so the
  finest level is gated at 6 % (eps_eff) / 4 % (Z0) with monotone eps_eff
  convergence; this is a stated resolution limit, not a fitted tolerance.
* CPW (two coplanar grounds) with explicit polylines: the custom-2D and composite
  voltage/current integrals converge under refinement (first order, successive
  steps 5.8 -> 2.8 ohm at G = 2/4/8 cells), the V/I impedance from loops agrees with
  the power-voltage impedance to 1e-6, and loops at different distances from the
  conductor edges give the same Z (no edge-proximity bias).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.terminal_waveport import resolve_terminal_waveport_run_manifest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "support"))
from coupled_line_references import cohn_coupled_stripline, kirschning_jansen_coupled_microstrip  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="terminal mode solves run on CUDA")

EPS = 4.4


def _box_scene(dx, half_y, z_low, z_high):
    ny = round(half_y / dx)
    x = np.arange(-3, 4) * dx
    y = np.arange(-ny, ny + 1) * dx
    z = np.arange(round(z_low / dx) - 1, round(z_high / dx) + 2) * dx
    return mw.Scene(
        domain=mw.Domain(bounds=((x[0], x[-1]), (y[0], y[-1]), (z[0], z[-1]))),
        grid=mw.GridSpec.custom(x, y, z),
        boundary=mw.BoundarySpec.pec(),
        device="cuda",
    ), 2 * x[-1] + 20 * dx


def _sheet(name, y_low, y_high, z, length):
    vertices = ((-length / 2, y_low), (length / 2, y_low), (length / 2, y_high), (-length / 2, y_high))
    return mw.Structure(mw.PlanarPEC(vertices, normal_axis="z", coordinate=z), mw.Material.pec(), name=name)


def _prepare(scene):
    manifest = resolve_terminal_waveport_run_manifest(prepare_scene(scene), mw.PortSweep(), (1.0e9,))
    return manifest.prepared_ports[0]


def _even_odd(prepared):
    zc = prepared.characteristic_impedance[0].real
    return float(zc[0, 0] + zc[0, 1]), float(zc[0, 0] - zc[0, 1])


def _coupled_scene(dx, *, kind, half_y, top, width, gap, height):
    scene, length = _box_scene(dx, half_y, 0.0, top)
    scene.add_structure(_sheet("ground", -2 * half_y, 2 * half_y, 0.0, length))
    if kind == "stripline":
        scene.add_structure(_sheet("cover", -2 * half_y, 2 * half_y, top, length))
        scene.add_structure(mw.Structure(mw.Box(position=(0, 0, top / 2), size=(length, 4 * half_y, top)),
                                         mw.Material(eps_r=EPS), name="fill"))
    else:
        scene.add_structure(mw.Structure(mw.Box(position=(0, 0, height / 2), size=(length, 4 * half_y, height)),
                                         mw.Material(eps_r=EPS), name="substrate"))
    offset = 0.5 * (width + gap)
    for name, center in (("p", offset), ("n", -offset)):
        scene.add_structure(_sheet(name, center - width / 2, center + width / 2, height, length))
    scene.add_port(mw.TerminalWavePort(
        "P", position=(0.0, 0.0, top / 2), size=(0.0, 2 * half_y - 2 * dx, top), direction="+", reference_plane=0.0,
        terminals=(mw.TerminalRef("p"), mw.TerminalRef("n")),
        reference=(mw.TerminalRef("ground"), mw.TerminalRef("cover")) if kind == "stripline" else mw.TerminalRef("ground"),
        voltage_integrals=tuple(mw.CustomVoltageIntegral(mw.AxisSegment((0.0, c, height), (0.0, c, 0.0)))
                                for c in (offset, -offset)),
    ))
    return scene


def test_coupled_stripline_even_odd_impedance_converges_to_cohn():
    width, gap, spacing = 0.01, 0.005, 0.02
    exact_even, exact_odd = cohn_coupled_stripline(width, gap, spacing, EPS)
    levels = []
    for dx in (0.00125, 0.000625):
        prepared = _prepare(_coupled_scene(dx, kind="stripline", half_y=0.05, top=spacing,
                                           width=width, gap=gap, height=spacing / 2))
        levels.append(_even_odd(prepared))
        # Power-voltage Zc on the mode field equals the quasi-TEM C/C0 cross-check.
        assert torch.allclose(prepared.characteristic_impedance[0].real.cpu(), prepared.quasi_tem_impedance.cpu(),
                              rtol=1e-5, atol=1e-5)
        modal = prepared.modal_potentials.cpu()
        assert torch.allclose(modal[:, 0].abs(), torch.ones(2, dtype=modal.dtype), atol=1e-6)
    print("stripline", levels, "cohn", (exact_even, exact_odd))
    (even_1, odd_1), (even_2, odd_2) = levels
    assert abs(even_2 - exact_even) < abs(even_1 - exact_even)
    assert abs(odd_2 - exact_odd) < abs(odd_1 - exact_odd)
    assert abs(even_2 / exact_even - 1) < 0.02 and abs(odd_2 / exact_odd - 1) < 0.04
    assert abs((2 * even_2 - even_1) / exact_even - 1) < 0.005
    assert abs((2 * odd_2 - odd_1) / exact_odd - 1) < 0.005


def test_coupled_microstrip_even_odd_matches_kirschning_jansen():
    height, width, gap = 0.02, 0.025, 0.015
    eps_even, eps_odd, z_even, z_odd = kirschning_jansen_coupled_microstrip(width / height, gap / height, EPS)
    levels = []
    for dx in (0.0025, 0.00125):
        prepared = _prepare(_coupled_scene(dx, kind="microstrip", half_y=0.14, top=0.14,
                                           width=width, gap=gap, height=height))
        even, odd = _even_odd(prepared)
        levels.append((*prepared.eps_eff.tolist(), even, odd))
    print("microstrip", levels, "KJ", (eps_even, eps_odd, z_even, z_odd))
    coarse, fine = levels
    for index, reference in enumerate((eps_even, eps_odd)):
        assert coarse[index] < fine[index] < reference * 1.01
        assert abs(fine[index] / reference - 1) < 0.06
    for index, reference in ((2, z_even), (3, z_odd)):
        assert abs(fine[index] / reference - 1) < 0.04


def _cpw_scene(dx, *, polylines=True):
    height, half_signal, gap = 0.02, 0.01, 0.01
    scene, length = _box_scene(dx, 0.08, -0.06, 0.08)
    scene.add_structure(mw.Structure(mw.Box(position=(0, 0, height / 2), size=(length, 0.4, height)),
                                     mw.Material(eps_r=EPS), name="substrate"))
    scene.add_structure(_sheet("signal", -half_signal, half_signal, height, length))
    edge = half_signal + gap
    scene.add_structure(_sheet("ground_left", -0.2, -edge, height, length))
    scene.add_structure(_sheet("ground_right", edge, 0.2, height, length))

    def path(sign):
        return mw.CustomVoltageIntegral(mw.PolylinePath((
            (0.0, sign * 0.005, height), (0.0, sign * 0.01, height + 0.005),
            (0.0, sign * 0.025, height + 0.005), (0.0, sign * 0.025, height))))

    def loop(half_y, low, high):
        return mw.CustomCurrentIntegral(mw.RectangleLoop(center=(0.0, 0.0, 0.5 * (low + high)),
                                                         size=(0.0, 2 * half_y, high - low)))

    near = loop(0.0153, height - 0.0117, height + 0.0117)
    far = loop(0.0188, height - 0.0263, height + 0.0263)
    variants = {
        "right": (path(1.0), near),
        "composite": (mw.CompositeVoltageIntegral(((0.5, path(1.0)), (0.5, path(-1.0)))),
                      mw.CompositeCurrentIntegral(((0.5, near), (0.5, far)))),
    }
    for name, (voltage, current) in variants.items():
        scene.add_port(mw.TerminalWavePort(
            f"cpw_{name}", position=(0.0, 0.0, 0.01), size=(0.0, 0.16 - 2 * dx, 0.14), direction="+", reference_plane=0.0,
            terminals=(mw.TerminalRef("signal"),),
            reference=(mw.TerminalRef("ground_left"), mw.TerminalRef("ground_right")),
            voltage_integrals=(voltage,) if polylines else None, current_integrals=(current,)))
    return scene


def test_cpw_requires_an_explicit_polyline():
    with pytest.raises(ValueError, match="CPW and multi-ground"):
        _cpw_scene(0.0025, polylines=False)


def test_cpw_custom_and_composite_integrals_converge_and_agree_with_power_impedance():
    history = {}
    for dx in (0.005, 0.0025, 0.00125):
        scene = _cpw_scene(dx)
        manifest = resolve_terminal_waveport_run_manifest(prepare_scene(scene), mw.PortSweep(), (1.0e9,))
        for prepared in manifest.prepared_ports:
            power = float(prepared.characteristic_impedance[0, 0, 0].real)
            loop = float(prepared.loop_impedance[0, 0, 0].real)
            quasi = float(prepared.quasi_tem_impedance[0, 0])
            # Exact discrete Stokes/Ampere: the loop V/I equals the power-voltage Z.
            assert loop == pytest.approx(power, rel=1e-6)
            assert power == pytest.approx(quasi, rel=1e-6)
            assert prepared.integral_audit["voltage/T1"]["kind"] == "voltage"
            history.setdefault(prepared.port.name, []).append(power)
        # Loop placement does not bias Z: the composite averages a loop through the
        # gaps (near the edges) with one far outside them, and the left/right paths.
        values = [history[name][-1] for name in history]
        assert max(values) - min(values) < 1e-6 * values[0]
    print("cpw", history)
    for name, (coarse, medium, fine) in history.items():
        assert abs(fine - medium) < 0.6 * abs(medium - coarse), (name, coarse, medium, fine)
        # First-order (zero-thickness edge) convergence: successive steps halve.
        assert abs(fine - medium) / fine < 0.05, (name, coarse, medium, fine)


def _foil_pair_port(scene, loops, conductor_model):
    from benchmark.scenes.rf.printed_lines import HEIGHT, PAIR_OFFSET

    source = scene.ports[0]
    x = source.position[0]
    return mw.TerminalWavePort(
        "T", position=source.position, size=source.size, direction=source.direction,
        reference_plane=source.reference_plane,
        terminals=(mw.TerminalRef("signal_positive"), mw.TerminalRef("signal_negative")),
        reference=mw.TerminalRef("ground"),
        voltage_integrals=tuple(mw.CustomVoltageIntegral(mw.AxisSegment((x, s * PAIR_OFFSET, 0.0),
                                                                        (x, s * PAIR_OFFSET, -HEIGHT))) for s in (1, -1)),
        current_integrals=tuple(mw.CustomCurrentIntegral(mw.RectangleLoop(center=(x, s * PAIR_OFFSET, z),
                                                                          size=(0.0, 0.0071, height)))
                                for s, (z, height) in zip((1, -1), loops)),
        conductor_model=conductor_model,
    )


def test_split_trace_copper_foil_terminals_use_the_two_sided_foil_current():
    """Decision 6: split-trace CopperFoil strips are terminal conductors.

    The good-conductor mode keeps the PEC-limit transverse shape plus a passive
    first-order series-impedance correction, so the terminal matrix matches the
    PEC pair closely and beta gains a positive attenuation. A current loop must
    enclose the whole foil (both trace sides); a loop around one side only
    encloses no conductor node and is rejected by the topology audit.
    """
    from benchmark.scenes.rf.printed_lines import printed_line_scene

    results = {}
    for metal, model in (("pec", None), ("copper", "good_conductor")):
        base = printed_line_scene("differential_pair", metal=metal, device="cuda")
        scene = base.clone(ports=[_foil_pair_port(base, ((0.0, 0.0049), (0.0, 0.0049)), model)])
        results[metal] = _prepare(scene)
    pec, copper = results["pec"], results["copper"]
    assert torch.allclose(copper.characteristic_impedance.real, pec.characteristic_impedance.real, rtol=2e-3)
    assert torch.allclose(copper.loop_impedance.real, copper.characteristic_impedance.real, rtol=1e-6)
    assert bool(torch.all(copper.beta.imag > 0)) and bool(torch.all(pec.beta.imag == 0))
    assert copper.integral_audit["current/T1"]["checked"] == "encloses_terminal"
    base = printed_line_scene("differential_pair", metal="copper", device="cuda")
    one_sided = base.clone(ports=[_foil_pair_port(base, ((0.0025, 0.0035), (0.0, 0.0049)), "good_conductor")])
    with pytest.raises(ValueError, match="must enclose exactly that terminal"):
        _prepare(one_sided)
