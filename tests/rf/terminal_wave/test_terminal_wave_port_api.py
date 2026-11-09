"""Declaration, lowering and network-algebra contracts of the TerminalWavePort.

CPU-only: no mode solve or FDTD run. Covers decision 1 (explicit polylines for
ambiguous apertures), decision 4 (full reference-impedance matrices) and the
exact discrete Stokes/Ampere identities of the custom-2D / composite lowering.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.path_integrals import (
    PortPlane,
    audit_terminal_integral,
    compile_current_integral,
    compile_voltage_integral,
)
from witwin.maxwell.scene import prepare_scene

DX = 0.01


def _grid_scene(**kwargs):
    nodes = np.arange(-10, 11) * DX
    return mw.Scene(
        domain=mw.Domain(bounds=((nodes[0], nodes[-1]),) * 3),
        grid=mw.GridSpec.custom(nodes, nodes, nodes),
        boundary=mw.BoundarySpec.pec(),
        device="cpu",
        **kwargs,
    )


def _plane(direction_sign=1):
    # x-normal aperture spanning y, z in [-0.08, 0.08] on the node grid.
    return PortPlane(
        normal_axis="x",
        tangential_axes=("y", "z"),
        electric_plane_index=10,
        magnetic_plane_index=10 if direction_sign > 0 else 9,
        lower=(2, 2),
        upper=(18, 18),
        reference_plane=0.0,
        direction_sign=direction_sign,
    )


def _microstrip_scene(*, with_port=True, **port_kwargs):
    scene = _grid_scene()
    scene.add_structure(mw.Structure(mw.Box(position=(0, 0, -0.065), size=(1.0, 1.0, 0.01)), mw.Material.pec(), name="ground"))
    scene.add_structure(mw.Structure(mw.Box(position=(0, 0, 0.005), size=(1.0, 0.04, 0.01)), mw.Material.pec(), name="strip"))
    if with_port:
        scene.add_port(mw.TerminalWavePort(
            "P", position=(0, 0, 0), size=(0, 0.16, 0.16), direction="+", reference_plane=0.0,
            terminals=(mw.TerminalRef("strip"),), reference=mw.TerminalRef("ground"), **port_kwargs))
    return scene


# --------------------------------------------------------------------------- #
# Declaration contracts.                                                      #
# --------------------------------------------------------------------------- #
def test_multi_terminal_and_multi_ground_ports_require_explicit_voltage_paths():
    common = dict(position=(0, 0, 0), size=(0, 0.16, 0.16), direction="+", reference_plane=0.0)
    with pytest.raises(ValueError, match="default voltage path is defined only for one signal"):
        mw.TerminalWavePort("P", terminals=(mw.TerminalRef("a"), mw.TerminalRef("b")),
                            reference=mw.TerminalRef("g"), **common)
    with pytest.raises(ValueError, match="CPW and multi-ground"):
        mw.TerminalWavePort("P", terminals=(mw.TerminalRef("s"),),
                            reference=(mw.TerminalRef("g1"), mw.TerminalRef("g2")), **common)
    with pytest.raises(ValueError, match="requires current_integrals"):
        mw.TerminalWavePort("P", terminals=(mw.TerminalRef("s"),), reference=mw.TerminalRef("g"),
                            impedance_definition="power_current", **common)


def test_coplanar_single_ground_without_polyline_is_rejected_as_ambiguous():
    scene = _grid_scene()
    scene.add_structure(mw.Structure(
        mw.PlanarPEC(((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)),
                     holes=(((-0.4, -0.04), (0.4, -0.04), (0.4, 0.04), (-0.4, 0.04)),), normal_axis="z", coordinate=0.0),
        mw.Material.pec(), name="ground"))
    scene.add_structure(mw.Structure(
        mw.PlanarPEC(((-0.3, -0.01), (0.3, -0.01), (0.3, 0.01), (-0.3, 0.01)), normal_axis="z", coordinate=0.0),
        mw.Material.pec(), name="signal"))
    port = mw.TerminalWavePort("cpw", position=(0, 0, 0), size=(0, 0.16, 0.16), direction="+", reference_plane=0.0,
                               terminals=(mw.TerminalRef("signal"),), reference=mw.TerminalRef("ground"))
    with pytest.raises(ValueError, match="default voltage path is ambiguous"):
        scene.add_port(port)


def test_microstrip_default_path_is_the_footprint_overlap_segment():
    scene = _microstrip_scene()
    port = scene.ports[0]
    record = port._default_voltage_path
    assert record["path_axis"] == "z" and record["offset_axis"] == "y"
    assert record["start"] == pytest.approx(0.0) and record["end"] == pytest.approx(-0.06)
    assert record["offset"] == pytest.approx(0.0)
    compiled = prepare_scene(scene).compile_ports  # scene compiles lazily; geometry compile is below
    from witwin.maxwell.compiler.terminal_waveports import compile_terminal_waveport

    compiled = compile_terminal_waveport(scene, port, device="cpu")
    assert compiled.default_voltage_path.start == pytest.approx((0.0, 0.0, 0.0))
    assert compiled.default_voltage_path.end == pytest.approx((0.0, 0.0, -0.06))
    ((component, indices, weights),) = compiled.voltage_integrals[0].terms
    assert component == "Ez" and indices.shape == (6, 3)
    assert torch.allclose(weights, torch.full((6,), -DX, dtype=torch.float64))


def test_reference_impedance_scalar_vector_and_full_matrix():
    common = dict(position=(0, 0, 0), size=(0, 0.16, 0.16), direction="+", reference_plane=0.0,
                  terminals=(mw.TerminalRef("a"), mw.TerminalRef("b")), reference=mw.TerminalRef("g"),
                  voltage_integrals=(mw.CustomVoltageIntegral(mw.AxisSegment((0, -0.02, 0), (0, -0.02, -0.06))),
                                     mw.CustomVoltageIntegral(mw.AxisSegment((0, 0.02, 0), (0, 0.02, -0.06)))))
    assert mw.TerminalWavePort("P", **common).reference_impedance == ((50.0, 0.0), (0.0, 50.0))
    assert mw.TerminalWavePort("P", reference_impedance=(40.0, 60.0), **common).reference_impedance == ((40.0, 0.0), (0.0, 60.0))
    full = mw.TerminalWavePort("P", reference_impedance=((60.0, 15.0), (15.0, 60.0)), **common)
    assert full.reference_impedance == ((60.0, 15.0), (15.0, 60.0))
    assert mw.TerminalWavePort("P", reference_impedance="characteristic", **common).reference_impedance == "characteristic"
    for bad, message in ((((60.0, 15.0), (14.0, 60.0)), "symmetric"), (((10.0, 50.0), (50.0, 10.0)), "positive definite"),
                         (((60.0, 1j), (1j, 60.0)), "must be real")):
        with pytest.raises(ValueError, match=message):
            mw.TerminalWavePort("P", reference_impedance=bad, **common)
    with pytest.raises(ValueError, match="terminal basis only"):
        mw.TerminalWavePort("P", basis="modal", reference_impedance=50.0, **common)
    modal = mw.TerminalWavePort("P", basis="modal", **common)
    assert modal.channel_names == ("P::M1", "P::M2")
    assert mw.TerminalWavePort("P", **common).channel_names == ("P::T1", "P::T2")


# --------------------------------------------------------------------------- #
# Lowering: exact discrete Stokes / Ampere identities.                        #
# --------------------------------------------------------------------------- #
def _gather(scene, compiled, fields):
    total = 0.0
    for component, indices, weights in compiled.terms:
        array = fields[component]
        total += float(torch.sum(array[tuple(indices.T)] * weights))
    return total


def test_polyline_staircase_integrates_any_gradient_exactly():
    scene = prepare_scene(_grid_scene())
    plane = _plane()
    generator = torch.Generator().manual_seed(7)
    phi = torch.randn((21, 21, 21), generator=generator, dtype=torch.float64)
    widths = DX
    fields = {
        "Ey": -(phi[:, 1:, :] - phi[:, :-1, :]) / widths,
        "Ez": -(phi[:, :, 1:] - phi[:, :, :-1]) / widths,
    }
    points = ((0.0, -0.05, -0.06), (0.0, 0.03, -0.01), (0.0, 0.03, 0.04), (0.0, -0.02, 0.07))
    spec = mw.CustomVoltageIntegral(mw.PolylinePath(points))
    compiled = compile_voltage_integral(scene, spec, plane, device="cpu")
    start, end = (10, 5, 4), (10, 8, 17)
    assert _gather(scene, compiled, fields) == pytest.approx(float(phi[start] - phi[end]), rel=1e-12, abs=1e-12)
    composite = mw.CompositeVoltageIntegral(((0.5, spec), (-2.0, mw.CustomVoltageIntegral(
        mw.AxisSegment((0.0, 0.0, 0.0), (0.0, 0.06, 0.0))))))
    compiled = compile_voltage_integral(scene, composite, plane, device="cpu")
    expected = 0.5 * float(phi[start] - phi[end]) - 2.0 * float(phi[10, 10, 10] - phi[10, 16, 10])
    assert _gather(scene, compiled, fields) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_off_grid_polyline_vertex_is_rejected_not_snapped():
    scene = prepare_scene(_grid_scene())
    spec = mw.CustomVoltageIntegral(mw.PolylinePath(((0.0, 0.0, 0.0), (0.0, 0.034, 0.0))))
    with pytest.raises(ValueError, match="off-grid"):
        compile_voltage_integral(scene, spec, _plane(), device="cpu")


@pytest.mark.parametrize("direction_sign", (1, -1))
@pytest.mark.parametrize("loop", (
    mw.RectangleLoop(center=(0.0, 0.003, -0.004), size=(0.0, 0.071, 0.053)),
    mw.CircleLoop(center=(0.0, 0.001, 0.002), radius=0.047, axis="x", segments=48),
    mw.PolygonLoop(((0.0, -0.055, -0.045), (0.0, 0.052, -0.035), (0.0, 0.015, 0.061))),
))
def test_current_loop_is_the_exact_dual_cell_ampere_boundary(loop, direction_sign):
    scene = prepare_scene(_grid_scene())
    plane = _plane(direction_sign)
    compiled = compile_current_integral(scene, mw.CustomCurrentIntegral(loop), plane, device="cpu")
    generator = torch.Generator().manual_seed(3)
    hy = torch.randn((21, 20, 20), generator=generator, dtype=torch.float64)   # (x node, y half, z half)
    hz = torch.randn((20, 20, 21), generator=generator, dtype=torch.float64)   # (x half, y half, z node)
    fields = {"Hy": hy, "Hz": hz}
    circulation = _gather(scene, compiled, fields)
    # Discrete curl of each enclosed dual cell; magnetic plane index along x.
    m = plane.magnetic_plane_index
    ((_coefficient, enclosed),) = compiled.loops
    expected = 0.0
    for flat in enclosed:
        i, j = divmod(flat, 17)
        y, z = i + 2, j + 2
        expected += (hz[m, y, z] - hz[m, y - 1, z]) * DX - (hy[m, y, z] - hy[m, y, z - 1]) * DX
    assert circulation == pytest.approx(direction_sign * float(expected), rel=1e-12, abs=1e-12)


def test_loop_through_a_node_and_integral_audits_are_rejected():
    scene = prepare_scene(_grid_scene())
    plane = _plane()
    with pytest.raises(ValueError, match="passes through a Yee node"):
        compile_current_integral(scene, mw.CustomCurrentIntegral(
            mw.RectangleLoop(center=(0.0, 0.0, 0.0), size=(0.0, 0.04, 0.05))), plane, device="cpu")
    labels = -np.ones((17, 17), dtype=np.int64)
    labels[:, :3] = 0          # reference along low z
    labels[4:7, 9] = 1         # terminal 1
    labels[11:13, 9] = 2       # terminal 2
    good = compile_voltage_integral(scene, mw.CustomVoltageIntegral(
        mw.AxisSegment((0.0, -0.03, 0.01), (0.0, -0.03, -0.06))), plane, device="cpu")
    assert audit_terminal_integral(good, labels, 0)["checked"] == "terminal_to_reference"
    with pytest.raises(ValueError, match="from that terminal to the reference"):
        audit_terminal_integral(good, labels, 1)
    loop = compile_current_integral(scene, mw.CustomCurrentIntegral(
        mw.RectangleLoop(center=(0.0, -0.03, 0.01), size=(0.0, 0.035, 0.025))), plane, device="cpu")
    assert audit_terminal_integral(loop, labels, 0)["checked"] == "encloses_terminal"
    cut = compile_current_integral(scene, mw.CustomCurrentIntegral(
        mw.RectangleLoop(center=(0.0, -0.035, 0.01), size=(0.0, 0.015, 0.025))), plane, device="cpu")
    with pytest.raises(ValueError, match="cuts through conductor 1"):
        audit_terminal_integral(cut, labels, 0)


# --------------------------------------------------------------------------- #
# Network algebra with a full reference-impedance matrix.                      #
# --------------------------------------------------------------------------- #
def _random_z(frequency_count=2, port_count=4, seed=11):
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn((frequency_count, port_count, port_count), generator=generator, dtype=torch.float64)
    imag = torch.randn((frequency_count, port_count, port_count), generator=generator, dtype=torch.float64)
    z = (real @ real.transpose(-2, -1) + port_count * torch.eye(port_count)) * 20 + 1j * (imag + imag.transpose(-2, -1)) * 5
    return z.to(torch.complex128)


def test_full_reference_matrix_network_round_trips_and_converts_to_mixed_mode(tmp_path):
    z = _random_z()
    frequencies = torch.tensor([1e9, 2e9], dtype=torch.float64)
    names = ("A::T1", "A::T2", "B::T1", "B::T2")
    block = torch.tensor([[60.0, 12.0], [12.0, 60.0]], dtype=torch.float64)
    matrix = torch.block_diag(block, block).to(torch.complex128).expand(2, 4, 4)
    coupled = mw.NetworkData.from_z(frequencies=frequencies, z=z, z0=50.0, port_names=names, reference_matrix=matrix)
    assert coupled.reference_matrix is not None
    assert torch.allclose(coupled.z0.real, torch.full((2, 4), 60.0, dtype=torch.float64))
    assert torch.allclose(coupled.to_z(), z, rtol=1e-10, atol=1e-9)
    diagonal = mw.NetworkData.from_z(frequencies=frequencies, z=z, z0=60.0, port_names=names,
                                     reference_matrix=torch.diag_embed(torch.full((2, 4), 60.0, dtype=torch.complex128)))
    assert torch.allclose(diagonal.s, mw.NetworkData.from_z(frequencies=frequencies, z=z, z0=60.0, port_names=names).s,
                          rtol=1e-12, atol=1e-12)
    renormalized = coupled.renormalize(50.0)
    assert renormalized.reference_matrix is None
    assert torch.allclose(renormalized.to_z(), z, rtol=1e-10, atol=1e-9)
    mixed = coupled.to_mixed_mode((("A::T1", "A::T2"), ("B::T1", "B::T2")), z0=(100.0, 25.0, 100.0, 25.0))
    reference = renormalized.to_mixed_mode((("A::T1", "A::T2"), ("B::T1", "B::T2")))
    assert torch.allclose(mixed.s, reference.s, rtol=1e-9, atol=1e-10)
    with pytest.raises(ValueError, match="diagonal per-port reference"):
        coupled.to_mixed_mode((("A::T1", "A::T2"),))
    with pytest.raises(ValueError, match="diagonal per-port reference"):
        coupled.terminate("A::T1", gamma=0.0)
    path = tmp_path / "coupled.pt"
    coupled.save(path)
    loaded = mw.NetworkData.load(path)
    assert torch.equal(loaded.reference_matrix, coupled.reference_matrix.cpu())
    assert torch.equal(loaded.s, coupled.s.cpu())
    # Generalized power waves conserve power: S is unitary for a lossless reactive Z.
    lossless = 1j * torch.imag(z)
    unitary = mw.NetworkData.from_z(frequencies=frequencies, z=lossless, z0=50.0, port_names=names, reference_matrix=matrix)
    identity = unitary.s.conj().transpose(-2, -1) @ unitary.s
    assert torch.allclose(identity, torch.eye(4, dtype=torch.complex128).expand(2, 4, 4), atol=1e-10)


def test_to_basis_matches_to_mixed_mode():
    z = _random_z()
    frequencies = torch.tensor([1e9, 2e9], dtype=torch.float64)
    network = mw.NetworkData.from_z(frequencies=frequencies, z=z, z0=50.0, port_names=("a", "b", "c", "d"))
    transform = torch.tensor([[1.0, -1.0, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 1.0, -1.0], [0, 0, 0.5, 0.5]], dtype=torch.float64)
    names = ("d(a,b)", "c(a,b)", "d(c,d)", "c(c,d)")
    based = network.to_basis(transform, z0=(100.0, 25.0, 100.0, 25.0), port_names=names)
    mixed = network.to_mixed_mode((("a", "b"), ("c", "d")))
    assert torch.allclose(based.s, mixed.s, rtol=1e-10, atol=1e-12)
    with pytest.raises(ValueError, match="terminal_transform"):
        network.to_basis("modal")


@pytest.mark.parametrize("length", [1., 1e-6])
def test_terminal_column_retains_small_power_wave_at_both_scales(length):
    from types import SimpleNamespace
    from witwin.maxwell.compiler.terminal_waveports import compile_terminal_waveport
    from witwin.maxwell.terminal_waveport import _column_scene

    scene = mw.Scene(domain=mw.Domain(bounds=((-0.1*length,.1*length),)*3),
                     grid=mw.GridSpec.uniform(.01*length), boundary=mw.BoundarySpec.pec(), device="cpu")
    for name, z, width in (("ground",-.065,1.), ("strip",.005,.04)):
        scene.add_structure(mw.Structure(mw.Box(position=(0.,0.,z*length), size=(length,width*length,.01*length)), mw.Material.pec(), name=name))
    port = mw.TerminalWavePort("P", position=(0.,0.,0.), size=(0.,.16*length,.16*length), direction="+", reference_plane=0.,
                               terminals=(mw.TerminalRef("strip"),), reference=mw.TerminalRef("ground"))
    scene.add_port(port)
    port = scene.ports[0]
    compiled = compile_terminal_waveport(scene, port, device="cpu")
    prepared = SimpleNamespace(port=port, compiled=compiled, polarization_axis="z",
                               modal_potentials=torch.ones((1,1), dtype=torch.float64),
                               drive=torch.ones((1,1,1), dtype=torch.complex128))
    manifest = SimpleNamespace(frequencies=(1e9/length,), prepared_ports=(prepared,))
    amplitude = 1e-12*length
    column = _column_scene(scene, manifest, active_port=0, active_channel=0, frequency_index=0, amplitude=amplitude)
    assert len(column.sources) == 1
    assert column.sources[0].source_time.amplitude == pytest.approx(amplitude, rel=1e-12, abs=0)
