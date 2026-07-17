"""End-to-end antenna matching-block scenario (plan 03 section 8.2 / section 10).

An antenna's realized gain is its intrinsic gain scaled by the port mismatch
efficiency 1 - |Gamma|^2. Inserting a lossless external matching network in front
of the feed transforms the antenna's input reflection Gamma_L to a new value
Gamma_in and, at constant accepted (radiated) power, leaves the intrinsic gain
untouched while lifting the realized gain by (1 - |Gamma_in|^2)/(1 - |Gamma_L|^2).

This closes the loop through ``Result.antenna``: the embedded matching block is a
first-class ``mw.NetworkData`` two-port; its S-parameters predict Gamma_in via the
standard load-cascade formula; and the realized-gain shift that ``Result.antenna``
reports for the matched port must equal that network prediction.

The matching network is applied to the driven port's power waves rather than to a
port inside the FDTD run because the embedded-network contract forbids attaching a
network to an excited port (a port may be driven or network-terminated, not both).
The radiated near field is supplied by a first-class closed-surface monitor, the
same construction the antenna postprocessing is validated against.
"""

from __future__ import annotations

import math

import torch

import witwin.maxwell as mw


_C0 = 299792458.0
_MU0 = 4.0 * math.pi * 1e-7
_EPS0 = 1.0 / (_MU0 * _C0**2)
_FREQUENCY = 1.0e9
_Z0 = 50.0


def _closed_surface_scene():
    from types import SimpleNamespace

    surface = mw.ClosedSurfaceMonitor.box(
        "nf2ff",
        position=(0.0, 0.0, 0.0),
        size=(0.4, 0.4, 0.4),
        frequencies=(_FREQUENCY,),
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
        grid=mw.GridSpec.uniform(0.1),
        device="cpu",
    ).add_monitor(surface)
    coords = torch.linspace(-0.3, 0.3, 7, dtype=torch.float64)
    monitors = {}
    for face_index, face in enumerate(surface.faces):
        coord_names = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}[face.axis]
        payload = {
            "kind": "plane",
            "fields": face.fields,
            "components": {},
            "samples": 8,
            "frequency": _FREQUENCY,
            "frequencies": (_FREQUENCY,),
            "axis": face.axis,
            "position": face.plane_position,
            "compute_flux": False,
            "normal_direction": face.normal_direction,
            coord_names[0]: coords,
            coord_names[1]: coords,
            "coords": (coords, coords),
        }
        for component_index, component in enumerate(face.fields):
            amplitude = (face_index + 1) * (component_index + 1)
            data = torch.full(
                (coords.numel(), coords.numel()),
                complex(amplitude, 0.1 * amplitude),
                dtype=torch.complex128,
            )
            payload[component] = data
            payload["components"][component] = {
                "data": data,
                "coords": (coords, coords),
                "plane_index": 0,
                "plane_indices": (0,),
                "plane_weights": (1.0,),
                "plane_positions": (face.plane_position,),
            }
        monitors[face.name] = payload
    return scene, monitors, SimpleNamespace(c=_C0, eps0=_EPS0, mu0=_MU0)


def _port_from_reflection(gamma: complex, *, accepted_power: float = 1.0) -> mw.PortData:
    """Build a driven port with the given input reflection and accepted power."""

    magnitude = abs(gamma)
    incident = accepted_power / (1.0 - magnitude * magnitude)
    a = torch.tensor([complex(math.sqrt(incident), 0.0)], dtype=torch.complex128)
    b = a * torch.tensor([gamma], dtype=torch.complex128)
    return mw.PortData.from_power_waves(
        port_name="feed",
        frequencies=torch.tensor([_FREQUENCY], dtype=torch.float64),
        a=a,
        b=b,
        z0=_Z0,
    )


def _series_reactance_matching_block(reactance_ohms: float) -> mw.NetworkData:
    """A lossless series-reactance two-port as a first-class matching network.

    ABCD [[1, jX], [0, 1]] maps to the unitary scattering matrix
    S11 = S22 = z/(z+2), S21 = S12 = 2/(z+2) with z = jX/Z0.
    """

    z = 1j * reactance_ohms / _Z0
    s11 = z / (z + 2.0)
    s21 = 2.0 / (z + 2.0)
    s = torch.tensor(
        [[[s11, s21], [s21, s11]]],
        dtype=torch.complex128,
    )
    return mw.NetworkData(
        frequencies=torch.tensor([_FREQUENCY], dtype=torch.float64),
        s=s,
        z0=_Z0,
        port_names=("in", "out"),
    )


def _input_reflection_through_block(block: mw.NetworkData, load_gamma: complex) -> complex:
    """Cascade a two-port matching block with a load reflection Gamma_L."""

    s = block.s[0]
    s11 = complex(s[0, 0])
    s12 = complex(s[0, 1])
    s21 = complex(s[1, 0])
    s22 = complex(s[1, 1])
    return s11 + s12 * s21 * load_gamma / (1.0 - s22 * load_gamma)


def _realized_and_gain(result, port):
    data = result.antenna(
        surface="nf2ff",
        driven_port=port,
        theta_points=15,
        phi_points=21,
        radius=4.0,
    )
    return data


def test_matching_block_lifts_realized_gain_by_network_predicted_mismatch_ratio():
    scene, monitors, solver = _closed_surface_scene()
    result = mw.Result(
        method="fdtd",
        scene=scene,
        frequency=_FREQUENCY,
        solver=solver,
        monitors=monitors,
        ports={},
    )

    # Antenna input impedance 50 + j60 Ohm: a reactive mismatch a lossless series
    # reactance can conjugate-match. Gamma_L = j60 / (100 + j60).
    z_load = complex(50.0, 60.0)
    load_gamma = (z_load - _Z0) / (z_load + _Z0)
    accepted_power = 1.0 - abs(load_gamma) ** 2

    bare_port = _port_from_reflection(load_gamma, accepted_power=accepted_power)
    torch.testing.assert_close(
        bare_port.reflection_coefficient,
        torch.tensor([load_gamma], dtype=torch.complex128),
    )

    # The matching block is a real NetworkData two-port; its S-parameters predict
    # the matched input reflection independently of the antenna postprocessing.
    block = _series_reactance_matching_block(-60.0)
    matched_gamma = _input_reflection_through_block(block, load_gamma)
    # The block genuinely improves the match toward a conjugate match.
    assert abs(matched_gamma) < abs(load_gamma)
    assert abs(matched_gamma) < 1.0e-9

    matched_port = _port_from_reflection(matched_gamma, accepted_power=accepted_power)

    bare = _realized_and_gain(result, bare_port)
    matched = _realized_and_gain(result, matched_port)

    # Intrinsic gain (referenced to accepted/radiated power) is invariant under an
    # external lossless match; only the realized gain moves.
    torch.testing.assert_close(matched.gain, bare.gain, rtol=1.0e-10, atol=0.0)

    predicted_ratio = (1.0 - abs(matched_gamma) ** 2) / (1.0 - abs(load_gamma) ** 2)
    measured_ratio = (matched.realized_gain / bare.realized_gain)
    torch.testing.assert_close(
        measured_ratio,
        torch.full_like(measured_ratio, predicted_ratio),
        rtol=1.0e-9,
        atol=0.0,
    )

    # The realized-gain lift equals the mismatch-efficiency improvement that the
    # network cascade predicts, reported through Result.antenna.
    torch.testing.assert_close(
        matched.mismatch_efficiency / bare.mismatch_efficiency,
        torch.full_like(bare.mismatch_efficiency, predicted_ratio),
        rtol=1.0e-9,
        atol=0.0,
    )
    assert float(bare.mismatch_efficiency[0]) < 0.8
    assert float(matched.mismatch_efficiency[0]) > 0.999
