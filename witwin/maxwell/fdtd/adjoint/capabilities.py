"""Machine-readable native CUDA FDTD adjoint capability inventory."""

from __future__ import annotations

NATIVE_ADJOINT_CAPABILITIES = {
    "STANDARD": {"complex": False, "cpml": False},
    "CPML": {"complex": False, "cpml": True},
    "CONDUCTIVE": {"complex": False, "cpml": True, "electric_conductivity": True},
    "KERR": {"complex": False, "cpml": True, "chi3": True},
    "GENERAL_NONLINEAR": {
        "complex": False,
        "standard_or_cpml": True,
        "chi2": True,
        "chi3": True,
        "two_photon_absorption": True,
    },
    "FULL_ANISO": {"complex": False, "cpml": True, "off_diagonal_eps": True},
    "BLOCH": {"complex": True, "cpml": False},
    "BLOCH_DISPERSIVE": {"complex": True, "cpml": False, "electric_ade": True, "magnetic_ade": True},
    "MIXED_BLOCH_CPML": {"complex": True, "cpml": True, "single_pml_axis": True},
    "DISPERSIVE": {"electric_ade": True, "magnetic_ade": True},
    "TFSF": {"tfsf": True},
    "GRATING_TFSF": {"complex": True, "cpml": True, "tfsf": True},
    "WIRE_STANDARD": {"complex": False, "cpml": False, "thin_wire": True},
    "WIRE_CPML": {"complex": False, "cpml": True, "thin_wire": True},
}


def native_adjoint_capabilities() -> dict[str, dict[str, bool]]:
    """Return a defensive copy for tests and diagnostics."""
    return {name: dict(features) for name, features in NATIVE_ADJOINT_CAPABILITIES.items()}
