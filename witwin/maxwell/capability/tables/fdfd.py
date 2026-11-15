"""``Simulation.fdfd(...)``: the frequency-domain curl-curl runtime.

FDFD solves one linear operator per frequency from per-axis diagonal media,
point/uniform/lumped current sources and per-face none/PML/PEC boundaries.
Everything it would otherwise drop, approximate or refuse only after CUDA work
is refused here.
"""
from __future__ import annotations

from ..keys import IN_SIMULATION_MONITOR_TYPES, MONITOR_TYPES
from ..model import inherent, supported, table, unsupported
from ._shared import (
    ALL_SOURCES, CORE_SURFACE_TESTS, REGISTRY_TEST, breakdown_requires_fdtd,
    conductor_loss_gradients_require_layered_mom, coordinates_require_fdtd, core_surface_fields,
    homogeneous_background_or_plane_modes, layer_stack_requires_layered_mom, networks_require_fdtd,
    optical_assignment, planar_pec_requires_sheet_runtime, thin_wires_require_fdtd,
)

_FDFD = "tests/fdfd/test_fdfd_silent_ignore_refusals.py::test_fdfd_refuses_scene_feature_at_construction"
_ALL_IN_ONE = "tests/fdfd/test_fdfd_silent_ignore_refusals.py::test_fdfd_lists_every_unsupported_feature_in_one_error"
_USE_FDTD = "use Simulation.fdtd(...), which supports it"
_OBSERVED = ("PermittivityMonitor", "MediumMonitor")


def _conductor(key, kind):
    return unsupported(key, f"FDFD has no {kind} conductor model and would solve it as vacuum", _USE_FDTD,
                       tests=(_FDFD,))


TABLE = table(
    homogeneous_background_or_plane_modes("background.homogeneous"),
    layer_stack_requires_layered_mom(),
    coordinates_require_fdtd(),
    optical_assignment(),
    conductor_loss_gradients_require_layered_mom(),
    unsupported("symmetry", "Scene(symmetry=...) is not supported by FDFD", _USE_FDTD, tests=(_FDFD,)),
    unsupported("subpixel.pec_conformal",
                "{owner} is a time-domain partial-fill update; FDFD would staircase the PEC",
                'use pec="staircase" or Simulation.fdtd(...)', tests=(_FDFD,)),
    supported("boundary.none", "boundary.pml", "boundary.pec"),
    tuple(unsupported(f"boundary.{kind}", "FDFD supports per-face 'none', 'pml' and 'pec' boundaries only",
                      _USE_FDTD, tests=(_FDFD,))
          for kind in ("periodic", "bloch", "pmc", "mur", "mode_abc")),
    unsupported("geometry.surface_mesh",
                "FDFD has no zero-volume SurfaceMesh trace or thin-sheet operator and would drop the surface",
                _USE_FDTD, tests=(_FDFD,)),
    planar_pec_requires_sheet_runtime(),
    _conductor("material.lossy_metal", "LossyMetalMedium"),
    _conductor("material.surface_impedance", "SurfaceImpedanceMedium"),
    _conductor("material.copper_foil", "CopperFoil"),
    unsupported("material.gyromagnetic",
                "FDFD does not support GyromagneticFerrite media: the frequency-domain solver would ingest only the "
                "diagonal background permeability and drop the non-reciprocal off-diagonal Polder gyrotropy",
                "use Simulation.fdtd(...), whose magnetization-ADE runtime carries the gyromagnetic tensor",
                tests=(_FDFD,)),
    inherent("material.nonlinear",
             "FDFD does not support Kerr nonlinear media (chi3, chi2, TPA): a frequency-domain solve has no single "
             "linear operator for a field-amplitude-dependent permittivity",
             "use Simulation.fdtd(...), which models the nonlinearity directly", tests=(_FDFD,)),
    unsupported("material.full_tensor",
                "FDFD does not support full off-diagonal Tensor3x3 permittivity: the curl-curl assembly forms only "
                "per-axis diagonal material components",
                "use Simulation.fdtd(...), which models the coupled tensor", tests=(_FDFD,)),
    inherent("material.modulation",
             "FDFD would solve the time-modulated (ModulationSpec) material as its static permittivity", _USE_FDTD,
             tests=(_FDFD,)),
    breakdown_requires_fdtd(tests=(_FDFD,)),
    supported("material.medium2d", "material.pmc", "material.sigma_e", "material.trainable_mu_r",
              "material.perturbation"),
    core_surface_fields("FDFD", tests=CORE_SURFACE_TESTS),
    supported("source", "monitor", "port", "circuit", "electrostatic_terminal", "charge_density"),
    unsupported("lumped_element", "Lumped elements ({names}) are not stamped into the FDFD operator", _USE_FDTD,
                tests=(_FDFD,)),
    networks_require_fdtd(),
    thin_wires_require_fdtd(),
    unsupported("internal_absorber", "{count} InternalAbsorber(s) are not applied by FDFD", _USE_FDTD,
                tests=(_FDFD,)),
    supported(*(key for key in ALL_SOURCES if key != "source.MaxwellPlaneModes")),
    homogeneous_background_or_plane_modes("source.MaxwellPlaneModes"),
    supported("source.injection_tfsf", "source.injection_axis"),
    unsupported("source.cw_frequency_mismatch",
                "{owner} declares CW(frequency={cw_frequency:.12g}) but FDFD solves at {frequencies} Hz and uses "
                "only the CW amplitude and phase",
                "set CW.frequency to the solve frequency, or source_time=None to follow every solve frequency",
                tests=(_FDFD,)),
    supported(*(f"monitor.{name}" for name in _OBSERVED)),
    tuple(unsupported(f"monitor.{name}", "{owner} is not supported by FDFD",
                      "use Simulation.fdtd(...) or FDFDConfig.observations",
                      tests=(_ALL_IN_ONE, "tests/monitors/in_simulation/test_in_simulation_monitors_unit.py::"
                                          "test_non_fdtd_simulations_reject_in_simulation_monitors")
                      if name in IN_SIMULATION_MONITOR_TYPES else (_ALL_IN_ONE,))
          for name in MONITOR_TYPES if name not in _OBSERVED),
    supported("port.LumpedPort", "port.TerminalPort", "port.CoaxialLumpedPort"),
    tuple(unsupported(f"port.{name}", "FDFD neither launches nor projects port modes", _USE_FDTD, tests=(_FDFD,))
          for name in ("ModePort", "WavePort", "TerminalWavePort")),
    unsupported("port.termination", "the termination load is not stamped into the FDFD operator", _USE_FDTD,
                tests=(_FDFD,)),
    unsupported("port.wire_binding", "FDFD has no thin-wire runtime for the wire binding", _USE_FDTD,
                tests=(REGISTRY_TEST,)),
    supported("electrostatic_terminal.trainable_geometry", "electrostatic_terminal.floating"),
)
