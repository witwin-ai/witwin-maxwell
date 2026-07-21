"""Wave-level gates for the microstrip / differential-pair two-port benches (F2b).

Gate taxonomy (S0.3): **wave-level** (extraction preconditions) plus
**modal-eigensolve** routing checks.

Both scenes were BLOCKED for the whole S1 round on two stacked blockers: a
single-precision current-contour snap error that fired before the mode solve, and
the categorical inapplicability of ``WaveModeSpec('tem')`` to their inhomogeneous
(substrate + air) cross-section. F2b resolves both: the inhomogeneous interior-PEC
quasi-TEM mode routes through the quasi-static electrostatic line-mode engine
(``eps_eff = C/C0``) inside ``_assemble_vector_mode_data``, and the scenes were
rebuilt on the coax_thru precedent (measurement ports near the origin so the
contour planes stay on the Yee half-grid, conductors run through the PML so the
launched waves terminate, integer-cell node arrays so faces/contours land on Yee
nodes).

The routing checks (manifest-only, fast) pin that the microstrip / diff-pair ports
select the quasi-static engine while a UNIFORM-fill interior-PEC line (air coax)
still selects the closed-form electrostatic TEM solve -- the fallback is content
dependent, not blanket. The two-port checks mirror the coax_thru wave-level
precondition (extraction conditioning + post-solve passivity). The absolute
quasi-TEM impedance carries a documented resolution-limited gap (the discrete thin
substrate under-loads the field at feasible dx; see the F2b acceptance doc and the
microstrip gate in ``test_interior_pec_operator.py`` for the convergence record),
so this file gates the extraction quality, not the absolute eps_eff.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.excitation.modes import _tem_signal_potentials
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.waveport_sweep import resolve_waveport_run_manifest

from benchmark.scenes.rf.coax_thru import coax_thru_scene
from benchmark.scenes.rf.differential_pair import differential_pair_scene, SUBSTRATE_EPS
from benchmark.scenes.rf.microstrip_two_port import microstrip_two_port_scene

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="wave-level FDTD gate requires CUDA"
)

C0 = 299792458.0
COND_LIMIT = 10.0
PASSIVITY_SLACK = 1.10


def _port_mode_kinds(scene, frequency=1.0e9):
    prepared = prepare_scene(scene)
    manifest = resolve_waveport_run_manifest(prepared, mw.PortSweep(), (frequency,))
    out = []
    for port in manifest.prepared_ports:
        md = port.mode_data[0][0]
        out.append((md["mode_solver_kind"], float(md["effective_index"])))
    return out


# --------------------------------------------------------------------------- #
# Routing (modal-eigensolve): inhomogeneous -> quasi-static, uniform -> legacy. #
# --------------------------------------------------------------------------- #
def test_microstrip_waveport_routes_to_quasistatic_line_mode():
    kinds = _port_mode_kinds(microstrip_two_port_scene(dx=0.005, device="cuda"))
    assert len(kinds) == 2
    n_max = math.sqrt(SUBSTRATE_EPS)
    for kind, neff in kinds:
        assert kind == "quasistatic_line_torch", kind
        # A quasi-TEM slow wave: 1 < n_eff < sqrt(eps_r) (partial dielectric fill).
        assert 1.0 < neff < n_max, neff


def test_differential_pair_waveports_route_to_quasistatic_line_mode():
    kinds = _port_mode_kinds(differential_pair_scene(dx=0.005, device="cuda"))
    assert len(kinds) == 4
    n_max = math.sqrt(SUBSTRATE_EPS)
    for kind, neff in kinds:
        assert kind == "quasistatic_line_torch", kind
        assert 1.0 < neff < n_max, neff


def test_uniform_fill_interior_pec_tem_keeps_the_closed_form_electrostatic_solve():
    """Falsification of 'always quasi-static': a UNIFORM (air) coax still uses the
    closed-form electrostatic TEM solve; the quasi-static fallback fires only for the
    inhomogeneous cross-section that the closed-form path rejects."""
    kinds = _port_mode_kinds(coax_thru_scene(dx=0.005, device="cuda"))
    assert len(kinds) == 2
    for kind, neff in kinds:
        assert kind == "tem_electrostatic_torch", kind
        assert neff == pytest.approx(1.0, abs=1.0e-3)  # air-filled: n_eff = 1


def test_tem_signal_potentials_maps_conductor_count_to_drive():
    """Unit gate: single-signal -> [1], coupled pair -> even [1,1] / odd [1,-1]."""
    # One isolated conductor (a centred square) in a grounded box.
    occ_single = torch.zeros((9, 9))
    occ_single[0, :] = 1.0
    occ_single[-1, :] = 1.0
    occ_single[:, 0] = 1.0
    occ_single[:, -1] = 1.0
    occ_single[4, 4] = 1.0  # isolated signal
    assert _tem_signal_potentials(occ_single, 0) == [1.0]
    with pytest.raises(ValueError):
        _tem_signal_potentials(occ_single, 1)  # single line carries one mode

    # Two isolated conductors.
    occ_pair = torch.zeros((9, 11))
    occ_pair[0, :] = 1.0
    occ_pair[-1, :] = 1.0
    occ_pair[:, 0] = 1.0
    occ_pair[:, -1] = 1.0
    occ_pair[4, 3] = 1.0
    occ_pair[4, 7] = 1.0
    assert _tem_signal_potentials(occ_pair, 0) == [1.0, 1.0]   # even / common
    assert _tem_signal_potentials(occ_pair, 1) == [1.0, -1.0]  # odd / differential


# --------------------------------------------------------------------------- #
# Two-port extraction (wave-level precondition, coax_thru precedent).           #
# --------------------------------------------------------------------------- #
def test_microstrip_two_port_extraction_is_conditioned_and_passive():
    freqs = tuple(float(x) for x in np.linspace(0.6e9, 1.6e9, 6))
    result = mw.Simulation.fdtd(
        microstrip_two_port_scene(dx=0.005, device="cuda"),
        frequencies=freqs,
        excitations=mw.PortSweep(),
        run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=16),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()
    s = result.network.s.cpu().numpy()
    cond = float(np.max(result.network.metadata["extraction_condition_number"].cpu().numpy()))
    sv_max = max(float(np.linalg.svd(s[i], compute_uv=False).max()) for i in range(len(freqs)))

    assert cond <= COND_LIMIT, f"extraction cond(A) {cond:.3f} > {COND_LIMIT}"
    assert sv_max <= PASSIVITY_SLACK, f"max singular value {sv_max:.4f} > {PASSIVITY_SLACK}"
    # A terminated line carries a substantial through wave and a bounded reflection.
    assert float(np.abs(s[:, 1, 0]).max()) > 0.6, "no through transmission (untermimated?)"
    assert float(np.abs(s[:, 0, 0]).min()) < 0.3, "|S11| never dips (mismatch-dominated)"
    # The quasi-TEM wave is a genuine slow wave: eps_eff from arg(S21)/L exceeds air.
    length = 2.0 * 0.02  # 2 * PORT_X
    k0 = 2.0 * np.pi * np.array(freqs) / C0
    beta_meas = np.abs(np.unwrap(np.angle(s[:, 1, 0]))) / length
    eps_eff_meas = float(np.median((beta_meas / k0) ** 2))
    assert eps_eff_meas > 1.2, f"measured eps_eff {eps_eff_meas:.2f} is not a slow wave"


def test_differential_pair_four_port_shows_line_coupling():
    freqs = tuple(float(x) for x in np.linspace(0.6e9, 1.2e9, 4))
    result = mw.Simulation.fdtd(
        differential_pair_scene(dx=0.005, device="cuda"),
        frequencies=freqs,
        excitations=mw.PortSweep(),
        run_time=mw.TimeConfig.auto(steady_cycles=5, transient_cycles=12),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()
    s = result.network.s.cpu().numpy()
    cond = float(np.max(result.network.metadata["extraction_condition_number"].cpu().numpy()))
    assert s.shape[1:] == (4, 4)
    assert cond <= COND_LIMIT, f"extraction cond(A) {cond:.3f} > {COND_LIMIT}"

    # Line-to-line coupling: the near-end single-ended S21 (p1 -> p2) is non-zero.
    assert float(np.abs(s[:, 1, 0]).max()) > 0.05, "no line-to-line coupling"

    # Mixed-mode conversion: even and odd modes propagate differently, so the
    # differential and common insertions differ (a coupled line, not two isolated).
    m = np.array(
        [[1, -1, 0, 0], [0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=float
    ) / np.sqrt(2.0)
    mixed = np.stack([m @ s[i] @ m.T for i in range(len(freqs))])
    sdd21 = np.median(np.abs(mixed[:, 1, 0]))
    scc21 = np.median(np.abs(mixed[:, 3, 2]))
    assert abs(sdd21 - scc21) > 0.02, f"|Sdd21| {sdd21:.3f} == |Scc21| {scc21:.3f} (uncoupled)"
