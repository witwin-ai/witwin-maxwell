"""
Cross-validation test suite: FDFD vs FDTD frequency-domain results.

Both solvers solve Maxwell's equations on identical Yee grids. FDFD solves
directly in frequency domain; FDTD time-steps then extracts frequency content
via running DFT. Their results should match in pattern (normalized correlation).

Two simple benchmark scenes:
  1. Center point source in free space
  2. Center point source + single dielectric cube

Each test saves comparison PNG slices (FDFD row, FDTD row, difference row)
of |E| field through the domain center, to test_output/fields/.

Usage:
    pytest tests/validation/cross_solver/test_fdfd_vs_fdtd.py -v --gpu
    python tests/validation/cross_solver/test_fdfd_vs_fdtd.py                  # standalone, all tests
    python tests/validation/cross_solver/test_fdfd_vs_fdtd.py --visual          # with matplotlib.show()
"""

import os
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TESTS_ROOT = Path(__file__).resolve()
while TESTS_ROOT.name != "tests" and TESTS_ROOT.parent != TESTS_ROOT:
    TESTS_ROOT = TESTS_ROOT.parent
if TESTS_ROOT.name != "tests":
    raise RuntimeError("Unable to locate tests root directory.")

import witwin.maxwell as mw

OUTPUT_DIR = str(TESTS_ROOT / "test_output" / "fields")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def interpolate_yee_to_center(Ex, Ey, Ez):
    """Interpolate Yee-staggered fields to cell centers, return |E| array."""
    Ex_int = 0.5 * (Ex[:, :-1, :] + Ex[:, 1:, :])
    Ex_int = 0.5 * (Ex_int[:, :, :-1] + Ex_int[:, :, 1:])
    Ey_int = 0.5 * (Ey[:-1, :, :] + Ey[1:, :, :])
    Ey_int = 0.5 * (Ey_int[:, :, :-1] + Ey_int[:, :, 1:])
    Ez_int = 0.5 * (Ez[:-1, :, :] + Ez[1:, :, :])
    Ez_int = 0.5 * (Ez_int[:, :-1, :] + Ez_int[:, 1:, :])
    return np.sqrt(np.abs(Ex_int)**2 + np.abs(Ey_int)**2 + np.abs(Ez_int)**2)


def normalized_correlation(a, b):
    """Normalized cross-correlation (1 = identical pattern, 0 = uncorrelated)."""
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)
    a_flat -= a_flat.mean()
    b_flat -= b_flat.mean()
    norm = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if norm < 1e-30:
        return 0.0
    return float(np.dot(a_flat, b_flat) / norm)


def peak_location_match(a, b, max_offset=3):
    """Check if peak locations match within max_offset cells."""
    peak_a = np.unravel_index(np.argmax(np.abs(a)), a.shape)
    peak_b = np.unravel_index(np.argmax(np.abs(b)), b.shape)
    offset = max(abs(pa - pb) for pa, pb in zip(peak_a, peak_b))
    return offset <= max_offset, offset, peak_a, peak_b


def trim_common(a, b):
    """Trim two 3D arrays to their common shape."""
    s = tuple(min(sa, sb) for sa, sb in zip(a.shape, b.shape))
    return a[:s[0], :s[1], :s[2]], b[:s[0], :s[1], :s[2]]


# ---------------------------------------------------------------------------
# Save field comparison PNGs
# ---------------------------------------------------------------------------

def save_field_comparison(fdfd_fields, fdtd_fields, name, scene, show=False):
    """Save |E| cross-section comparison PNGs (FDFD vs FDTD vs diff)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # |E| at cell centers
    fdfd_E = interpolate_yee_to_center(fdfd_fields['Ex'], fdfd_fields['Ey'], fdfd_fields['Ez'])
    fdtd_E = interpolate_yee_to_center(fdtd_fields['Ex'], fdtd_fields['Ey'], fdtd_fields['Ez'])
    fdfd_E, fdtd_E = trim_common(fdfd_E, fdtd_E)

    # Normalize each to its own max for pattern comparison
    fdfd_max = fdfd_E.max()
    fdtd_max = fdtd_E.max()
    fdfd_norm = fdfd_E / (fdfd_max + 1e-30)
    fdtd_norm = fdtd_E / (fdtd_max + 1e-30)

    x0, x1 = scene.domain.domain_range[0], scene.domain.domain_range[1]
    y0, y1 = scene.domain.domain_range[2], scene.domain.domain_range[3]
    z0, z1 = scene.domain.domain_range[4], scene.domain.domain_range[5]
    Nx, Ny, Nz = fdfd_E.shape
    mid = [Nx // 2, Ny // 2, Nz // 2]

    slice_specs = [
        ('z', mid[2], lambda a: a[:, :, mid[2]], [x0, x1, y0, y1], 'X [m]', 'Y [m]'),
        ('x', mid[0], lambda a: a[mid[0], :, :], [y0, y1, z0, z1], 'Y [m]', 'Z [m]'),
        ('y', mid[1], lambda a: a[:, mid[1], :], [x0, x1, z0, z1], 'X [m]', 'Z [m]'),
    ]

    # Per-component slices (Ex, Ey, Ez individual)
    for comp in ['Ex', 'Ey', 'Ez']:
        a, b = trim_common(np.abs(fdfd_fields[comp]), np.abs(fdtd_fields[comp]))
        a_max = a.max() + 1e-30
        b_max = b.max() + 1e-30

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{name} -|{comp}| @ z=0 (normalized to own max)', fontsize=13)

        # z-mid slice
        iz = a.shape[2] // 2
        sl_a = a[:, :, iz] / a_max
        sl_b = b[:, :, iz] / b_max

        axes[0].imshow(sl_a.T, origin='lower', extent=[x0, x1, y0, y1],
                       vmin=0, vmax=1, cmap='hot')
        axes[0].set_title(f'FDFD |{comp}| (max={a_max:.3e})')
        axes[0].set_xlabel('X [m]'); axes[0].set_ylabel('Y [m]')

        axes[1].imshow(sl_b.T, origin='lower', extent=[x0, x1, y0, y1],
                       vmin=0, vmax=1, cmap='hot')
        axes[1].set_title(f'FDTD |{comp}| (max={b_max:.3e})')
        axes[1].set_xlabel('X [m]'); axes[1].set_ylabel('Y [m]')

        diff = np.abs(sl_a - sl_b)
        im = axes[2].imshow(diff.T, origin='lower', extent=[x0, x1, y0, y1],
                            vmin=0, vmax=0.5, cmap='RdBu_r')
        axes[2].set_title(f'|diff| (normalized)')
        axes[2].set_xlabel('X [m]'); axes[2].set_ylabel('Y [m]')
        plt.colorbar(im, ax=axes[2], shrink=0.8)

        for ax in axes:
            ax.set_aspect('equal')
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f'{name}_{comp}_z.png')
        fig.savefig(path, dpi=120)
        if show:
            plt.show()
        plt.close(fig)
        print(f"  saved: {path}")

    # |E| total: 3 slice directions, each with FDFD / FDTD / diff rows
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'{name} -|E| cross-sections (each normalized to own max)', fontsize=14)

    for col, (axis_name, idx, slicer, extent, xlabel, ylabel) in enumerate(slice_specs):
        sl_fdfd = slicer(fdfd_norm)
        sl_fdtd = slicer(fdtd_norm)
        sl_diff = np.abs(sl_fdfd - sl_fdtd)

        axes[0, col].imshow(sl_fdfd.T, origin='lower', extent=extent,
                            vmin=0, vmax=1, cmap='hot')
        axes[0, col].set_title(f'FDFD |E| @ {axis_name}=0 (max={fdfd_max:.3e})')
        axes[0, col].set_xlabel(xlabel); axes[0, col].set_ylabel(ylabel)

        axes[1, col].imshow(sl_fdtd.T, origin='lower', extent=extent,
                            vmin=0, vmax=1, cmap='hot')
        axes[1, col].set_title(f'FDTD |E| @ {axis_name}=0 (max={fdtd_max:.3e})')
        axes[1, col].set_xlabel(xlabel); axes[1, col].set_ylabel(ylabel)

        im = axes[2, col].imshow(sl_diff.T, origin='lower', extent=extent,
                                 vmin=0, vmax=0.5, cmap='RdBu_r')
        axes[2, col].set_title(f'|diff| @ {axis_name}=0')
        axes[2, col].set_xlabel(xlabel); axes[2, col].set_ylabel(ylabel)

    for ax in axes.flat:
        ax.set_aspect('equal')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'{name}_E_total.png')
    fig.savefig(path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    print(f"  saved: {path}")


# ---------------------------------------------------------------------------
# Scene factory
# ---------------------------------------------------------------------------

def make_scene_pair(domain_half, resolution, pml_thickness, pml_strength,
                    source_pos, source_width, source_amp, source_pol,
                    geometry_fn=None, device='cuda', source_frequency=1e9,
                    subpixel_samples=1):
    """Build matched FDFD / FDTD scenes with identical grid and geometry."""
    def _build(add_source_fn):
        scene = mw.Scene(
            domain=mw.Domain(
                bounds=(
                    (-domain_half, domain_half),
                    (-domain_half, domain_half),
                    (-domain_half, domain_half),
                )
            ),
            grid=mw.GridSpec.uniform(resolution),
            boundary=mw.BoundarySpec.pml(num_layers=pml_thickness, strength=pml_strength),
            subpixel_samples=subpixel_samples,
            device=device,
        )
        if geometry_fn is not None:
            geometry_fn(scene)
        add_source_fn(scene)
        return scene

    scene_fdfd = _build(
        lambda s: s.add_source(
            mw.PointDipole(
                position=tuple(source_pos),
                width=source_width,
                polarization=tuple(source_pol),
                source_time=mw.CW(frequency=source_frequency, amplitude=source_amp),
                name="src",
            )
        )
    )
    scene_fdtd = _build(
        lambda s: s.add_source(
            mw.PointDipole(
                position=tuple(source_pos),
                width=source_width,
                polarization=tuple(source_pol),
                source_time=mw.CW(frequency=source_frequency, amplitude=source_amp),
                name="src",
            )
        )
    )

    return scene_fdfd, scene_fdtd


# ---------------------------------------------------------------------------
# Run solvers
# ---------------------------------------------------------------------------

def run_fdfd(scene, frequency, max_iter=5000, tol=1e-6, restart=200):
    """Run FDFD (GMRES) and return raw Yee-grid complex fields {Ex, Ey, Ez}."""
    result = mw.Simulation.fdfd(
        scene,
        frequency=frequency,
        solver=mw.GMRES(max_iter=max_iter, tol=tol, restart=restart),
        enable_plot=False,
        verbose=True,
    ).run()
    fdfd = result.solver
    if not fdfd.converged:
        print(f"WARNING: FDFD did not converge (residual={getattr(fdfd, 'final_residual', '?')})")
    fields = {name.title(): tensor.detach().cpu().numpy() for name, tensor in result.fields.items()}
    return fields, fdfd


def run_fdtd(scene, frequency, num_cycles=25, transient_cycles=20, dft_window='hanning'):
    """Run FDTD with running DFT and return complex fields {Ex, Ey, Ez}."""
    result = mw.Simulation.fdtd(
        scene,
        frequencies=[frequency],
        run_time=mw.TimeConfig.auto(steady_cycles=num_cycles, transient_cycles=transient_cycles),
        spectral_sampler=mw.SpectralSampler(window=dft_window),
        full_field_dft=True,
    ).run()
    fdtd = result.solver
    print(f"FDTD: {result.stats()['time_steps']} steps, dt={fdtd.dt:.4e}")
    fields = {name.title(): tensor.detach().cpu().numpy() for name, tensor in result.fields.items()}
    return fields, fdtd


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def compare_fields(fdfd_fields, fdtd_fields, scene, name, label="",
                   verbose=True, show=False):
    """Compare fields, save PNG slices, return metrics dict."""
    results = {}

    for comp in ['Ex', 'Ey', 'Ez']:
        a, b = trim_common(fdfd_fields[comp], fdtd_fields[comp])
        mag_a, mag_b = np.abs(a), np.abs(b)
        corr = normalized_correlation(mag_a, mag_b)
        peak_ok, peak_off, pa, pb = peak_location_match(mag_a, mag_b)
        results[comp] = {'correlation': corr, 'peak_offset': peak_off, 'peak_match': peak_ok}
        if verbose:
            print(f"  {label}{comp}: corr={corr:.4f}  peak_off={peak_off}  "
                  f"fdfd_max={mag_a.max():.4e}  fdtd_max={mag_b.max():.4e}")

    # |E| at cell centers
    fdfd_E = interpolate_yee_to_center(fdfd_fields['Ex'], fdfd_fields['Ey'], fdfd_fields['Ez'])
    fdtd_E = interpolate_yee_to_center(fdtd_fields['Ex'], fdtd_fields['Ey'], fdtd_fields['Ez'])
    fdfd_E, fdtd_E = trim_common(fdfd_E, fdtd_E)

    corr = normalized_correlation(fdfd_E, fdtd_E)
    peak_ok, peak_off, pa, pb = peak_location_match(fdfd_E, fdtd_E)
    results['|E|'] = {'correlation': corr, 'peak_offset': peak_off, 'peak_match': peak_ok}
    if verbose:
        print(f"  {label}|E|: corr={corr:.4f}  peak_off={peak_off}  "
              f"fdfd_max={fdfd_E.max():.4e}  fdtd_max={fdtd_E.max():.4e}")

    # Save PNG slices of electric field
    save_field_comparison(fdfd_fields, fdtd_fields, name, scene, show=show)

    return results


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestFDFDvsFDTD:
    """Cross-validation: FDFD frequency-domain vs FDTD running-DFT."""

    # Current thresholds with GMRES solver.
    # GMRES convergence is the bottleneck -raise thresholds as solver improves.
    MIN_CORRELATION = 0.70
    MAX_PEAK_OFFSET = 12

    def _assert_match(self, results, label=""):
        e = results['|E|']
        assert e['correlation'] >= self.MIN_CORRELATION, \
            f"{label}|E| correlation {e['correlation']:.4f} < {self.MIN_CORRELATION}"
        assert e['peak_offset'] <= self.MAX_PEAK_OFFSET, \
            f"{label}|E| peak offset {e['peak_offset']} > {self.MAX_PEAK_OFFSET}"

    # ------------------------------------------------------------------
    # Case 1: center point source, free space
    # ------------------------------------------------------------------
    def test_point_source_free_space(self):
        """Single z-polarized point source at origin, no objects."""
        freq = 1e9
        scene_fdfd, scene_fdtd = make_scene_pair(
            domain_half=0.64, resolution=0.04,
            pml_thickness=8, pml_strength=1e6,
            source_pos=[0, 0, 0], source_width=0.05,
            source_amp=100.0, source_pol=[0, 0, 1],
            source_frequency=freq,
        )
        fdfd_fields, _ = run_fdfd(scene_fdfd, freq)
        fdtd_fields, _ = run_fdtd(scene_fdtd, freq)
        assert fdfd_fields is not None, "FDFD returned None"
        assert fdtd_fields is not None, "FDTD returned None"
        results = compare_fields(fdfd_fields, fdtd_fields, scene_fdfd,
                                 name="free_space", label="free_space ")
        self._assert_match(results, "free_space ")

    # ------------------------------------------------------------------
    # Case 2: center point source + single dielectric cube
    # ------------------------------------------------------------------
    def test_point_source_with_cube(self):
        """Single z-polarized point source + one dielectric cube (eps_r=4)."""
        freq = 1e9

        def add_cube(scene):
            scene.add_structure(
                mw.Structure(
                    name="cube",
                    geometry=mw.Box(position=(0, 0.3, 0), size=(0.3, 0.3, 0.3)),
                    material=mw.Material(eps_r=4.0),
                )
            )

        scene_fdfd, scene_fdtd = make_scene_pair(
            domain_half=0.64, resolution=0.04,
            pml_thickness=8, pml_strength=1e6,
            source_pos=[0, 0, 0], source_width=0.05,
            source_amp=100.0, source_pol=[0, 0, 1],
            geometry_fn=add_cube,
            source_frequency=freq,
        )
        fdfd_fields, _ = run_fdfd(scene_fdfd, freq)
        fdtd_fields, _ = run_fdtd(scene_fdtd, freq)
        assert fdfd_fields is not None, "FDFD returned None"
        assert fdtd_fields is not None, "FDTD returned None"
        results = compare_fields(fdfd_fields, fdtd_fields, scene_fdfd,
                                 name="cube", label="cube ")
        self._assert_match(results, "cube ")

    # ------------------------------------------------------------------
    # Case 3: point source + dielectric cube with polarized subpixel
    # ------------------------------------------------------------------
    def test_polarized_dielectric_consistency(self):
        """FDFD and FDTD stay consistent under polarized (Kottke) subpixel averaging.

        Both solvers consume the same node-based normal-aware material components
        (compiler emits them once; each solver re-averages node -> Yee edge with its
        own arithmetic stencil), so enabling ``averaging="polarized"`` must keep the
        two runtimes cross-consistent. This locks the FDFD-for-free path (design 6).
        """
        freq = 1e9

        def add_cube(scene):
            scene.add_structure(
                mw.Structure(
                    name="cube",
                    geometry=mw.Box(position=(0, 0.3, 0), size=(0.3, 0.3, 0.3)),
                    material=mw.Material(eps_r=4.0),
                )
            )

        scene_fdfd, scene_fdtd = make_scene_pair(
            domain_half=0.64, resolution=0.04,
            pml_thickness=8, pml_strength=1e6,
            source_pos=[0, 0, 0], source_width=0.05,
            source_amp=100.0, source_pol=[0, 0, 1],
            geometry_fn=add_cube,
            source_frequency=freq,
            subpixel_samples=mw.SubpixelSpec(averaging="polarized"),
        )
        # Both scenes must actually carry the polarized policy.
        assert scene_fdfd.subpixel.averaging == "polarized"
        assert scene_fdtd.subpixel.averaging == "polarized"

        fdfd_fields, _ = run_fdfd(scene_fdfd, freq)
        fdtd_fields, _ = run_fdtd(scene_fdtd, freq)
        assert fdfd_fields is not None, "FDFD returned None"
        assert fdtd_fields is not None, "FDTD returned None"
        results = compare_fields(fdfd_fields, fdtd_fields, scene_fdfd,
                                 name="cube_polarized", label="cube_polarized ")
        self._assert_match(results, "cube_polarized ")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='FDFD vs FDTD cross-validation')
    parser.add_argument('--visual', action='store_true',
                        help='Also call plt.show() for interactive viewing')
    args = parser.parse_args()

    if args.visual:
        matplotlib.use('TkAgg')

    suite = TestFDFDvsFDTD()
    tests = [m for m in dir(suite) if m.startswith('test_')]
    passed, failed = 0, 0
    for test_name in sorted(tests):
        print(f"\n{'='*60}")
        print(f"Running {test_name}...")
        print(f"{'='*60}")
        try:
            getattr(suite, test_name)()
            print(f"  -> PASSED")
            passed += 1
        except Exception as e:
            print(f"  -> FAILED: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed+failed}")
    print(f"PNGs saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'='*60}")
