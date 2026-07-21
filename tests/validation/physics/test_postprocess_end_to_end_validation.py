import math
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import witwin.maxwell as mw
from witwin.maxwell.postprocess import (
    NearFieldFarFieldTransformer,
    StrattonChuPropagator,
    compute_bistatic_rcs,
    compute_directivity,
    equivalent_surface_currents_from_monitor,
)


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "test_output" / "validation" / "postprocess"


def _to_numpy(value, dtype=None):
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if dtype is not None:
        return array.astype(dtype, copy=False)
    return array


def _closed_surface_box_monitor(name: str, half: float, frequency: float) -> mw.ClosedSurfaceMonitor:
    return mw.ClosedSurfaceMonitor.box(
        name,
        position=(0.0, 0.0, 0.0),
        size=(2.0 * half, 2.0 * half, 2.0 * half),
        frequencies=(frequency,),
    )


def _nonrect_closed_surface_monitor(name: str, frequency: float) -> mw.ClosedSurfaceMonitor:
    half_x = 0.10
    half_y_left = 0.12
    half_y_right = 0.08
    half_z = 0.10
    faces = (
        mw.FinitePlaneMonitor(
            "x_neg",
            position=(-half_x, 0.0, 0.0),
            size=(0.0, 2.0 * half_y_left, 2.0 * half_z),
            fields=("Ey", "Ez", "Hy", "Hz"),
            frequencies=(frequency,),
            normal_direction="-",
        ),
        mw.FinitePlaneMonitor(
            "x_pos",
            position=(half_x, 0.0, 0.0),
            size=(0.0, 2.0 * half_y_right, 2.0 * half_z),
            fields=("Ey", "Ez", "Hy", "Hz"),
            frequencies=(frequency,),
            normal_direction="+",
        ),
        mw.FinitePlaneMonitor(
            "x_step_upper",
            position=(0.0, 0.5 * (half_y_left + half_y_right), 0.0),
            size=(0.0, half_y_left - half_y_right, 2.0 * half_z),
            fields=("Ey", "Ez", "Hy", "Hz"),
            frequencies=(frequency,),
            normal_direction="+",
        ),
        mw.FinitePlaneMonitor(
            "x_step_lower",
            position=(0.0, -0.5 * (half_y_left + half_y_right), 0.0),
            size=(0.0, half_y_left - half_y_right, 2.0 * half_z),
            fields=("Ey", "Ez", "Hy", "Hz"),
            frequencies=(frequency,),
            normal_direction="+",
        ),
        mw.FinitePlaneMonitor(
            "y_neg_left",
            position=(-0.5 * half_x, -half_y_left, 0.0),
            size=(half_x, 0.0, 2.0 * half_z),
            fields=("Ex", "Ez", "Hx", "Hz"),
            frequencies=(frequency,),
            normal_direction="-",
        ),
        mw.FinitePlaneMonitor(
            "y_pos_left",
            position=(-0.5 * half_x, half_y_left, 0.0),
            size=(half_x, 0.0, 2.0 * half_z),
            fields=("Ex", "Ez", "Hx", "Hz"),
            frequencies=(frequency,),
            normal_direction="+",
        ),
        mw.FinitePlaneMonitor(
            "y_neg_right",
            position=(0.5 * half_x, -half_y_right, 0.0),
            size=(half_x, 0.0, 2.0 * half_z),
            fields=("Ex", "Ez", "Hx", "Hz"),
            frequencies=(frequency,),
            normal_direction="-",
        ),
        mw.FinitePlaneMonitor(
            "y_pos_right",
            position=(0.5 * half_x, half_y_right, 0.0),
            size=(half_x, 0.0, 2.0 * half_z),
            fields=("Ex", "Ez", "Hx", "Hz"),
            frequencies=(frequency,),
            normal_direction="+",
        ),
        mw.FinitePlaneMonitor(
            "z_neg_left",
            position=(-0.5 * half_x, 0.0, -half_z),
            size=(half_x, 2.0 * half_y_left, 0.0),
            fields=("Ex", "Ey", "Hx", "Hy"),
            frequencies=(frequency,),
            normal_direction="-",
        ),
        mw.FinitePlaneMonitor(
            "z_neg_right",
            position=(0.5 * half_x, 0.0, -half_z),
            size=(half_x, 2.0 * half_y_right, 0.0),
            fields=("Ex", "Ey", "Hx", "Hy"),
            frequencies=(frequency,),
            normal_direction="-",
        ),
        mw.FinitePlaneMonitor(
            "z_pos_left",
            position=(-0.5 * half_x, 0.0, half_z),
            size=(half_x, 2.0 * half_y_left, 0.0),
            fields=("Ex", "Ey", "Hx", "Hy"),
            frequencies=(frequency,),
            normal_direction="+",
        ),
        mw.FinitePlaneMonitor(
            "z_pos_right",
            position=(0.5 * half_x, 0.0, half_z),
            size=(half_x, 2.0 * half_y_right, 0.0),
            fields=("Ex", "Ey", "Hx", "Hy"),
            frequencies=(frequency,),
            normal_direction="+",
        ),
    )
    return mw.ClosedSurfaceMonitor(name, faces, frequencies=(frequency,))


def _crop_indices(coords, lower: float, upper: float) -> np.ndarray:
    coord_array = _to_numpy(coords, float)
    return np.nonzero((coord_array >= lower - 1e-12) & (coord_array <= upper + 1e-12))[0]


def _manual_cropped_flux(monitor, tangential_bounds: dict[str, tuple[float, float]] | None = None) -> float:
    axis = monitor["axis"]
    tangential = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}[axis]
    u_name, v_name = tangential
    if tangential_bounds is None:
        u_indices = np.arange(_to_numpy(monitor[u_name], float).size)
        v_indices = np.arange(_to_numpy(monitor[v_name], float).size)
    else:
        u_indices = _crop_indices(_to_numpy(monitor[u_name], float), *tangential_bounds[u_name])
        v_indices = _crop_indices(_to_numpy(monitor[v_name], float), *tangential_bounds[v_name])

    u_coords = _to_numpy(monitor[u_name], float)[u_indices]
    v_coords = _to_numpy(monitor[v_name], float)[v_indices]
    # Cell-centered midpoint quadrature over each Yee sample's exact primal
    # control volume (boundary weight = full end diff, NOT the trapezoidal
    # d/2).  A closed Huygens box has no gaps between abutting faces, so the
    # control volumes must tile the whole box surface: the cell-centered rule
    # gives per-face weight sums equal to each face area and a total equal to
    # the box surface area to machine precision, matching both the Stratton-Chu
    # radiation quadrature and production `_compute_plane_flux`.  The former
    # trapezoidal rule truncated each face to the sampled span, dropping the
    # half-cell edge strips and under-integrating the near-field power flux,
    # which spuriously inflated the far-field/near-field power_ratio above unity.
    u_weights = np.empty_like(u_coords)
    v_weights = np.empty_like(v_coords)
    u_diffs = np.diff(u_coords)
    v_diffs = np.diff(v_coords)
    u_weights[0] = u_diffs[0]
    u_weights[-1] = u_diffs[-1]
    v_weights[0] = v_diffs[0]
    v_weights[-1] = v_diffs[-1]
    if u_coords.size > 2:
        u_weights[1:-1] = (u_diffs[:-1] + u_diffs[1:]) / 2.0
    if v_coords.size > 2:
        v_weights[1:-1] = (v_diffs[:-1] + v_diffs[1:]) / 2.0
    weights = u_weights[:, None] * v_weights[None, :]

    shape = (u_coords.size, v_coords.size)
    e_field = np.zeros(shape + (3,), dtype=np.complex128)
    h_field = np.zeros(shape + (3,), dtype=np.complex128)
    for component in ("Ex", "Ey", "Ez"):
        if component in monitor:
            e_field[..., "xyz".index(component[1].lower())] = _to_numpy(monitor[component], np.complex128)[
                np.ix_(u_indices, v_indices)
            ]
    for component in ("Hx", "Hy", "Hz"):
        if component in monitor:
            h_field[..., "xyz".index(component[1].lower())] = _to_numpy(monitor[component], np.complex128)[
                np.ix_(u_indices, v_indices)
            ]

    normal = 1.0 if monitor.get("normal_direction", "+") == "+" else -1.0
    poynting = 0.5 * np.real(np.cross(e_field, np.conj(h_field), axis=-1)[..., "xyz".index(axis)]) * normal
    return float(np.sum(poynting * weights))


def _save_face_field_plot(path: Path, monitor, component: str) -> None:
    tangential = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}[monitor["axis"]]
    u = _to_numpy(monitor[tangential[0]], float)
    v = _to_numpy(monitor[tangential[1]], float)
    data = np.abs(_to_numpy(monitor[component], np.complex128))
    extent = [float(u[0]), float(u[-1]), float(v[0]), float(v[-1])]

    fig, axis = plt.subplots(figsize=(5.0, 4.2))
    image = axis.imshow(data.T, origin="lower", extent=extent, cmap="inferno")
    axis.set_title(f"{monitor['face_label']} |{component}|")
    axis.set_xlabel(f"{tangential[0]} (m)")
    axis.set_ylabel(f"{tangential[1]} (m)")
    fig.colorbar(image, ax=axis, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_nearfield_plot(path: Path, target_monitor, reconstructed) -> None:
    x = _to_numpy(target_monitor["x"], float)
    y = _to_numpy(target_monitor["y"], float)
    target = np.abs(_to_numpy(target_monitor["Ez"], np.complex128))
    estimate = np.abs(_to_numpy(reconstructed["Ez"], np.complex128))
    error = np.abs(estimate - target)
    extent = [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, data, title in zip(
        axes,
        (target, estimate, error),
        ("Direct FDTD |Ez|", "Closed-Surface Reconstruction |Ez|", "Absolute Error"),
        strict=True,
    ):
        image = axis.imshow(data.T, origin="lower", extent=extent, cmap="inferno")
        axis.set_title(title)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        fig.colorbar(image, ax=axis, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_directivity_plot(path: Path, theta, computed, reference) -> None:
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(np.degrees(theta), reference, label="Hertzian Reference", linewidth=2.0)
    axis.plot(np.degrees(theta), computed, label="Closed-Surface NF2FF", linewidth=2.0)
    axis.set_xlabel("theta (deg)")
    axis.set_ylabel("Directivity")
    axis.set_xlim(0.0, 180.0)
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_rcs_plot(path: Path, theta, computed_phi0, ref_phi0, computed_phi90, ref_phi90) -> None:
    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.plot(np.degrees(theta), 10.0 * np.log10(np.clip(ref_phi0, 1e-30, None)), label="Rayleigh phi=0")
    axis.plot(
        np.degrees(theta),
        10.0 * np.log10(np.clip(computed_phi0, 1e-30, None)),
        label="Closed-Surface phi=0",
    )
    axis.plot(np.degrees(theta), 10.0 * np.log10(np.clip(ref_phi90, 1e-30, None)), label="Rayleigh phi=90")
    axis.plot(
        np.degrees(theta),
        10.0 * np.log10(np.clip(computed_phi90, 1e-30, None)),
        label="Closed-Surface phi=90",
    )
    axis.set_xlabel("theta (deg)")
    axis.set_ylabel("Bistatic RCS (dBsm)")
    axis.set_xlim(0.0, 180.0)
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_scattered_field_plot(path: Path, monitor, tangential_bounds: dict[str, tuple[float, float]] | None) -> None:
    if tangential_bounds is None:
        x = _to_numpy(monitor["x"], float)
        y = _to_numpy(monitor["y"], float)
        data = np.abs(_to_numpy(monitor["Ex"], np.complex128))
    else:
        x_indices = _crop_indices(monitor["x"], *tangential_bounds["x"])
        y_indices = _crop_indices(monitor["y"], *tangential_bounds["y"])
        x = _to_numpy(monitor["x"], float)[x_indices]
        y = _to_numpy(monitor["y"], float)[y_indices]
        data = np.abs(_to_numpy(monitor["Ex"], np.complex128))[np.ix_(x_indices, y_indices)]
    extent = [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]

    fig, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(data.T, origin="lower", extent=extent, cmap="inferno")
    axis.set_title("Scattered |Ex| on Closed Huygens Face")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    fig.colorbar(image, ax=axis, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


@pytest.fixture(scope="module")
def dipole_closed_surface_validation():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frequency = 1.0e9
    box_half = 0.08
    surface_monitor = _closed_surface_box_monitor("huygens", box_half, frequency)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.48, 0.48), (-0.48, 0.48), (-0.48, 0.48))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.pml(num_layers=8, strength=1.0),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            profile="ideal",
            source_time=mw.CW(frequency=frequency, amplitude=1.0),
            name="dipole",
        )
    )
    scene.add_monitor(surface_monitor)
    scene.add_monitor(
        mw.PlaneMonitor(
            "target",
            axis="z",
            position=0.20,
            fields=("Ex", "Ey", "Ez", "Hx", "Hy"),
        )
    )

    result = mw.Simulation.fdtd(
        scene,
        frequencies=[frequency],
        run_time=mw.TimeConfig.auto(steady_cycles=10, transient_cycles=25),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()

    currents = equivalent_surface_currents_from_monitor(result, surface_monitor.name)
    propagator = StrattonChuPropagator(currents, solver=result.solver, device="cuda")
    target_monitor = result.monitor("target")
    reconstructed = propagator.propagate_plane(
        axis="z",
        position=0.20,
        u=target_monitor["x"],
        v=target_monitor["y"],
        batch_size=2048,
    )

    x_indices = _crop_indices(target_monitor["x"], -0.12, 0.12)
    y_indices = _crop_indices(target_monitor["y"], -0.12, 0.12)
    near_field_errors = {}
    for component in ("Ex", "Ey", "Ez", "Hx", "Hy"):
        reference = _to_numpy(target_monitor[component], np.complex128)[np.ix_(x_indices, y_indices)]
        estimate = _to_numpy(reconstructed[component], np.complex128)[np.ix_(x_indices, y_indices)]
        near_field_errors[component] = float(np.linalg.norm(estimate - reference) / np.linalg.norm(reference))

    transformer = NearFieldFarFieldTransformer(currents, solver=result.solver, device="cuda")
    theta = torch.linspace(0.0, torch.pi, 181, device="cuda", dtype=torch.float64)
    phi = torch.linspace(0.0, 2.0 * torch.pi, 361, device="cuda", dtype=torch.float64)
    theta_grid, phi_grid = torch.broadcast_tensors(theta[:, None], phi[None, :])
    far_field = transformer.transform(theta_grid, phi_grid, radius=12.0, batch_size=4096)
    directivity = compute_directivity(far_field)

    theta_np = _to_numpy(theta, float)
    phi_np = _to_numpy(directivity["phi"][0], float)
    phi0_index = int(np.argmin(np.abs(phi_np)))
    computed_line = _to_numpy(torch.real(directivity["directivity"][:, phi0_index]), float)
    reference_line = 1.5 * np.sin(theta_np) ** 2
    line_rel = float(np.linalg.norm(computed_line - reference_line) / np.linalg.norm(reference_line))
    surface_payload = result.monitor(surface_monitor.name)
    power_flux = sum(_manual_cropped_flux(face_payload) for face_payload in surface_payload["faces"].values())
    power_ratio = float(directivity["P_rad"].item() / power_flux)

    _save_nearfield_plot(OUTPUT_DIR / "nearfield_closed_surface_dipole_ez_reconstruction.png", target_monitor, reconstructed)
    _save_directivity_plot(OUTPUT_DIR / "directivity_closed_surface_dipole_phi0.png", theta_np, computed_line, reference_line)

    return {
        "near_field_errors": near_field_errors,
        "D_max": float(directivity["D_max"].item()),
        "D_max_theta_deg": float(math.degrees(directivity["D_max_theta"])),
        "line_rel": line_rel,
        "power_ratio": power_ratio,
    }


@pytest.fixture(scope="module")
def tfsf_rayleigh_rcs_validation():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frequency = 1.0e9
    amplitude = 1.0
    sphere_radius = 0.01
    box_half = 0.06
    surface_monitor = _closed_surface_box_monitor("huygens", box_half, frequency)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.01),
        boundary=mw.BoundarySpec.pml(num_layers=8, strength=1.0),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            name="rayleigh_sphere",
            geometry=mw.Sphere(position=(0.0, 0.0, 0.0), radius=sphere_radius),
            material=mw.Material(eps_r=4.0),
        )
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization="Ex",
            source_time=mw.CW(frequency=frequency, amplitude=amplitude),
            injection=mw.TFSF(bounds=((-0.04, 0.04), (-0.04, 0.04), (-0.04, 0.04))),
            name="tfsf_pw",
        )
    )
    scene.add_monitor(surface_monitor)

    reference_scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.01),
        boundary=mw.BoundarySpec.pml(num_layers=8, strength=1.0),
        device="cuda",
    )
    reference_scene.add_source(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization="Ex",
            source_time=mw.CW(frequency=frequency, amplitude=amplitude),
            injection=mw.TFSF(bounds=((-0.04, 0.04), (-0.04, 0.04), (-0.04, 0.04))),
            name="tfsf_pw",
        )
    )
    reference_scene.add_monitor(
        mw.PointMonitor("incident_probe", (0.0, 0.0, 0.0), fields=("Ex",))
    )

    result = mw.Simulation.fdtd(
        scene,
        frequencies=[frequency],
        run_time=mw.TimeConfig.auto(steady_cycles=12, transient_cycles=30),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()

    reference_result = mw.Simulation.fdtd(
        reference_scene,
        frequencies=[frequency],
        run_time=mw.TimeConfig.auto(steady_cycles=12, transient_cycles=30),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()
    incident_amplitude = abs(reference_result.monitor("incident_probe")["data"].item())
    del reference_result

    currents = equivalent_surface_currents_from_monitor(result, surface_monitor.name)
    transformer = NearFieldFarFieldTransformer(currents, solver=result.solver, device="cuda")
    theta = torch.linspace(0.0, torch.pi, 121, device="cuda", dtype=torch.float64)
    phi = torch.linspace(0.0, 2.0 * torch.pi, 241, device="cuda", dtype=torch.float64)
    theta_grid, phi_grid = torch.broadcast_tensors(theta[:, None], phi[None, :])
    far_field = transformer.transform(theta_grid, phi_grid, radius=8.0, batch_size=4096)
    rcs = compute_bistatic_rcs(
        far_field,
        incident_amplitude=incident_amplitude,
        c=result.solver.c,
    )

    theta_np = _to_numpy(theta_grid, float)
    phi_np = _to_numpy(phi_grid, float)
    sigma = _to_numpy(rcs["rcs"], float)
    wave_number = 2.0 * math.pi * frequency / result.solver.c
    rayleigh_factor = abs((4.0 - 1.0) / (4.0 + 2.0))
    reference = (
        4.0
        * math.pi
        * (wave_number**4)
        * (sphere_radius**6)
        * (rayleigh_factor**2)
        * (1.0 - (np.sin(theta_np) * np.cos(phi_np)) ** 2)
    )
    phi90_index = int(np.argmin(np.abs(phi_np[0] - 0.5 * math.pi)))

    _save_rcs_plot(
        OUTPUT_DIR / "rcs_closed_surface_tfsf_rayleigh_phi_cuts.png",
        _to_numpy(theta, float),
        sigma[:, 0],
        reference[:, 0],
        sigma[:, phi90_index],
        reference[:, phi90_index],
    )
    _save_scattered_field_plot(
        OUTPUT_DIR / "tfsf_rayleigh_scattered_ex_xy.png",
        result.monitor(surface_monitor.name)["faces"]["z_pos"],
        None,
    )

    sigma_max = float(np.max(sigma))
    reference_max = float(np.max(reference))
    sigma_shape = sigma / sigma_max
    reference_shape = reference / reference_max
    return {
        "rel_l2": float(
            np.linalg.norm(sigma_shape - reference_shape) / np.linalg.norm(reference_shape)
        ),
        "phi0_rel": float(
            np.linalg.norm(sigma_shape[:, 0] - reference_shape[:, 0])
            / np.linalg.norm(reference_shape[:, 0])
        ),
        "phi90_rel": float(
            np.linalg.norm(sigma_shape[:, phi90_index] - reference_shape[:, phi90_index])
            / np.linalg.norm(reference_shape[:, phi90_index])
        ),
        "sigma_max_ratio": sigma_max / reference_max,
    }


@pytest.fixture(scope="module")
def dipole_nonrect_closed_surface_validation():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frequency = 1.0e9
    surface_monitor = _nonrect_closed_surface_monitor("huygens_poly", frequency)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.48, 0.48), (-0.48, 0.48), (-0.48, 0.48))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.pml(num_layers=8, strength=1.0),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            profile="ideal",
            source_time=mw.CW(frequency=frequency, amplitude=1.0),
            name="dipole",
        )
    )
    scene.add_monitor(surface_monitor)

    result = mw.Simulation.fdtd(
        scene,
        frequencies=[frequency],
        run_time=mw.TimeConfig.auto(steady_cycles=10, transient_cycles=25),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()

    currents = equivalent_surface_currents_from_monitor(result, surface_monitor.name)
    transformer = NearFieldFarFieldTransformer(currents, solver=result.solver, device="cuda")
    theta = torch.linspace(0.0, torch.pi, 181, device="cuda", dtype=torch.float64)
    phi = torch.linspace(0.0, 2.0 * torch.pi, 361, device="cuda", dtype=torch.float64)
    theta_grid, phi_grid = torch.broadcast_tensors(theta[:, None], phi[None, :])
    far_field = transformer.transform(theta_grid, phi_grid, radius=12.0, batch_size=4096)
    directivity = compute_directivity(far_field)

    theta_np = _to_numpy(theta, float)
    phi_np = _to_numpy(directivity["phi"][0], float)
    phi0_index = int(np.argmin(np.abs(phi_np)))
    computed_line = _to_numpy(torch.real(directivity["directivity"][:, phi0_index]), float)
    reference_line = 1.5 * np.sin(theta_np) ** 2
    line_rel = float(np.linalg.norm(computed_line - reference_line) / np.linalg.norm(reference_line))
    theta90_index = int(np.argmin(np.abs(np.degrees(theta_np) - 90.0)))
    theta90_ring = _to_numpy(torch.real(directivity["directivity"][theta90_index]), float)

    surface_payload = result.monitor(surface_monitor.name)
    _save_directivity_plot(
        OUTPUT_DIR / "directivity_nonrect_closed_surface_dipole_phi0.png",
        theta_np,
        computed_line,
        reference_line,
    )
    _save_face_field_plot(
        OUTPUT_DIR / "nonrect_closed_surface_step_face_ez.png",
        surface_payload["faces"]["x_step_upper"],
        "Ez",
    )

    return {
        "theta90_mean": float(np.mean(theta90_ring)),
        "theta90_spread": float(np.max(theta90_ring) / np.min(theta90_ring)),
        "D_max_theta_deg": float(math.degrees(directivity["D_max_theta"])),
        "line_rel": line_rel,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_closed_surface_dipole_near_field_reconstruction_is_physically_consistent(dipole_closed_surface_validation):
    errors = dipole_closed_surface_validation["near_field_errors"]
    assert errors["Ex"] < 0.25
    assert errors["Ey"] < 0.25
    assert errors["Ez"] < 0.25
    assert errors["Hx"] < 0.25
    assert errors["Hy"] < 0.25
    # Re-anchored around unity.  Old band 0.65-0.95 was measured against a
    # trapezoidal near-field flux reference that truncated each closed-box face
    # to its sampled span and dropped the half-cell edge strips, artificially
    # deflating the denominator.  With the flux reference on the same
    # cell-centered box quadrature as the Stratton-Chu radiation integral (the
    # control volumes tile the box surface to machine precision), the far-field
    # radiated power and the near-field flux agree to ~1.6% at 8 cells/box:
    # measured power_ratio = 1.016, box-size independent (0.02% drift between
    # box_half=0.08 and 0.12), which is the correct surface-equivalence result.
    assert 0.95 < dipole_closed_surface_validation["power_ratio"] < 1.10


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_closed_surface_dipole_directivity_matches_hertzian_reference(dipole_closed_surface_validation):
    assert abs(dipole_closed_surface_validation["D_max"] - 1.5) < 0.08
    assert abs(dipole_closed_surface_validation["D_max_theta_deg"] - 90.0) < 1.0
    assert dipole_closed_surface_validation["line_rel"] < 0.05


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_closed_surface_tfsf_rayleigh_rcs_matches_analytic_bistatic_pattern(tfsf_rayleigh_rcs_validation):
    assert tfsf_rayleigh_rcs_validation["rel_l2"] < 0.1
    assert tfsf_rayleigh_rcs_validation["phi0_rel"] < 0.1
    assert tfsf_rayleigh_rcs_validation["phi90_rel"] < 0.1
    # Re-anchored for edge-native (per-Yee-component) subpixel material sampling.
    # The radius is only one grid cell, so the ABSOLUTE Rayleigh cross section is
    # not converged even though the NORMALIZED bistatic pattern is (rel_l2 < 0.1
    # above -- the physics/NF2FF invariant that this test really guards).  The
    # absolute sigma_max_ratio is a discretization-dependent scalar set by how the
    # one-cell eps=4 sphere is rasterized: the previous hard node-staircase over-
    # represented the polarizability and gave ~2.30, while edge-native subpixel
    # sampling evaluates the occupancy-weighted eps at each Yee component's own
    # staggered location and gives ~0.396.  The re-anchor keeps the same box-
    # independence rigor that validated the old value: the new ratio is box-size
    # independent (0.3961, 0.3965, 0.3969 at box_half = 0.05, 0.06, 0.07 -> 0.2%
    # drift, scratch/rayleigh_box_independence.py), so the NF2FF far field is still
    # self-consistent -- only the marginally-resolved sphere's effective
    # polarizability moved with the sampling.  Band centered on the box-independent
    # edge-native value.  See the F4 subpixel acceptance ledger.
    assert 0.35 < tfsf_rayleigh_rcs_validation["sigma_max_ratio"] < 0.45


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_nonrect_closed_surface_dipole_directivity_matches_hertzian_reference(
    dipole_nonrect_closed_surface_validation,
):
    assert abs(dipole_nonrect_closed_surface_validation["D_max_theta_deg"] - 90.0) < 1.0
    assert abs(dipole_nonrect_closed_surface_validation["theta90_mean"] - 1.5) < 0.08
    assert dipole_nonrect_closed_surface_validation["theta90_spread"] < 1.25
    assert dipole_nonrect_closed_surface_validation["line_rel"] < 0.05
