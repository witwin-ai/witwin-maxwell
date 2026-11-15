import numpy as np
import torch

from ..visualization import (
    extract_orthogonal_slice,
    plot_cross_section_panels,
)
from .postprocess import (
    get_centered_permittivity,
    get_frequency_solution,
    interpolate_yee_to_center,
)
from .coords import component_coords


def _to_numpy_field(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def plot_cross_section(
    solver,
    axis="z",
    position=0.0,
    component="abs",
    field_log_scale=False,
    figsize=(12, 5),
    save_path=None,
    verbose=True,
):
    freq_solution = get_frequency_solution(solver)

    ex = _to_numpy_field(freq_solution["Ex"])
    ey = _to_numpy_field(freq_solution["Ey"])
    ez = _to_numpy_field(freq_solution["Ez"])
    component_str = component.lower()

    if component_str == "abs":
        field_data, x_coords, y_coords, z_coords = interpolate_yee_to_center(solver, freq_solution)
        title_prefix = "|E|"
        eps_np = get_centered_permittivity(solver).cpu().numpy()
    elif component_str == "ex":
        field_data = np.abs(ex)
        title_prefix = "|Ex|"
        x_coords, y_coords, z_coords = component_coords(solver.scene, "Ex", device=solver.scene.device)
        eps_np = (solver.eps_Ex / solver.eps0).cpu().numpy()
    elif component_str == "ey":
        field_data = np.abs(ey)
        title_prefix = "|Ey|"
        x_coords, y_coords, z_coords = component_coords(solver.scene, "Ey", device=solver.scene.device)
        eps_np = (solver.eps_Ey / solver.eps0).cpu().numpy()
    elif component_str == "ez":
        field_data = np.abs(ez)
        title_prefix = "|Ez|"
        x_coords, y_coords, z_coords = component_coords(solver.scene, "Ez", device=solver.scene.device)
        eps_np = (solver.eps_Ez / solver.eps0).cpu().numpy()
    else:
        field_data, x_coords, y_coords, z_coords = interpolate_yee_to_center(solver, freq_solution)
        title_prefix = "|E|"
        eps_np = get_centered_permittivity(solver).cpu().numpy()

    field_info = extract_orthogonal_slice(field_data, axis, position, x_coords, y_coords, z_coords)
    eps_info = extract_orthogonal_slice(eps_np, axis, position, x_coords, y_coords, z_coords)

    field_slice = field_info["slice"]
    if field_log_scale:
        field_slice = np.where(field_slice > 1e-12, field_slice, 1e-12)
        field_max = np.max(field_slice)
        if field_max > 0:
            field_slice = 20 * np.log10(field_slice / field_max)
        cbar_label = f"{title_prefix} (dB)"
    else:
        cbar_label = f"{title_prefix} (V/m)"

    plot_cross_section_panels(
        field_slice=field_slice,
        permittivity_slice=eps_info["slice"],
        x_coords=field_info["x_coords"],
        y_coords=field_info["y_coords"],
        xlabel=field_info["xlabel"],
        ylabel=field_info["ylabel"],
        field_title=f"{title_prefix} at {axis}={position:.3f}m",
        field_cbar_label=cbar_label,
        permittivity_title=f"Permittivity at {axis}={position:.3f}m",
        figsize=figsize,
        save_path=save_path,
        verbose=verbose,
        contour_levels=[1.5],
        contour_color="cyan",
    )
