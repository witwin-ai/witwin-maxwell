import numpy as np


ZINC = {
    "950": "#09090b",
    "900": "#18181b",
    "800": "#27272a",
    "700": "#3f3f46",
    "600": "#52525b",
    "400": "#a1a1aa",
    "200": "#e4e4e7",
    "100": "#f4f4f5",
}


def build_fdtd_pyvista_grid(fdtd, freq_solution):
    import pyvista as pv

    field_mag, _, _, _ = fdtd._interpolate_yee_to_center(freq_solution)
    permittivity = fdtd._get_centered_permittivity().cpu().numpy()
    x0, _, y0, _, z0, _ = fdtd.scene.domain_range

    image = pv.ImageData()
    image.dimensions = (
        field_mag.shape[0] + 1,
        field_mag.shape[1] + 1,
        field_mag.shape[2] + 1,
    )
    image.origin = (x0, y0, z0)
    image.spacing = (fdtd.dx, fdtd.dy, fdtd.dz)
    image.cell_data["field_mag"] = field_mag.ravel(order="F")
    image.cell_data["permittivity"] = permittivity.ravel(order="F")
    return image.cell_data_to_point_data(pass_cell_data=True)


def _iso_value_from_frac(field_max, iso_frac):
    return max(float(field_max) * float(iso_frac), 1e-12)


def _build_view_state(grid, iso_frac):
    field_mag = np.asarray(grid["field_mag"])
    permittivity = np.asarray(grid["permittivity"])
    field_min = float(np.nanmin(field_mag))
    field_max = float(np.nanmax(field_mag))
    eps_min = float(np.nanmin(permittivity))
    eps_max = float(np.nanmax(permittivity))
    iso_frac = float(np.clip(iso_frac, 0.05, 0.95))
    dielectric_threshold = float(np.clip(1.05, eps_min, eps_max))
    return {
        "field_min": field_min,
        "field_max": field_max,
        "eps_min": eps_min,
        "eps_max": eps_max,
        "mode": "field",
        "iso_frac": iso_frac,
        "iso_value": _iso_value_from_frac(field_max, iso_frac),
        "dielectric_threshold": dielectric_threshold,
        "volume_opacity": 0.18,
        "show_outline": True,
        "show_slices": True,
        "show_contour": True,
    }


def _safe_remove_actor(plotter, actor):
    if actor is None:
        return
    if isinstance(actor, (list, tuple)):
        for item in actor:
            _safe_remove_actor(plotter, item)
        return
    try:
        plotter.remove_actor(actor, render=False)
    except (RuntimeError, TypeError, ValueError):
        return


def _clear_scene_actors(plotter, actors):
    for key, actor in list(actors.items()):
        _safe_remove_actor(plotter, actor)
        actors[key] = None
    if hasattr(plotter, "clear_plane_widgets"):
        plotter.clear_plane_widgets()


def _format_status_panel(state):
    mode_label = "Field" if state["mode"] == "field" else "Dielectric"
    return "\n".join(
        [
            "Status",
            f"Mode: {mode_label}",
            f"Field max: {state['field_max']:.3e}",
            f"Eps range: {state['eps_min']:.3f} .. {state['eps_max']:.3f}",
            f"Iso / cutoff: {state['iso_frac']:.2f} / {state['dielectric_threshold']:.3f}",
        ]
    )


def _update_status_panel(plotter, state):
    plotter.subplot(0, 1)
    plotter.add_text(
        _format_status_panel(state),
        position=(0.08, 0.015),
        viewport=True,
        font_size=8,
        color=ZINC["100"],
        name="status_panel",
        render=False,
    )


def _update_scene_title(plotter, state):
    mode_label = "Field" if state["mode"] == "field" else "Dielectric"
    plotter.subplot(0, 0)
    plotter.add_text(
        f"Scene View: {mode_label}",
        position="upper_left",
        font_size=12,
        color=ZINC["100"],
        name="scene_title",
        render=False,
    )


def _add_outline(plotter, grid):
    plotter.subplot(0, 0)
    return plotter.add_mesh(
        grid.outline(),
        color=ZINC["200"],
        line_width=1,
        name="outline",
        render=False,
    )


def _add_field_volume(plotter, grid, state):
    plotter.subplot(0, 0)
    return plotter.add_volume(
        grid,
        scalars="field_mag",
        cmap="inferno",
        opacity=state["volume_opacity"],
        shade=False,
        mapper="smart",
        show_scalar_bar=False,
        clim=(state["field_min"], state["field_max"]),
        name="field_volume",
        render=False,
    )


def _add_field_contour(plotter, grid, state):
    plotter.subplot(0, 0)
    contour = grid.contour(isosurfaces=[state["iso_value"]], scalars="field_mag")
    if contour.n_cells == 0:
        return None
    return plotter.add_mesh(
        contour,
        scalars="field_mag",
        cmap="inferno",
        opacity=0.9,
        smooth_shading=True,
        show_scalar_bar=False,
        clim=(state["field_min"], state["field_max"]),
        specular=0.2,
        name="field_contour",
        render=False,
    )


def _add_field_slices(plotter, grid, state):
    plotter.subplot(0, 0)
    return plotter.add_mesh_slice_orthogonal(
        grid,
        scalars="field_mag",
        cmap="inferno",
        line_width=2,
        widget_color=ZINC["400"],
        show_scalar_bar=False,
        clim=(state["field_min"], state["field_max"]),
        render=False,
    )


def _add_dielectric_mesh(plotter, grid, state):
    plotter.subplot(0, 0)
    dielectric = grid.threshold(
        value=state["dielectric_threshold"],
        scalars="permittivity",
    )
    if dielectric.n_cells == 0:
        return None
    return plotter.add_mesh(
        dielectric,
        scalars="permittivity",
        cmap="viridis",
        opacity=0.28,
        smooth_shading=True,
        show_scalar_bar=False,
        clim=(state["eps_min"], state["eps_max"]),
        specular=0.12,
        name="dielectric_mesh",
        render=False,
    )


def _add_permittivity_slices(plotter, grid, state):
    plotter.subplot(0, 0)
    return plotter.add_mesh_slice_orthogonal(
        grid,
        scalars="permittivity",
        cmap="viridis",
        line_width=2,
        widget_color=ZINC["400"],
        show_scalar_bar=False,
        clim=(state["eps_min"], state["eps_max"]),
        render=False,
    )


def _setup_scene_view(plotter, state):
    plotter.subplot(0, 0)
    plotter.set_background(ZINC["900"], top=ZINC["800"], all_renderers=False)
    plotter.add_axes(line_width=2, color=ZINC["200"])
    plotter.show_grid(color=ZINC["600"])
    plotter.camera_position = "iso"
    _update_scene_title(plotter, state)


def _setup_control_panel(plotter, state):
    plotter.subplot(0, 1)
    plotter.set_background(ZINC["800"], top=ZINC["700"], all_renderers=False)
    plotter.add_text(
        "Controls",
        position=(0.08, 0.94),
        viewport=True,
        font_size=16,
        color=ZINC["100"],
        name="panel_title",
        render=False,
    )
    plotter.add_text(
        "Switch the scene between Field and Dielectric.",
        position=(0.08, 0.89),
        viewport=True,
        font_size=10,
        color=ZINC["200"],
        name="panel_subtitle",
        render=False,
    )
    plotter.add_text(
        "Display Mode",
        position=(0.08, 0.82),
        viewport=True,
        font_size=11,
        color=ZINC["100"],
        name="panel_mode_label",
        render=False,
    )
    plotter.add_text(
        "Scene Toggles",
        position=(0.08, 0.66),
        viewport=True,
        font_size=11,
        color=ZINC["100"],
        name="panel_toggle_label",
        render=False,
    )
    plotter.add_text(
        "Parameters",
        position=(0.08, 0.49),
        viewport=True,
        font_size=11,
        color=ZINC["100"],
        name="panel_param_label",
        render=False,
    )
    _update_status_panel(plotter, state)


def _rebuild_scene(plotter, grid, state, actors):
    plotter.subplot(0, 0)
    _clear_scene_actors(plotter, actors)
    _update_scene_title(plotter, state)

    if state["show_outline"]:
        actors["outline"] = _add_outline(plotter, grid)

    if state["mode"] == "field":
        actors["field_volume"] = _add_field_volume(plotter, grid, state)
        actors["field_contour"] = (
            _add_field_contour(plotter, grid, state) if state["show_contour"] else None
        )
        actors["field_slices"] = (
            _add_field_slices(plotter, grid, state) if state["show_slices"] else None
        )
        return

    actors["dielectric_mesh"] = _add_dielectric_mesh(plotter, grid, state)
    actors["dielectric_slices"] = (
        _add_permittivity_slices(plotter, grid, state) if state["show_slices"] else None
    )


def _add_labeled_checkbox(plotter, callback, value, label, button_position, text_position):
    plotter.add_checkbox_button_widget(
        callback,
        value=value,
        position=button_position,
        size=26,
        border_size=2,
        color_on="#f97316",
        color_off=ZINC["600"],
        background_color=ZINC["100"],
    )
    plotter.add_text(
        label,
        position=text_position,
        viewport=True,
        font_size=10,
        color=ZINC["100"],
        render=False,
    )


def _add_control_widgets(plotter, grid, state, actors, window_size):
    plotter.subplot(0, 1)
    checkbox_x = 24
    y0 = int(window_size[1] * 0.56)
    gap = int(window_size[1] * 0.055)

    def refresh_and_render():
        _rebuild_scene(plotter, grid, state, actors)
        _update_status_panel(plotter, state)
        plotter.render()

    def update_mode(value):
        state["mode"] = "field" if str(value).lower().startswith("field") else "dielectric"
        refresh_and_render()

    def toggle_slices(value):
        state["show_slices"] = bool(value)
        refresh_and_render()

    def toggle_outline(value):
        state["show_outline"] = bool(value)
        refresh_and_render()

    def toggle_contour(value):
        state["show_contour"] = bool(value)
        refresh_and_render()

    def update_volume_opacity(value):
        state["volume_opacity"] = float(value)
        refresh_and_render()

    def update_iso_frac(value):
        state["iso_frac"] = float(np.clip(value, 0.05, 0.95))
        state["iso_value"] = _iso_value_from_frac(state["field_max"], state["iso_frac"])
        refresh_and_render()

    def update_dielectric_threshold(value):
        state["dielectric_threshold"] = float(
            np.clip(value, state["eps_min"], state["eps_max"])
        )
        refresh_and_render()

    plotter.add_text_slider_widget(
        update_mode,
        data=["Field", "Dielectric"],
        value=0,
        pointa=(0.12, 0.77),
        pointb=(0.88, 0.77),
        color=ZINC["100"],
        interaction_event="end",
    )

    _add_labeled_checkbox(
        plotter,
        toggle_slices,
        state["show_slices"],
        "Show slices",
        (checkbox_x, y0),
        (0.20, 0.57),
    )
    _add_labeled_checkbox(
        plotter,
        toggle_outline,
        state["show_outline"],
        "Show outline",
        (checkbox_x, y0 - gap),
        (0.20, 0.515),
    )
    _add_labeled_checkbox(
        plotter,
        toggle_contour,
        state["show_contour"],
        "Field contour",
        (checkbox_x, y0 - 2 * gap),
        (0.20, 0.46),
    )

    plotter.add_slider_widget(
        update_volume_opacity,
        rng=(0.03, 0.45),
        value=state["volume_opacity"],
        title="Field volume opacity",
        pointa=(0.10, 0.42),
        pointb=(0.90, 0.42),
        fmt="%.2f",
        interaction_event="end",
        color=ZINC["100"],
    )
    plotter.add_slider_widget(
        update_iso_frac,
        rng=(0.05, 0.95),
        value=state["iso_frac"],
        title="Field isosurface fraction",
        pointa=(0.10, 0.34),
        pointb=(0.90, 0.34),
        fmt="%.2f",
        interaction_event="end",
        color=ZINC["100"],
    )
    plotter.add_slider_widget(
        update_dielectric_threshold,
        rng=(state["eps_min"], max(state["eps_max"], state["eps_min"] + 1e-6)),
        value=state["dielectric_threshold"],
        title="Dielectric cutoff",
        pointa=(0.10, 0.18),
        pointb=(0.90, 0.18),
        fmt="%.3f",
        interaction_event="end",
        color=ZINC["100"],
    )


def show_pyvista_solution(grid, iso_frac=0.35, headless=False, screenshot=None):
    import pyvista as pv

    pv.global_theme.allow_empty_mesh = True

    off_screen = headless or screenshot is not None
    state = _build_view_state(grid, iso_frac)
    window_size = (1920, 980)
    plotter = pv.Plotter(
        shape=(1, 2),
        col_weights=[4, 2],
        border=True,
        border_color=ZINC["600"],
        window_size=window_size,
        off_screen=off_screen,
    )
    if hasattr(plotter, "enable_depth_peeling"):
        plotter.enable_depth_peeling()

    actors = {
        "outline": None,
        "field_volume": None,
        "field_contour": None,
        "field_slices": None,
        "dielectric_mesh": None,
        "dielectric_slices": None,
    }

    _setup_scene_view(plotter, state)
    _setup_control_panel(plotter, state)
    _rebuild_scene(plotter, grid, state, actors)
    plotter.subplot(0, 0)
    if hasattr(plotter, "add_camera_orientation_widget"):
        plotter.add_camera_orientation_widget()
    _add_control_widgets(plotter, grid, state, actors, window_size)

    if screenshot is not None:
        plotter.show(screenshot=str(screenshot), auto_close=True)
        return

    if headless:
        plotter.show(auto_close=True)
        return

    print("PyVista controls: left-drag rotate, right-drag zoom, middle-drag pan.")
    print("Use the zinc control panel on the right to switch between Field and Dielectric.")
    plotter.show()
