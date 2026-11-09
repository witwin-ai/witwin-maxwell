"""Single-device FDTD coupling of annular lumped ports and resistors.

An annular object spans the two tangential E components of its plane. Each
component keeps the exact per-edge response, control volume and injection of
the shared rectangular lumped preparation; one scalar series branch then
couples the summed voltage row. The update order mirrors the series hot path
of :func:`~witwin.maxwell.fdtd.lumped.apply_lumped_runtime`, so a one-component
row reproduces it bit for bit.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .lumped import (
    LumpedRuntime,
    _optional_component,
    _prepared_scalar,
    _termination_components,
    prepare_lumped_runtime,
)


@dataclass(frozen=True)
class _ComponentRow:
    port_name: str
    voltage_component: str
    voltage_indices: torch.Tensor
    voltage_weights: torch.Tensor


@dataclass
class CoaxialLumpedRuntime:
    """Prepared scalar series branch coupled to a multi-component voltage row."""

    port_name: str
    topology: str
    field_names: tuple[str, ...]
    parts: tuple[LumpedRuntime, ...]
    field_dtype: torch.dtype
    dt: torch.Tensor
    resistance: torch.Tensor
    inductance: torch.Tensor
    capacitance: torch.Tensor
    inductance_active: bool
    capacitance_active: bool
    resistance_is_open_host: bool
    default_thevenin_voltage: torch.Tensor
    coupling_impedance: torch.Tensor
    discrete_port_impedance: torch.Tensor
    series_inductive_impedance: torch.Tensor
    series_capacitive_impedance_x2: torch.Tensor
    inverse_denominator: torch.Tensor
    part_voltage: torch.Tensor
    fast_state_scratch: torch.Tensor
    inductor_current: torch.Tensor
    capacitor_voltage: torch.Tensor
    last_voltage_before: torch.Tensor
    last_voltage_midpoint: torch.Tensor
    last_voltage_after: torch.Tensor
    last_branch_current: torch.Tensor
    electric_history: None = None

    @property
    def linear_indices(self) -> tuple[torch.Tensor, ...]:
        return tuple(part.linear_indices for part in self.parts)


def harmonic_coaxial_geometry(solver, geometry):
    """Replace the analytic radial row by the discrete-harmonic sheet of the live grid.

    The in-plane edges left unsuppressed by PEC between the two discrete
    conductors form a resistive network with edge conductance
    ``dual_width / primal_length``. Its potential is solved with 1 on the inner
    conductor and 0 on the outer one, and the normalized edge currents become the
    row. Kirchhoff's current law then holds exactly at every vacuum node, so the
    annular drive deposits no trapped node charge, and for any in-plane
    curl-free field the row equals the inner-to-outer conductor voltage.
    """
    from dataclasses import replace

    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    scene = solver.scene
    axes = "xyz"
    normal = axes.index(geometry.axis)
    u_name, v_name = (component[1] for component in geometry.voltage_components)
    u, v = axes.index(u_name), axes.index(v_name)
    k = int(geometry.plane_index)
    nodes = [np.asarray(getattr(scene, f"{axis}_nodes64"), dtype=np.float64) for axis in axes]
    primal = [np.asarray(getattr(scene, f"d{axis}_primal64"), dtype=np.float64) for axis in axes]
    dual = [np.asarray(getattr(scene, f"d{axis}_dual64"), dtype=np.float64) for axis in axes]
    meta = geometry.metadata
    inner, outer = float(meta["inner_radius"]), float(meta["outer_radius"])
    row_inner, row_outer = float(meta["row_inner_radius"]), float(meta["row_outer_radius"])
    nu, nv = nodes[u].size, nodes[v].size

    def plane(component):
        tensor = getattr(solver, f"c{component.lower()}_curl")
        index = [slice(None)] * 3
        index[normal] = k
        values = tensor[tuple(index)].detach().cpu().numpy() > 0.0
        return values if u < v else values.T

    free_u, free_v = plane(f"E{u_name}"), plane(f"E{v_name}")
    conductor = np.zeros((nu, nv), dtype=bool)
    conductor[:-1, :] |= ~free_u
    conductor[1:, :] |= ~free_u
    conductor[:, :-1] |= ~free_v
    conductor[:, 1:] |= ~free_v
    ez = getattr(solver, f"ce{geometry.axis}_curl")
    for layer in (k - 1, k):
        if 0 <= layer < ez.shape[normal]:
            index = [slice(None)] * 3
            index[normal] = layer
            values = ez[tuple(index)].detach().cpu().numpy() <= 0.0
            conductor |= values if u < v else values.T
    uu, vv = np.meshgrid(nodes[u] - scene_center(geometry, u), nodes[v] - scene_center(geometry, v), indexing="ij")
    radius = np.hypot(uu, vv)
    seeds = (~conductor) & (radius > inner) & (radius < outer)
    region = np.zeros_like(seeds)
    frontier = list(zip(*np.nonzero(seeds)))
    for item in frontier:
        region[item] = True
    while frontier:
        i, j = frontier.pop()
        for di, dj, free in ((1, 0, i + 1 < nu and free_u[i, j]), (-1, 0, i > 0 and free_u[i - 1, j]),
                             (0, 1, j + 1 < nv and free_v[i, j]), (0, -1, j > 0 and free_v[i, j - 1])):
            ni, nj = i + di, j + dj
            if not free or conductor[ni, nj] or region[ni, nj]:
                continue
            if not row_inner < radius[ni, nj] < row_outer:
                raise ValueError(
                    f"Coaxial lumped object {geometry.port_name!r} is not enclosed by PEC conductors "
                    "within two cells of its inner and outer radii."
                )
            region[ni, nj] = True
            frontier.append((ni, nj))
    middle = 0.5 * (inner + outer)
    edges = []
    for name, free, step, along, across in ((f"E{u_name}", free_u, (1, 0), u, v), (f"E{v_name}", free_v, (0, 1), v, u)):
        for i, j in zip(*np.nonzero(free)):
            ni, nj = i + step[0], j + step[1]
            if not (region[i, j] or region[ni, nj]):
                continue
            first = i if along == u else j
            second = j if along == u else i
            conductance = dual[across][second] / primal[along][first]
            edges.append((name, i, j, ni, nj, conductance, primal[along][first]))
    if not edges:
        raise ValueError(f"Coaxial lumped object {geometry.port_name!r} has no free sheet edge.")
    interior = {node: index for index, node in enumerate(zip(*np.nonzero(region)))}
    boundary_inner = conductor & (radius < middle)
    boundary_outer = conductor & (radius >= middle)
    rows, cols, vals = [], [], []
    rhs = np.zeros(len(interior))
    touched = set()
    for _name, i, j, ni, nj, conductance, _length in edges:
        for (a, b), (c, d) in (((i, j), (ni, nj)), ((ni, nj), (i, j))):
            if (a, b) not in interior:
                continue
            row = interior[(a, b)]
            rows.append(row); cols.append(row); vals.append(conductance)
            if (c, d) in interior:
                rows.append(row); cols.append(interior[(c, d)]); vals.append(-conductance)
            elif boundary_inner[c, d]:
                rhs[row] += conductance
                touched.add("inner")
            else:
                touched.add("outer")
    if touched != {"inner", "outer"}:
        raise ValueError(
            f"Coaxial lumped object {geometry.port_name!r} sheet does not connect the inner and outer conductors."
        )
    potential = np.asarray(spsolve(coo_matrix((vals, (rows, cols)), shape=(len(interior),) * 2).tocsc(), rhs))

    def node_potential(a, b):
        if (a, b) in interior:
            return float(potential[interior[(a, b)]])
        return 1.0 if boundary_inner[a, b] else 0.0

    currents = np.array([conductance * (node_potential(i, j) - node_potential(ni, nj))
                         for _name, i, j, ni, nj, conductance, _length in edges])
    total = sum(current for edge, current in zip(edges, currents)
                if boundary_inner[edge[1], edge[2]] and (edge[1], edge[2]) not in interior)
    total -= sum(current for edge, current in zip(edges, currents)
                 if boundary_inner[edge[3], edge[4]] and (edge[3], edge[4]) not in interior)
    components, indices, weights = [], [], []
    for name in geometry.voltage_components:
        selected = [(edge, current) for edge, current in zip(edges, currents) if edge[0] == name]
        entries = np.empty((len(selected), 3), dtype=np.int64)
        entries[:, normal] = k
        entries[:, u] = [edge[1] for edge, _current in selected]
        entries[:, v] = [edge[2] for edge, _current in selected]
        components.append(name)
        indices.append(torch.as_tensor(entries, device=solver.device, dtype=torch.int64))
        weights.append(torch.as_tensor([edge[6] * current / total for edge, current in selected],
                                       device=solver.device, dtype=torch.float64))
    metadata = dict(geometry.metadata)
    metadata.update(row="discrete_harmonic", sheet_resistance_ohm_per_square_equivalent=1.0 / total)
    return replace(geometry, voltage_components=tuple(components), voltage_indices=tuple(indices),
                   voltage_weights=tuple(weights), metadata=metadata)


def scene_center(geometry, axis):
    return float(geometry.metadata["center"][axis])


def prepare_coaxial_runtime(
    solver,
    geometry,
    *,
    resistance=None,
    termination=None,
    edge_control_volume,
    history_currents,
    validate_coefficient,
) -> CoaxialLumpedRuntime:
    """Prepare one annular branch on ``solver``'s live Yee coefficients.

    Edges whose electric update is suppressed by PEC carry no field and are
    removed from the row. ``edge_control_volume``, ``history_currents`` and
    ``validate_coefficient`` are the shared port-runtime helpers.
    """

    parts = []
    names = []
    for component, indices, weights in zip(
        geometry.voltage_components, geometry.voltage_indices, geometry.voltage_weights
    ):
        field = getattr(solver, component)
        curl = getattr(solver, f"c{component.lower()}_curl")
        local = indices.to(device=solver.device, dtype=torch.int64)
        shape = field.shape
        linear = local[:, 0] * shape[1] * shape[2] + local[:, 1] * shape[2] + local[:, 2]
        active = curl.reshape(-1).index_select(0, linear) > 0.0
        if not bool(torch.any(active)):
            continue
        row = _ComponentRow(
            geometry.port_name,
            component,
            local[active],
            weights.to(device=solver.device)[active],
        )
        if history_currents(solver, row):
            raise NotImplementedError(
                f"Coaxial lumped object {geometry.port_name!r} overlaps dispersive media; "
                "the annular branch does not forecast multi-component polarization currents."
            )
        part = prepare_lumped_runtime(
            row,
            dt=solver.dt,
            eps_edge=getattr(solver, f"eps_{component}"),
            electric_update_coefficient=curl,
            yee_control_volume=edge_control_volume(solver, component),
            resistance=torch.inf,
        )
        validate_coefficient(solver, part, component)
        parts.append(part)
        names.append(component)
    if not parts:
        raise ValueError(
            f"Coaxial lumped object {geometry.port_name!r} has no unsuppressed tangential "
            "edge between its conductors."
        )

    device = parts[0].linear_indices.device
    dtype = parts[0].field_dtype
    dt = _prepared_scalar(solver.dt, name="dt", device=device, dtype=dtype)
    topology, r_value, l_value, c_value = _termination_components(termination, resistance)
    if topology != "series":
        raise ValueError(f"Coaxial lumped object {geometry.port_name!r} supports series branches only.")
    r_tensor, _ = _optional_component(r_value, device=device, dtype=dtype)
    l_tensor, l_enabled = _optional_component(l_value, device=device, dtype=dtype)
    c_tensor, c_enabled = _optional_component(c_value, device=device, dtype=dtype)
    coupling = parts[0].coupling_impedance.clone()
    for part in parts[1:]:
        coupling = coupling + part.coupling_impedance
    discrete = 0.5 * coupling
    zero = torch.zeros((), device=device, dtype=dtype)
    one = torch.ones((), device=device, dtype=dtype)
    series_inductive = 2.0 * l_tensor / dt
    series_capacitive = torch.where(c_enabled, dt / (2.0 * torch.where(c_enabled, c_tensor, one)), zero)
    inverse = torch.reciprocal(r_tensor + series_inductive + series_capacitive + discrete)
    scalars = [torch.zeros((), device=device, dtype=dtype) for _ in range(9)]
    return CoaxialLumpedRuntime(
        port_name=str(geometry.port_name),
        topology="series",
        field_names=tuple(names),
        parts=tuple(parts),
        field_dtype=dtype,
        dt=dt,
        resistance=r_tensor,
        inductance=l_tensor,
        capacitance=c_tensor,
        inductance_active=bool(l_enabled),
        capacitance_active=bool(c_enabled),
        resistance_is_open_host=bool(torch.isinf(r_tensor)),
        default_thevenin_voltage=zero,
        coupling_impedance=coupling,
        discrete_port_impedance=discrete,
        series_inductive_impedance=series_inductive,
        series_capacitive_impedance_x2=2.0 * series_capacitive,
        inverse_denominator=inverse,
        part_voltage=scalars[0],
        fast_state_scratch=scalars[1],
        inductor_current=scalars[2],
        capacitor_voltage=scalars[3],
        last_voltage_before=scalars[4],
        last_voltage_midpoint=scalars[5],
        last_voltage_after=scalars[6],
        last_branch_current=scalars[7],
    )


def apply_coaxial_runtime(runtime: CoaxialLumpedRuntime, fields, *, thevenin_voltage=None) -> torch.Tensor:
    """Apply one implicit midpoint series correction across all row components in place."""

    drive = runtime.default_thevenin_voltage if thevenin_voltage is None else thevenin_voltage
    for index, (part, name) in enumerate(zip(runtime.parts, runtime.field_names)):
        flat = fields[name].view(-1)
        torch.index_select(flat, 0, part.linear_indices, out=part.edge_buffer)
        torch.mul(part.edge_buffer, part.voltage_weights, out=part.edge_buffer)
        if index == 0:
            torch.sum(part.edge_buffer, dim=0, out=runtime.last_voltage_before)
        else:
            torch.sum(part.edge_buffer, dim=0, out=runtime.part_voltage)
            runtime.last_voltage_before.add_(runtime.part_voltage)
    runtime.last_voltage_midpoint.copy_(runtime.last_voltage_after)
    runtime.last_voltage_midpoint.add_(runtime.last_voltage_before)
    runtime.last_voltage_midpoint.mul_(0.5)
    runtime.last_branch_current.copy_(runtime.last_voltage_midpoint)
    runtime.last_branch_current.sub_(drive)
    runtime.last_branch_current.sub_(runtime.capacitor_voltage)
    runtime.last_branch_current.addcmul_(runtime.series_inductive_impedance, runtime.inductor_current)
    runtime.last_branch_current.mul_(runtime.inverse_denominator)
    for part, name in zip(runtime.parts, runtime.field_names):
        torch.mul(part.injection, runtime.last_branch_current, out=part.correction_buffer)
        fields[name].view(-1).index_add_(0, part.linear_indices, part.correction_buffer, alpha=-1.0)
    runtime.last_voltage_midpoint.addcmul_(runtime.discrete_port_impedance, runtime.last_branch_current, value=-1.0)
    runtime.last_voltage_after.copy_(runtime.last_voltage_before)
    runtime.last_voltage_after.addcmul_(runtime.coupling_impedance, runtime.last_branch_current, value=-1.0)
    if not runtime.resistance_is_open_host:
        if runtime.inductance_active:
            torch.mul(runtime.last_branch_current, 2.0, out=runtime.fast_state_scratch)
            torch.sub(runtime.fast_state_scratch, runtime.inductor_current, out=runtime.inductor_current)
        if runtime.capacitance_active:
            torch.mul(runtime.series_capacitive_impedance_x2, runtime.last_branch_current, out=runtime.fast_state_scratch)
            runtime.capacitor_voltage.add_(runtime.fast_state_scratch)
    return runtime.last_branch_current


__all__ = ["CoaxialLumpedRuntime", "apply_coaxial_runtime", "harmonic_coaxial_geometry", "prepare_coaxial_runtime"]
