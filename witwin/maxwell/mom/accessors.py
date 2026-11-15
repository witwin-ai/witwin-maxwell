"""``mw.mom`` functions that read the MoM part of a ``Result``.

Each function takes the ``Result`` of ``Simulation.mom(...).run()`` (or a loaded one) as its
first argument and returns exactly what the former ``Result`` member returned.
``Result.query_fields`` stays on ``Result``; layered networks stay ``Result.network``.
"""
from __future__ import annotations

from .._stability import experimental


def _surface_data(result):
    if result._surface_data is None:
        raise ValueError("This Result has no native surface currents")
    return result._surface_data


def _harmonic_data(result, what):
    if result._harmonic_data is None:
        raise ValueError(f"This Result has no {what}")
    return result._harmonic_data


@experimental("mom")
def surface_currents(result):
    """Typed scattered RWG coefficients plus the exact incident carrier."""
    return _surface_data(result)


@experimental("mom")
def harmonic_state(result):
    """Coupled DC/AC surface and volume unknowns with their actual frequency axis."""
    data = _harmonic_data(result, "coupled harmonic state")
    data.state.assert_current()
    return data.state


@experimental("mom")
def refine_harmonic_power(result, control_surface, *, surface_orders=(2, 4, 8), volume_orders=(2, 4, 8),
                          relative_tolerance=1e-4, absolute_tolerance=1e-20, query_config=None,
                          query_batch_size=None, progress_callback=None):
    """Refine power quadrature; optional callbacks receive detached CPU step reports."""
    from .harmonic_observables import harmonic_power_refinement
    return harmonic_power_refinement(_harmonic_data(result, "coupled harmonic power balance"), control_surface,
                                     surface_orders=surface_orders, volume_orders=volume_orders,
                                     relative_tolerance=relative_tolerance, absolute_tolerance=absolute_tolerance,
                                     query_config=query_config, query_batch_size=query_batch_size,
                                     progress_callback=progress_callback)


@experimental("mom")
def harmonic_power_balance(result, control_surface, *, surface_order=3, volume_order=3, query_config=None,
                           query_batch_size=None):
    """Integrate exterior flux and regional work for a coupled harmonic Result."""
    from .harmonic_observables import harmonic_power_balance as balance
    return balance(_harmonic_data(result, "coupled harmonic power balance"), control_surface,
                   surface_order=surface_order, volume_order=volume_order, query_config=query_config,
                   query_batch_size=query_batch_size)


@experimental("mom")
def query_dc(result, points, *, region_id):
    """Return the DC scalar potential part [V] and full real electric field [V/m]."""
    return _harmonic_data(result, "harmonic DC field").query_dc(points, region_id)


@experimental("mom")
def query_dc_magnetic(result, points, *, region_id):
    """Return recovered static H [A/m] and B [T] at physical points."""
    return _harmonic_data(result, "harmonic DC magnetic field").query_dc_magnetic(points, region_id)


@experimental("mom")
def query_dc_dual(result, points, *, region_id):
    """Return coupled DC E/H, scalar potential parts and magnetic flux data.

    Conductive interiors report the residue lim(omega*B)=i*sigma_m*H
    in T/s and no finite B value. Vacuum exterior B remains finite.
    """
    return _harmonic_data(result, "coupled harmonic DC state").query_dc_dual(points, region_id)


@experimental("mom")
def far_field(result, directions, *, harmonic=None):
    """MoM outgoing amplitudes ``(F_E, F_H)`` with E,H ~ exp(i k r) F / r, as ``FarFieldData``.

    ``directions`` are ``[D, 3]`` unit vectors. Each result has dims
    ``("f", "direction", "component")`` (``f`` of length one: the solve
    frequency, or ``n*f0`` for ``harmonic=n``), units ``V`` (E) and ``A`` (H);
    the values are views of the computed amplitudes.
    """
    from .results import far_field_data
    if result._harmonic_data is not None:
        electric, magnetic = result._harmonic_data.far_field(directions, harmonic=harmonic)
        return far_field_data(electric, magnetic, directions, result.frequencies[harmonic])
    if harmonic is not None:
        raise ValueError("harmonic selects far fields only on a coupled harmonic Result")
    electric, magnetic = _surface_data(result).far_field(directions)
    return far_field_data(electric, magnetic, directions, result.frequencies[0])


@experimental("mom")
def rayleigh_orders(result, orders, *, side, plane_offset, quadrature_order=6):
    """2D periodic total plane amplitudes and signed outward order flux."""
    return _surface_data(result).rayleigh_orders(orders, side=side, plane_offset=plane_offset,
                                                 quadrature_order=quadrature_order)


@experimental("mom")
def cylindrical_orders(result, orders, directions, *, quadrature_order=6):
    """1D periodic scattered cylindrical amplitudes and power per radian."""
    return _surface_data(result).cylindrical_orders(orders, directions, quadrature_order=quadrature_order)


@experimental("mom")
def bloch_spectrum(result, orders, *, quadrature_order=6):
    """3D homogeneous-exterior equivalent-current potential coefficients.

    These coefficients are not body-interior physical fields or incident
    fields. Spectral poles reject ordinary forced resolvent evaluation.
    """
    return _surface_data(result).bloch_spectrum(orders, quadrature_order=quadrature_order)
