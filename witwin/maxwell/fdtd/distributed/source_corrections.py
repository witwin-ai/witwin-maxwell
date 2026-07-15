from __future__ import annotations

import numpy as np

from ...fdtd_parallel import FDTDShardLayout
from .source_rebuild import rebuild_ideal_ex_interface_plane


def correct_ideal_point_ex_control_volume(solver, layout: FDTDShardLayout, global_scene) -> None:
    """Restore the global Ex control volume at an artificial shard-high edge.

    The local point-source compiler derives sample control widths from its local
    ``x_half`` array.  On a non-final shard the last owned Ex sample is an
    artificial endpoint, so that generic endpoint rule would give it half a
    control volume.  The Yee owner is still this shard; rescaling the compiled
    source patch by local/global width makes interface-adjacent ideal dipoles
    identical to global compilation on uniform and nonuniform grids.
    """

    if layout.owns_physical_face("x", "high"):
        return
    global_x_half = np.asarray(global_scene.x_half64, dtype=np.float64)
    local_index = int(layout.storage_cell_owned.stop) - 1
    global_index = int(layout.global_cell_owned.stop) - 1
    if local_index <= 0 or global_index <= 0 or global_index + 1 >= global_x_half.size:
        return

    compiled_sources = tuple(getattr(solver, "_compiled_sources", ()) or ())
    for term in tuple(getattr(solver, "_source_terms", ()) or ()):
        if term.get("field_name") != "Ex" or term.get("patch") is None:
            continue
        source_index = term.get("source_index")
        if source_index is None or not (0 <= int(source_index) < len(compiled_sources)):
            continue
        source = compiled_sources[int(source_index)]
        if source.get("kind") != "point_dipole" or source.get("profile") != "ideal":
            continue
        offset = int(term["offsets"][0])
        relative_index = local_index - offset
        if 0 <= relative_index < int(term["patch"].shape[0]):
            rebuild_ideal_ex_interface_plane(
                term,
                source,
                solver,
                local_index=local_index,
                global_index=global_index,
                global_x_half64=global_x_half,
            )


__all__ = ["correct_ideal_point_ex_control_volume"]
