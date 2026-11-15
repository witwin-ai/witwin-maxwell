"""Body-aware near-field preconditioner pattern for closed conductor bodies (plan D2).

On a closed body mesh the top and bottom faces of a thin conductor carry
nearly cancelling currents, and their coupling beyond the assembly's
near radius (NEAR_RATIO cells) is strong compared with the far field of a
sheet. With the action's near set as preconditioner pattern, closed models
need several times the block GMRES steps of sheet models.

The preconditioner pattern therefore adds every face pair of the SAME body
(equal face owner: top, bottom and sidewall faces of one conductor box) whose
lateral centroid distance is below WIDE_RATIO times the larger face size,
with the predicate form of the assembly's pair_near. The key is the body
owner tensor of the closed body mesh, not the conductor_model name, so every
model built on that mesh uses it; polygon (planar-mesher) sheet layouts, like
rectangle sheet layouts, keep the near pattern. Only M^-1 changes: the action keeps the
exact near/far split, so admittances and currents stay those of the exact
system within the GMRES stopping bound. The added tiles come from the same
list kernel, bitwise the dense tiles.
"""

import torch

from .matrix_free import build_near_structure


# Closed F-A fine (18040 RWG), block steps at 0.1/1/10/100 GHz: near pattern 30/39/60/77;
# ratio 3: 16/22/30/36 at 1.26x LU fill; ratio 4: 15/21/29/34 at 1.52x. Gains saturate.
WIDE_RATIO = 3.0
ROW_CHUNK = 512


def body_pairs(prepared, structure):
    """Row-sorted (t, s) pairs: the near list plus same-body pairs within WIDE_RATIO cells, or None."""
    layout = prepared.layout
    # Closed body meshes carry face kinds (bottom/top/sidewall). Planar-mesher sheets also carry
    # face_owners, but a zero-thickness sheet has no cancelling face pair, and every wall face of a
    # via tube shares one lateral position: widening those would add O(wall faces^2) pairs.
    if getattr(layout, "face_kinds", None) is None:
        return None
    owners = layout.face_owners
    centres = layout.mesh.centroids[:, :2]
    edge2 = structure.geometry[4]
    faces = len(centres)
    near = structure.pairs
    keys = [near[:, 0] * faces + near[:, 1]]
    for start in range(0, faces, ROW_CHUNK):
        rows = torch.arange(start, min(faces, start + ROW_CHUNK), device=centres.device)
        lateral2 = (centres[rows, None] - centres[None]).square().sum(-1)
        size2 = torch.maximum(edge2[rows, None], edge2[None])
        body = (owners[rows, None] == owners[None]) & (owners[rows, None] >= 0)
        hit = (body & (lateral2 < WIDE_RATIO ** 2 * size2)).nonzero()
        keys.append(rows[hit[:, 0]] * faces + hit[:, 1])
    keys = torch.unique(torch.cat(keys), sorted=True)
    return torch.stack((keys // faces, keys % faces), 1).contiguous()


def body_preconditioner_structure(prepared, structure):
    """The widened preconditioner structure of a closed body mesh, or None for sheet layouts."""
    pairs = body_pairs(prepared, structure)
    if pairs is None:
        return None
    wide = build_near_structure(prepared, pairs, base=structure)
    wide.stats["wide_ratio"] = WIDE_RATIO
    return wide
