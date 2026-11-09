"""A single, scoped compiler dispatch; ineligible scenes retain torch/autograd."""
from contextlib import contextmanager
from contextvars import ContextVar

import torch

from ...media import Material
from ...monitors import PointMonitor
from .geometry import MAX_TREE_NODES, qualified_geometry, tree_size


setup_enabled = ContextVar("maxwell_exact_setup_enabled", default=False)
_requires_grad_memo = ContextVar("maxwell_prepare_requires_grad_memo", default=None)
UNCHANGED = object()
# Frozen declarations whose constructors coerce every field to str/float cannot
# hold a tensor; exact-type matches skip their recursive visit (16k-point scenes).
_TENSOR_FREE = (PointMonitor,)


@contextmanager
def requires_grad_scope(*, reuse=False):
    """Keep scene admission results local to one preparation, including failures."""
    if reuse and _requires_grad_memo.get() is not None:
        yield
        return
    token = _requires_grad_memo.set({})
    try:
        yield
    finally:
        _requires_grad_memo.reset(token)


def _same_children(value, original):
    """Recognize the shallow declaration copies made by PreparedScene."""
    if value is original:
        return True
    if type(value) is not type(original):
        return False
    if isinstance(value, (list, tuple)):
        return len(value) == len(original) and all(a is b for a, b in zip(value, original))
    if isinstance(value, dict):
        return value.keys() == original.keys() and all(value[k] is original[k] for k in value)
    return False


def requires_grad(value):
    memo = _requires_grad_memo.get()
    if memo is None:
        return _walk_requires_grad(value)
    # Cache only scene roots: small temporary compiler arguments may reuse ids.
    from ...scene import PreparedScene, Scene

    if not isinstance(value, Scene):
        return _walk_requires_grad(value)
    if isinstance(value, PreparedScene):
        original = value._public_scene
        declarations = vars(original)
        # The wrapper keeps the public scene and shallow copies of declarations.
        # Check new/replaced state too, without recursively revisiting monitors.
        return requires_grad(original) or _walk_requires_grad({
            name: child for name, child in vars(value).items()
            if name != "_public_scene" and (
                name not in declarations or not _same_children(child, declarations[name]))
        })
    # The entry holds the scene itself: a scene dropped inside the scope cannot
    # hand its id to a new scene and return a stale answer.
    entry = memo.get(id(value))
    if entry is None or entry[0] is not value:
        entry = memo[id(value)] = (value, _walk_requires_grad(value))
    return entry[1]


def _walk_requires_grad(value, seen=None):
    if torch.is_tensor(value):
        return value.requires_grad
    if type(value) in _TENSOR_FREE:
        return False
    if seen is None:
        seen = set()
    if id(value) in seen:
        return False
    seen.add(id(value))
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (list, tuple)):
        children = value
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        children = vars(value).values()
    else:
        return False
    return any(_walk_requires_grad(child, seen) for child in children)


def eligible(scene, eps_background, mu_background, subpixel, *, coefficients=False):
    from .. import materials as m

    if not setup_enabled.get() or torch.device(scene.device).type != "cuda":
        return False
    if scene.coordinates is not None or m._material_dtype(scene) != torch.float32:
        return False
    if any(getattr(scene, axis).ndim != 1 for axis in "xyz") or scene.X.ndim != 3:
        return False
    if (getattr(scene, "_material_model_override", None) is not None
            or getattr(scene, "_conductivity_capture", None) is not None):
        return False
    # Cropped material-cache scenes retain the full component cardinalities.
    if (scene.Nx_ex, scene.Ny_ey, scene.Nz_ez) != (scene.Nx-1, scene.Ny-1, scene.Nz-1):
        return False
    if requires_grad(scene) or requires_grad((eps_background, mu_background, subpixel)):
        return False
    if not coefficients and scene.material_regions:
        return False
    samples, averaging, _ = m._resolve_subpixel(subpixel)
    if not coefficients and (samples, averaging) not in (((4, 4, 4), "kottke_edge"),
                                    ((4, 4, 4), "kottke_smooth"),
                                    ((3, 3, 3), "polarized"),
                                    ((4, 4, 4), "polarized"),
                                    ((1, 1, 1), "arithmetic")):
        return False
    if not all(isinstance(v, (int, float)) for v in (eps_background, mu_background)):
        return False
    bulk = {id(s) for s in m._bulk_structures(scene)}
    for structure in scene.structures:
        material = structure.material
        if getattr(material, "modulation", None) is not None:
            return False
        # Surfaces and conductor masks are compiled by their existing helpers;
        # their geometry/material declarations do not enter bulk sample blends.
        if coefficients and getattr(material, "is_dispersive", False):
            return False
        if id(structure) not in bulk:
            continue
        if type(material) is not Material:
            return False
        if not coefficients and (not qualified_geometry(structure.geometry)
                or tree_size(structure.geometry) > MAX_TREE_NODES):
            return False
        if (material.epsilon_tensor is not None or material.mu_tensor is not None
                or material.sigma_e_tensor is not None or material.nonlinearity
                or material.kerr_chi3 or material.breakdown is not None):
            return False
        if any(torch.is_tensor(v) for v in vars(material).values()):
            return False
        if not coefficients and not _setup_geometry_precision(structure.geometry):
            return False
    return True


def _fp32_geometry(value):
    """Strict scope for pole ownership and conductor kernels."""
    return _geometry_precision(value, (torch.float32,))


def _setup_geometry_precision(value):
    # Primitive SDFs cast dimensions to the query dtype; indexed transform
    # scalars do not promote fp32 queries. Mesh declarations stay fp32-only.
    return _geometry_precision(value, (torch.float32, torch.float64))


def _geometry_precision(value, dtypes):
    from witwin.core import Mesh

    if type(value) is Mesh:
        return all(_geometry_precision(v, (torch.float32,)) for k, v in vars(value).items()
                   if k not in ("_faces_tensor", "_sdf_cache", "_voxel_cache"))
    if torch.is_tensor(value):
        return value.dtype in dtypes
    if isinstance(value, (tuple, list)):
        return all(_geometry_precision(v, dtypes) for v in value)
    if hasattr(value, "__dict__"):
        # Boolean nodes mirror child tensors, including integer mesh topology.
        # Inspect the actual children once through geometries instead.
        return all(_geometry_precision(v, dtypes) for k, v in vars(value).items()
                   if not k.startswith("_child_tensor_"))
    return True


def try_compile_material_model(scene, eps_background, mu_background, subpixel, include_edges):
    if not eligible(scene, eps_background, mu_background, subpixel):
        return UNCHANGED
    from .setup import compile_setup
    return compile_setup(scene, eps_background, mu_background, subpixel, include_edges)
