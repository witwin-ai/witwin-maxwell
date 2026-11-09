"""Measure mesh material compilation without changing the material compiler.

Run with the maxwell interpreter on an explicitly assigned CUDA device. The
mesh archive must contain ``vertices`` (metres) and ``faces``. Examples::

    python scripts/dev/fdtd/mesh_subpixel_compile_perf.py compile --cells 16
    python scripts/dev/fdtd/mesh_subpixel_compile_perf.py micro --cells 32

JSON output includes synchronized wall times, SDF call/point counts by compiler
phase, and the core accelerator registration. Each compile uses a fresh scene.
The optional historical native audit never installs a backend.
"""

from __future__ import annotations

import tempfile
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from itertools import combinations
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

import witwin.maxwell as mw
from witwin.core.geometry import mesh_sdf
from witwin.maxwell.compiler import materials
from witwin.maxwell.scene import prepare_scene


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def make_scene(args, kind):
    data = np.load(args.mesh)
    vertices, faces = data["vertices"], data["faces"]
    center = (vertices.min(0) + vertices.max(0)) / 2
    span = float(np.ptp(vertices, axis=0).max()) * 2.5
    # Binary-exact span/dl keeps ceil(span/dl) at the requested cell count.
    unit = 2.0 ** (np.floor(np.log2(span)) - 4)
    span = float(np.ceil(span / unit) * unit)
    domain_center = np.round(center / (span / 64)) * (span / 64)
    bounds = tuple((float(c - span / 2), float(c + span / 2)) for c in domain_center)
    scene = mw.Scene(
        domain=mw.Domain(bounds=bounds), grid=mw.GridSpec.uniform(span / args.cells),
        boundary=mw.BoundarySpec.pml(num_layers=args.pml), device="cuda",
        subpixel_samples=mw.SubpixelSpec(samples=args.samples, averaging="polarized"),
    )
    if kind == "mesh":
        geometry = mw.Mesh(vertices, faces, recenter=False, scale=1.0, fill_mode="solid")
    else:
        geometry = mw.Sphere(position=center, radius=float(np.ptp(vertices, axis=0).max()) / 2)
    scene.add_structure(mw.Structure(geometry=geometry, material=mw.Material(eps_r=6.83)))
    return scene


@contextmanager
def record_sdf(geometry_type, synchronize):
    original_sdf = geometry_type.signed_distance
    original_edge = materials._compile_edge_component
    phase = ["nodes"]
    records = defaultdict(lambda: {"calls": 0, "points": 0, "seconds": 0.0, "sizes": Counter()})

    def sdf(self, x, y, z):
        entry = records[phase[0]]
        count = x.numel() * y.numel() * z.numel() if x.ndim == y.ndim == z.ndim == 1 else x.numel()
        entry["calls"] += 1
        entry["points"] += count
        entry["sizes"][count] += 1
        if synchronize:
            result, elapsed = timed(lambda: original_sdf(self, x, y, z))
            entry["seconds"] += elapsed
            return result
        return original_sdf(self, x, y, z)

    def edge(scene, component, **kwargs):
        previous = phase[0]
        phase[0] = component
        try:
            return original_edge(scene, component, **kwargs)
        finally:
            print(json.dumps({"completed_component": component}), flush=True)
            phase[0] = previous

    geometry_type.signed_distance = sdf
    materials._compile_edge_component = edge
    try:
        yield records
    finally:
        geometry_type.signed_distance = original_sdf
        materials._compile_edge_component = original_edge


def compile_case(args, kind):
    scene = make_scene(args, kind)
    # Initialize the context and geometry state before recording compile work.
    geometry = scene.structures[0].geometry
    q = torch.zeros((1, 1, 1), device="cuda")
    geometry.signed_distance(q, q, q)
    prepared_scene = prepare_scene(scene)
    segments = {}
    with record_sdf(type(geometry), args.sdf_timers) as records:
        if args.entry == "compile":
            _, elapsed = timed(prepared_scene.compile_materials)
        elif args.entry == "nodes":
            _, elapsed = timed(lambda: materials.compile_material_model(
                prepared_scene, subpixel=scene.subpixel, include_edges=False))
        elif args.entry == "fdfd":
            prepared, prepare_seconds = timed(lambda: mw.Simulation.fdfd(scene, frequency=1e9).prepare())
            _, material_seconds = timed(prepared.solver._ensure_material_components)
            elapsed = prepare_seconds + material_seconds
            segments = dict(prepare_seconds=prepare_seconds, material_seconds=material_seconds)
        else:
            _, elapsed = timed(lambda: mw.Simulation.fdtd(
                scene, frequency=1e9, run_time=mw.TimeConfig(time_steps=1),
            ).prepare())
    return dict(kind=kind, entry=args.entry, seconds=elapsed, segments=segments,
                physical_cells=args.cells, pml=args.pml,
                domain_bounds=scene.domain.bounds,
                node_shape=[prepared_scene.Nx, prepared_scene.Ny, prepared_scene.Nz],
                samples=args.samples, sdf_timers=args.sdf_timers, phases=dict(records))


def micro(args):
    scene = make_scene(args, "mesh")
    geometry = scene.structures[0].geometry
    prepared = prepare_scene(scene)
    points = torch.stack((prepared.X, prepared.Y, prepared.Z), -1).reshape(-1, 3)
    vertices, faces, triangles, bvh = geometry._sdf_query_data(device=points.device, dtype=points.dtype)
    calls = {
        "unsigned_torch": lambda: mesh_sdf._triangle_mesh_unsigned_distance_torch_from_triangles(points, triangles),
        "winding_torch": lambda: mesh_sdf._triangle_mesh_winding_angle_torch_from_triangles(points, triangles),
        "signed_dispatch": lambda: geometry.signed_distance(prepared.X, prepared.Y, prepared.Z),
    }
    h = float(prepared.grid.dx) / args.samples
    padding = 0.75 * np.sqrt(3) * h + h / 4 + 1e-6
    lo, hi = vertices.amin(0) - padding, vertices.amax(0) + padding
    active_mask = ((points >= lo) & (points <= hi)).all(1)
    active_points = points[active_mask].contiguous()
    calls["aabb_packed_torch"] = lambda: mesh_sdf._triangle_mesh_smooth_signed_distance_torch(
        active_points, vertices, faces, _triangles=triangles)
    band_distance = geometry.signed_distance(prepared.X, prepared.Y, prepared.Z).reshape(-1)
    band_mask = band_distance.abs() <= 0.75 * np.sqrt(3) * h
    band_points = points[band_mask].contiguous()
    calls["band_packed_torch"] = lambda: mesh_sdf._triangle_mesh_smooth_signed_distance_torch(
        band_points, vertices, faces, _triangles=triangles)
    output = dict(points=len(points), triangles=len(triangles), bvh=bvh is not None, timings={})
    output.update(aabb_points=len(active_points), band_points=len(band_points), padding=padding)
    output.update(aabb_bit_identical=torch.equal(calls["aabb_packed_torch"](), band_distance[active_mask]),
                  band_bit_identical=torch.equal(calls["band_packed_torch"](), band_distance[band_mask]))
    for name, fn in calls.items():
        fn()
        times = [timed(fn)[1] for _ in range(args.repeats)]
        output["timings"][name] = dict(seconds=times, median=float(np.median(times)))
    return output


def workload(args):
    """Count exact float32 center reuse and conservatively bound the fill band."""
    scene = make_scene(args, "mesh")
    prepared = prepare_scene(scene)
    geometry = scene.structures[0].geometry
    sets = []
    signatures = [(False, False, False)] + [tuple(i in materials._EDGE_STAGGER_AXES[c] for i in range(3))
                                         for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")]
    for axis in "xyz":
        axis_sets = {}
        for staggered in (False, True):
            base = materials._edge_axis_base_1d(prepared, axis, staggered)
            offsets = materials._edge_axis_offsets(prepared, axis, args.samples, staggered)
            values = torch.cat([base + offset for offset in offsets]).cpu().numpy()
            axis_sets[staggered] = set(values.view(np.int32).tolist())
        sets.append(axis_sets)
    unique = 0
    for size in range(1, 8):
        for group in combinations(signatures, size):
            intersections = [set.intersection(*(sets[a][s[a]] for s in group)) for a in range(3)]
            unique += (-1) ** (size + 1) * np.prod([len(s) for s in intersections])
    requested = args.samples ** 3 * sum(np.prod([prepared.Nx - int(s[0]), prepared.Ny - int(s[1]), prepared.Nz - int(s[2])])
                                          for s in signatures)
    coords = materials._coordinate_grids(prepared, materials._sample_offsets(prepared, (args.samples,) * 3)[0])
    d = geometry.signed_distance(*coords)
    h = float(prepared.grid.dx) / args.samples
    threshold = 0.75 * np.sqrt(3) * h
    band = d.abs() <= threshold
    # Exact zeros/ones outside this band follow from |shift| <= width/4 and
    # width <= sqrt(hx^2+hy^2+hz^2). This measures one representative offset.
    from witwin.maxwell.compiler.subpixel import _subsample_fill_fraction
    fill = _subsample_fill_fraction(geometry, *coords, (h, h, h))
    outer_expected = (d < 0).to(fill.dtype)
    return dict(requested_centers=int(requested), unique_float32_centers=int(unique),
                center_reuse_ratio=float(requested / unique), representative_points=d.numel(),
                band_points=int(band.sum()), band_fraction=float(band.float().mean()), threshold=threshold,
                outside_band_fill_mismatches=int(((fill != outer_expected) & ~band).sum()),
                mesh_bounds=[geometry.vertices.amin(0).tolist(), geometry.vertices.amax(0).tolist()],
                domain_bounds=scene.domain.bounds)


def node_parity(args):
    """Check the existing include_edges flag without changing consumer routing."""
    prepared = prepare_scene(make_scene(args, "mesh"))
    full, full_seconds = timed(lambda: materials.compile_material_model(
        prepared, subpixel=prepared.subpixel, include_edges=True))
    nodes, node_seconds = timed(lambda: materials.compile_material_model(
        prepared, subpixel=prepared.subpixel, include_edges=False))

    def leaves(value, path=""):
        if torch.is_tensor(value):
            yield path, value
        elif isinstance(value, dict):
            for key, item in value.items():
                if key != "edge_components":
                    yield from leaves(item, f"{path}/{key}")
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                yield from leaves(item, f"{path}/{index}")

    a, b = dict(leaves(full)), dict(leaves(nodes))
    return dict(full_seconds=full_seconds, node_seconds=node_seconds, tensors=len(a),
                same_tensor_keys=a.keys() == b.keys(),
                mismatches=[key for key in a if not torch.equal(a[key], b[key])],
                full_has_edges=full["edge_components"] is not None, node_has_edges=nodes["edge_components"] is not None)


def native_audit(args):
    """Benchmark historical forward CUDA code through a scratch-only loader.

    Export core revision 6e75274's geometry/cuda/mesh_sdf_kernels.cu to the
    supplied path. Only its unchanged direct-query helpers/kernel are loaded;
    no accelerator is registered and no compiler behavior changes.
    """
    os.environ["CUPY_CACHE_DIR"] = str(args.native_source.parent / "cupy_cache")
    import cupy as cp

    source = args.native_source.read_text()
    helpers = source[source.index("constexpr float kSmoothEps"):source.index("__global__ void query_unsigned_distance_bvh_kernel")]
    module = cp.RawModule(code="#include <cuda_runtime.h>\n" + helpers,
                          options=("--std=c++17", "--fmad=false"),
                          name_expressions=("query_distance_and_winding_kernel",))
    kernel = module.get_function("query_distance_and_winding_kernel")
    geometry = make_scene(args, "mesh").structures[0].geometry
    device = torch.device("cuda")
    vertices, faces, triangles, _ = geometry._sdf_query_data(device=device, dtype=torch.float32)
    generator = torch.Generator(device=device).manual_seed(137)
    lo, hi = vertices.amin(0), vertices.amax(0)
    random = lo + (hi - lo) * (1.5 * torch.rand((8192, 3), generator=generator, device=device) - 0.25)
    points = torch.cat((random, vertices, triangles.mean(1)), 0).contiguous()
    output = {"source_sha256": hashlib.sha256(args.native_source.read_bytes()).hexdigest(), "scales": []}
    for scale in (1.0, 1e-3, 1e3):
        queries, tris = (points * scale).contiguous(), (triangles * scale).contiguous()
        unsigned = torch.empty(len(queries), device=device)
        angle = torch.empty_like(unsigned)
        closest = torch.empty(len(queries), device=device, dtype=torch.int32)

        def native():
            with cp.cuda.Stream.from_external(torch.cuda.current_stream()):
                kernel(((len(queries) + 255) // 256,), (256,), (
                    cp.from_dlpack(tris), cp.from_dlpack(queries), np.int32(len(tris)), np.int32(len(queries)),
                    cp.from_dlpack(unsigned), cp.from_dlpack(angle), cp.from_dlpack(closest),
                ))
            return mesh_sdf._signed_distance_from_unsigned_and_winding(unsigned, angle, 0.5)

        native()
        signed_native, elapsed = timed(native)
        unsigned_torch = mesh_sdf._triangle_mesh_unsigned_distance_torch_from_triangles(queries, tris)
        angle_torch = mesh_sdf._triangle_mesh_winding_angle_torch_from_triangles(queries, tris)
        signed_torch = mesh_sdf._signed_distance_from_unsigned_and_winding(unsigned_torch, angle_torch, 0.5)
        error = (signed_native - signed_torch).abs()
        output["scales"].append(dict(
            scale=scale, points=len(queries), seconds=elapsed,
            bit_identical=torch.equal(signed_native, signed_torch),
            signed_max_abs=float(error.max()), signed_mean_abs=float(error.mean()),
            unsigned_max_abs=float((unsigned - unsigned_torch).abs().max()),
            winding_max_abs=float((angle - angle_torch).abs().max()),
            sign_disagreements=int(((signed_native < 0) != (signed_torch < 0)).sum()),
            random_sign_disagreements=int(((signed_native[:8192] < 0) != (signed_torch[:8192] < 0)).sum()),
            core_inside=int((signed_torch[:8192] < 0).sum()),
        ))
    prepared = prepare_scene(make_scene(args, "mesh"))
    queries = torch.stack((prepared.X, prepared.Y, prepared.Z), -1).reshape(-1, 3)
    unsigned = torch.empty(len(queries), device=device)
    angle = torch.empty_like(unsigned)
    closest = torch.empty(len(queries), device=device, dtype=torch.int32)
    tris = triangles
    native()
    output["grid_points"] = len(queries)
    output["grid_seconds"] = [timed(native)[1] for _ in range(args.repeats)]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("compile", "micro", "native-audit", "workload", "node-parity"))
    parser.add_argument("--native-source", type=Path)
    parser.add_argument("--mesh", type=Path, default=Path(tempfile.gettempdir()) / "witwin_cov_meshperf" / "mesh168.npz")
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--pml", type=int, default=8)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--kind", choices=("mesh", "sphere", "both"), default="both")
    parser.add_argument("--entry", choices=("compile", "nodes", "fdfd", "fdtd"), default="compile")
    parser.add_argument("--sdf-timers", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fdtd-build-dir", type=Path, default=Path(tempfile.gettempdir()) / "witwin_cov_meshperf" / "fdtd_cuda")
    parser.add_argument("--timeout", type=int, default=570, help="Maximum run seconds after imports (Linux alarm).")
    args = parser.parse_args()
    signal.alarm(args.timeout)
    torch.cuda.init()
    metadata = dict(core=importlib.metadata.version("witwin"), torch=torch.__version__,
                    device=torch.cuda.get_device_name(), mesh_sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
                    accelerator=repr(mesh_sdf._MESH_SDF_ACCELERATOR),
                    accelerator_error=mesh_sdf._mesh_sdf_build_error(), core_source=mesh_sdf.__file__)
    if args.mode == "compile" and args.entry == "fdtd":
        os.environ["WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR"] = str(args.fdtd_build_dir)
        os.environ.setdefault("MAX_JOBS", "4")
        from witwin.maxwell.fdtd.cuda.backend import get_compiled_extension
        _, load_seconds = timed(get_compiled_extension)
        metadata["fdtd_extension_load_seconds"] = load_seconds
    print(json.dumps(metadata), flush=True)
    results = []
    if args.mode == "compile":
        for kind in ("sphere", "mesh") if args.kind == "both" else (args.kind,):
            result = compile_case(args, kind)
            results.append(result)
            print(json.dumps(result), flush=True)
    else:
        fn = {"micro": micro, "native-audit": native_audit, "workload": workload, "node-parity": node_parity}[args.mode]
        results.append(fn(args))
        print(json.dumps(results[-1]), flush=True)
    if args.output:
        args.output.write_text(json.dumps(dict(metadata=metadata, results=results), indent=2) + "\n")


if __name__ == "__main__":
    main()
