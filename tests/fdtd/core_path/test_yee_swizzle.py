"""CPU strip-policy/remap checks and separately marked CUDA integration gates."""
import gc
from itertools import product
from types import SimpleNamespace

import pytest
import torch

from witwin.maxwell.fdtd.runtime import core_path


L2_6_MIB = 6 * 1024 * 1024
FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Core path requires CUDA.")


@pytest.mark.parametrize("ny,nz,expected", [
    (128, 128, 0), (256, 256, 0), (320, 320, 0),
    (352, 352, 32), (384, 384, 32), (448, 448, 32),
    (96, 384, 0), (768, 384, 32),
])
def test_strip_policy(ny, nz, expected):
    assert core_path.yee_block_strip(ny, nz, L2_6_MIB) == expected


@pytest.mark.parametrize("l2_bytes,expected", [(53, 32), (54, 0), (55, 0)])
def test_strip_policy_strict_integer_threshold(l2_bytes, expected):
    assert core_path.yee_block_strip(1, 1, l2_bytes) == expected


def _remap(kb, jb, ib, gx, gy, gz, strip):
    """Reference of the unsigned kernel remap; coordinates are block indices."""
    if strip == 0:
        return kb, jb, ib
    strip = min(strip, gy)
    linear = kb + gx * (jb + gy * ib)
    base = gx * strip * gz
    s, r = divmod(linear, base)
    w = min(strip, gy - s * strip)
    ib, r2 = divmod(r, gx * w)
    jb, kb = divmod(r2, gx)
    return kb, s * strip + jb, ib


@pytest.mark.parametrize("gx,gy,gz", [(1, 1, 1), (3, 5, 4), (2, 33, 3), (4, 65, 5)])
@pytest.mark.parametrize("strip", [0, 1, 7, 32, 1000, 2**32 - 1])
def test_remap_bijection_and_order(gx, gy, gz, strip):
    stock = [(kb, jb, ib) for ib, jb, kb in product(range(gz), range(gy), range(gx))]
    remapped = [_remap(*block, gx, gy, gz, strip) for block in stock]
    assert len(set(remapped)) == len(stock)
    assert set(remapped) == set(stock)
    if strip == 0 or strip >= gy:
        assert remapped == stock
    else:
        expected = [(kb, jb, ib) for start in range(0, gy, strip)
                    for ib in range(gz) for jb in range(start, min(start + strip, gy))
                    for kb in range(gx)]
        assert remapped == expected


def test_runtime_records_strips_and_reads_l2_once(monkeypatch):
    from witwin.maxwell.fdtd.cuda.core import build

    launches, devices = [], []
    tensor = torch.empty(0)
    solver = SimpleNamespace(device=torch.device("cpu"), Nx=24, Ny=352, Nz=352, uses_cpml=False)
    for name in FIELDS:
        setattr(solver, name, tensor)
    for family, axis in product("he", "xyz"):
        setattr(solver, f"inv_d{axis}_{family}", tensor)
    for axis, side in product("xyz", ("low", "high")):
        setattr(solver, f"boundary_{axis}_{side}_code", 0)

    def l2_bytes(device):
        devices.append(device)
        return L2_6_MIB

    def launch(*args):
        launches.append(args)
        return object()

    monkeypatch.setattr(core_path, "_device_l2_bytes", l2_bytes)
    monkeypatch.setattr(build, "build_extension", lambda: SimpleNamespace(CoreLaunch=launch))
    monkeypatch.setattr(core_path, "_compress_coefficients",
                        lambda *args: (tensor, tensor, [0.] * 6, True))
    runtime = core_path.CorePathRuntime(solver)
    assert devices == [solver.device]
    assert runtime.yee_block_strip == {"electric": 32, "magnetic": 32}
    assert [args[1][-1] for args in launches] == [32, 32]


def test_host_runtime_records_stock_order_without_cuda(monkeypatch):
    from witwin.maxwell.fdtd.cuda.core import build

    launches = []
    tensor = torch.empty(0)
    solver = SimpleNamespace(device="cpu", Nx=24, Ny=352, Nz=352, uses_cpml=False)
    for name in FIELDS:
        setattr(solver, name, tensor)
    for family, axis in product("he", "xyz"):
        setattr(solver, f"inv_d{axis}_{family}", tensor)
    for axis, side in product("xyz", ("low", "high")):
        setattr(solver, f"boundary_{axis}_{side}_code", 0)

    def forbidden(*args, **kwargs):
        pytest.fail("host runtime construction must not query CUDA device properties")

    monkeypatch.setattr(torch.cuda, "get_device_properties", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(build, "build_extension",
                        lambda: SimpleNamespace(CoreLaunch=lambda *args: launches.append(args)))
    monkeypatch.setattr(core_path, "_compress_coefficients",
                        lambda *args: (tensor, tensor, [0.] * 6, True))
    runtime = core_path.CorePathRuntime(solver)
    assert runtime.yee_block_strip == {"electric": 0, "magnetic": 0}
    assert [args[1][-1] for args in launches] == [0, 0]


@pytest.mark.parametrize("mismatch", ["weights", "active"])
def test_observer_shared_storage_invariant(monkeypatch, mismatch):
    from witwin.maxwell.fdtd import observers
    from witwin.maxwell.fdtd.runtime import core_tables

    weights, active = object(), object()
    group = [None, None, None, weights, active]
    group[3 if mismatch == "weights" else 4] = object()
    engine = SimpleNamespace(state=object(), weights=weights,
                             slot=SimpleNamespace(active=active), groups=[group])
    monkeypatch.setattr(core_tables, "compile_monitor_routes", lambda solver: engine)
    monkeypatch.setattr(observers, "uses_native_point_observers", lambda solver: False)
    runtime = core_path.CorePathRuntime.__new__(core_path.CorePathRuntime)
    solver = SimpleNamespace(scene=SimpleNamespace())
    with pytest.raises(RuntimeError, match="must share weights and active tensors"):
        runtime.prepare_step(solver, use_gpu_dft=False, time_steps=1)


def _simulation(shape, memory_mode="slab", steps=1):
    import witwin.maxwell as mw
    from benchmark.core_path.scenes import _grid_scene, FREQUENCY

    return mw.Simulation.fdtd(
        _grid_scene(shape).add_monitor(mw.PointMonitor("probe", position=(0.0, 0.0, 0.0), fields=("Ez",)))
        .clone(device="cuda"), frequency=FREQUENCY,
        run_time=mw.TimeConfig(time_steps=steps), full_field_dft=False,
        cpml_config={"memory_mode": memory_mode}, fdtd_core_path=True, cuda_graph=True,
        shutoff=0.0,
    )


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("memory_mode", ["dense", "slab"])
def test_forced_strips_equal(monkeypatch, memory_mode):
    reference = None
    for strip in (0, 7, 32, 1000):
        monkeypatch.setattr(core_path, "yee_block_strip", lambda *args: strip)
        simulation = _simulation((24, 66, 260), memory_mode, steps=12)
        prepared = simulation.prepare()
        solver = prepared.solver
        assert (solver.Ny + 1) // 2 == 33
        assert solver._core_runtime.yee_block_strip == {"electric": strip, "magnetic": strip}
        # Nonzero initial fields exercise all six components and CPML slabs.
        generator = torch.Generator(device="cpu").manual_seed(142)
        for name in FIELDS:
            field = getattr(solver, name)
            field.copy_(torch.randn(field.shape, generator=generator) * 2.0**-20)
        prepared.run()
        fields = {name: getattr(solver, name).cpu().clone() for name in FIELDS}
        assert all(torch.isfinite(field).all() and torch.count_nonzero(field) for field in fields.values())
        if reference is None:
            reference = fields
        else:
            for name in FIELDS:
                assert torch.equal(fields[name], reference[name]), (strip, name)
        del solver, prepared, simulation
        gc.collect()
        torch.cuda.empty_cache()


def _check_prepared_record(n, expected=None):
    # A shallow x keeps dense setup below 3 GiB; the policy depends only on ny,nz.
    simulation = _simulation((24, n, n))
    prepared = simulation.prepare()
    solver = prepared.solver
    assert (solver.Nx, solver.Ny, solver.Nz) == (24, n, n)
    assert solver._cpml_memory_mode == "slab"
    l2 = torch.cuda.get_device_properties(solver.device).L2_cache_size
    policy = core_path.yee_block_strip(solver.Ny, solver.Nz, l2)
    assert solver._core_runtime.yee_block_strip == {"electric": policy, "magnetic": policy}
    if expected is not None:
        assert policy == expected
    del solver, prepared, simulation
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("n,expected_6_mib", [(128, 0), (256, 0), (384, 32)])
def test_real_device_record(n, expected_6_mib):
    l2 = torch.cuda.get_device_properties(torch.device("cuda")).L2_cache_size
    _check_prepared_record(n, expected_6_mib if l2 == L2_6_MIB else None)


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("n", [320, 352])
@pytest.mark.parametrize("l2_case", ["6_mib", "below_boundary", "above_boundary"])
def test_mocked_l2_record(monkeypatch, n, l2_case):
    boundary = 3 * (36 * n * n) // 2
    l2, expected = {
        "6_mib": (L2_6_MIB, 0 if n == 320 else 32),
        "below_boundary": (boundary - 1, 32),
        "above_boundary": (boundary + 1, 0),
    }[l2_case]
    get_properties = torch.cuda.get_device_properties

    class Properties:
        L2_cache_size = l2

        def __init__(self, device):
            self.actual = get_properties(device)

        def __getattr__(self, name):
            return getattr(self.actual, name)

    monkeypatch.setattr(torch.cuda, "get_device_properties", Properties)
    _check_prepared_record(n, expected)


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("shape,strip,message", [
    ((24, 66, 260), -1, "nonnegative"),
    ((24, 66, 260), 2**32, "32-bit"),
    ((65536, 131072, 128), 32, "fewer than 2\\^32 blocks"),
    ((2**32 - 1,) * 3, 32, "fewer than 2\\^32 blocks"),
])
def test_native_launch_integer_guards(shape, strip, message):
    from witwin.maxwell.fdtd.cuda.core.build import build_extension

    # The guards precede pointer assembly, so invalid launches need no field allocation.
    integers = [*shape, *([0] * 6), strip]
    with pytest.raises(RuntimeError, match=message):
        build_extension().CoreLaunch([], integers, [0.] * 6, False, 0, True, False, False, False)
