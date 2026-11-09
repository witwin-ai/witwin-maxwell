import gc
import os
import subprocess
import threading

import pytest
import torch

import witwin.maxwell as mw


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
MIB = 1024**2
# The 29^3 PML-padded grid plans 1,153,921,844 device bytes with a
# 504,285,588-byte hybrid minimum; this budget requires host factor storage.
DEVICE_LIMIT = 576 * MIB


def _simulation(limit=None, cells=20, eps_r=4.0, boundary=None):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.2, 0.2),) * 3),
        grid=mw.GridSpec.uniform(0.4 / cells),
        boundary=mw.BoundarySpec.pml(num_layers=4) if boundary is None else boundary,
        device="cuda",
    )
    scene.add_structure(mw.Structure(
        name="block", geometry=mw.Box(position=(0.0, 0.05, 0.0), size=(0.1,) * 3),
        material=mw.Material(eps_r=eps_r),
    ))
    scene.add_source(mw.PointDipole(
        position=(0.0, 0.0, 0.0), polarization="Ez", width=0.05,
        source_time=mw.CW(frequency=1e9),
    ))
    return mw.Simulation.fdfd(
        scene, frequency=1e9,
        solver=mw.GMRES(solver_type="direct", precision="double", device_memory_limit_bytes=limit),
    )


def _trim_pools():
    import cupy as cp
    from nvmath.internal.memory import free_reserved_memory

    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    # The direct runtime caches async workspace separately from tensor pools.
    free_reserved_memory()


def _used_memory():
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    if not uuid.startswith("GPU-"):
        uuid = f"GPU-{uuid}"
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits", "-i", uuid],
        text=True,
    )
    used = 0
    for line in output.splitlines():
        pid, memory = line.split(",")
        if int(pid) == os.getpid():
            used += int(memory) * MIB
    return used


@pytest.fixture(scope="module")
def warmed_device():
    # Load runtime libraries before measuring solve allocations.
    result = _simulation(cells=8).run()
    result.release_solver()
    del result
    _trim_pools()
    yield
    _trim_pools()


@requires_cuda
def test_direct_budget_matches_unlimited_fields_and_limits_peak(warmed_device):
    simulation = _simulation(DEVICE_LIMIT)
    _trim_pools()
    baseline = _used_memory()
    samples = [baseline]
    stop = threading.Event()

    def poll():
        while not stop.wait(0.05):
            samples.append(_used_memory())

    thread = threading.Thread(target=poll)
    thread.start()
    try:
        limited = simulation.run()
        torch.cuda.synchronize()
        samples.append(_used_memory())
    finally:
        stop.set()
        thread.join()
    peak_delta = max(samples) - baseline
    print(f"peak-minus-baseline: {peak_delta} bytes ({peak_delta / MIB:.2f} MiB)")
    assert peak_delta <= DEVICE_LIMIT + 256 * MIB
    limited.release_solver()
    _trim_pools()
    unlimited = _simulation().run()
    numerator = sum((limited.fields[key] - field).abs().square().sum()
                    for key, field in unlimited.fields.items())
    denominator = sum(field.abs().square().sum() for field in unlimited.fields.values())
    assert float((numerator / denominator).sqrt()) <= 1e-6
    unlimited.release_solver()


@requires_cuda
def test_release_solver_reclaims_direct_memory_and_is_idempotent(warmed_device):
    simulation = _simulation()
    _trim_pools()
    baseline = _used_memory()
    result = simulation.run()
    result.release_solver()
    _trim_pools()
    field_bytes = sum(field.numel() * field.element_size() for field in result.fields.values())
    after_release = _used_memory()
    release_delta = after_release - baseline
    print(f"after-release delta: {release_delta} bytes ({release_delta / MIB:.2f} MiB)")
    assert after_release <= baseline + field_bytes + 64 * MIB
    assert result.solver is None
    result.release_solver()
    assert result.solver is None


def test_direct_budget_constructor_validation():
    for limit in (True, False, 0, -1, 1.5, "1024"):
        with pytest.raises(ValueError, match="positive integer"):
            mw.GMRES(solver_type="direct", device_memory_limit_bytes=limit)
    with pytest.raises(ValueError, match="applies to solver_type='direct'"):
        mw.GMRES(solver_type="gmres", device_memory_limit_bytes=MIB)
    assert mw.GMRES().device_memory_limit_bytes is None
    assert mw.GMRES(solver_type="direct", device_memory_limit_bytes=MIB).device_memory_limit_bytes == MIB


@requires_cuda
def test_direct_budget_below_hybrid_minimum(warmed_device):
    with pytest.raises(ValueError, match="device_memory_limit_bytes=1048576.*hybrid minimum"):
        _simulation(MIB).run()
    simulation = _simulation(DEVICE_LIMIT)
    simulation.config.local_response = mw.LocalResponseConfig(
        mw.LocalDesignRegion(bounds=((-0.025, 0.025),) * 3),
        (mw.FieldProjection("probe", position=(0.07, 0.07, 0.0)),),
    )
    with pytest.raises(ValueError, match="local responses require GPU-resident direct factors"):
        simulation.prepare()


@requires_cuda
def test_limited_direct_refactorization_in_autograd_backward(warmed_device):
    gradients = []
    for release in (False, True):
        eps_r = torch.tensor(4.0, dtype=torch.float64, device="cuda", requires_grad=True)
        result = _simulation(DEVICE_LIMIT, cells=8, eps_r=eps_r).run()
        loss = sum(field.abs().square().mean() for field in result.fields.values())
        if release:
            result.release_solver()
        loss.backward()
        assert eps_r.grad is not None
        assert torch.isfinite(eps_r.grad).all()
        assert eps_r.grad.abs() > 0
        gradients.append(eps_r.grad.detach().clone())
        result.release_solver()
    # Double-precision factor reuse and refactorization may differ by roundoff.
    torch.testing.assert_close(gradients[1], gradients[0], rtol=1e-12, atol=0)


@requires_cuda
def test_limited_direct_forward_in_worker_thread(warmed_device):
    # Import on the main thread so a worker cannot inherit import-time thread-local state.
    import nvmath.sparse.advanced

    # A small cavity avoids nondeterministic parallel reductions in larger factors.
    simulation = _simulation(DEVICE_LIMIT, cells=2, boundary=mw.BoundarySpec.pec())
    reference = simulation.run()
    reference.release_solver()
    results = []
    errors = []

    def run():
        try:
            result = simulation.run()
            result.release_solver()
            results.append(result)
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    assert len(results) == 1
    assert results[0].fields.keys() == reference.fields.keys()
    assert any(torch.count_nonzero(field) > 0 for field in reference.fields.values())
    for key, field in reference.fields.items():
        assert torch.equal(results[0].fields[key], field)
