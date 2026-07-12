"""P6 acceptance package for the fully-native FDTD adjoint reverse hot path.

This module is the acceptance harness for the P6 effort that moved the FDTD
adjoint per-step reverse *math* off Torch and into fused native CUDA kernels. It
locks four guarantees in one place, parametrized over every differentiable scene
class that has a native reverse runner:

1. ``test_hotpath_reverse_region_is_native`` -- the ``__torch_dispatch__`` hot-path
   gate. Under the forced ``native`` backend it traces every dispatched op inside
   the per-step reverse region (the ``reverse_step`` call) and asserts (a) the
   step selected the matching ``NATIVE_*`` backend, (b) the fused reverse-math
   kernels actually ran (native ``witwin_maxwell_fdtd_cuda.*`` custom ops are
   present), and (c) every Torch ``aten.*`` op that ran is in the calibrated
   whitelist of legitimately-Torch orchestration / replay ops. The reverse math
   itself never shows up as an ``aten.*`` op because it is a custom CUDA op.

2. ``test_torch_reference_reverse_region_leaves_the_whitelist`` -- the calibration
   guard. The same reverse region under ``torch_reference`` runs the analytic math
   in Torch, which (a) invokes zero native reverse-math kernels and (b) dispatches
   ``aten.*`` ops *outside* the whitelist (``aten.zeros`` / ``aten.new_zeros`` /
   ``aten.bitwise_*`` / ...). This proves the whitelist is tight: a native runner
   silently regressing to the Torch reference would trip the gate.

3. ``test_native_reverse_matches_reference`` -- the step-level native==reference
   parity matrix. Each class' fused native reverse reproduces the analytic Torch
   reference to the single-precision step-level bar.

4. ``test_additive_sigma_m_gradient_matches_fd`` and
   ``test_multi_frequency_dispersive_gradient_matches_fd`` -- two end-to-end
   finite-difference gradient checks that extend the capability matrix: an
   additive magnetic-conductivity (``sigma_m``) medium (folded into the native
   standard/CPML magnetic decay) and a multi-frequency objective over an
   electric-dispersive design (native dispersive reverse + multi-frequency DFT
   seeds).

The residual Torch on the native hot path is exactly the calibrated whitelist:
the once-per-step mid-H / ADE replay, the ``dynamic_electric_curls`` coefficient
cast, and the per-step source-term VJP. Those are the "same bar" remainders every
landed P6 item reports; the reverse *math* is fully native. See
``docs/dev/fdtd/native_cuda/gradient_capability_matrix.md`` for the full map.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import witwin.maxwell as mw
from witwin.maxwell.fdtd.adjoint import reference as adjoint_reference
from witwin.maxwell.fdtd.adjoint.dispatch import reverse_step
from witwin.maxwell.fdtd.checkpoint import checkpoint_schema
from tests.gradients.test_fdtd_adjoint_b_complex_classes import _physical_reverse_state
from tests.gradients.test_fdtd_adjoint_bridge import (
    _fake_bloch_reverse_solver,
    _fake_conductive_cpml_reverse_solver,
    _fake_cpml_reverse_solver,
    _fake_dispersive_cpml_reverse_solver,
    _fake_dispersive_standard_reverse_solver,
    _fake_full_aniso_cpml_reverse_solver,
    _fake_kerr_cpml_reverse_solver,
    _fake_standard_reverse_solver,
    _fake_tfsf_cpml_reverse_solver,
    _bloch_reverse_state_shapes,
    _cpml_reverse_state_shapes,
    _dispersive_cpml_reverse_state_shapes,
    _dispersive_standard_reverse_state_shapes,
    _move_solver_tensors_to_cuda,
    _tfsf_cpml_reverse_state_shapes,
)

_ENV = "WITWIN_MAXWELL_FDTD_ADJOINT_BACKEND"
_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA for the native FDTD adjoint reverse"
)

# ---------------------------------------------------------------------------
# Calibrated Torch-op whitelist for the native reverse region.
#
# These are the ``aten.*`` overload-packet names that the *legitimately-Torch*
# once-per-step orchestration and replay dispatch when the reverse math runs on
# fused native kernels: the mid-H magnetic replay and the electric-dispersive ADE
# replay, the ``dynamic_electric_curls`` coefficient cast, the reverse-context
# tensor allocation, and the ``_accumulate_source_term_gradients`` bookkeeping.
# The list was calibrated by tracing every native class below; it deliberately
# excludes the allocation / masking ops (``aten.zeros``, ``aten.new_zeros``,
# ``aten.bitwise_*``, ``aten.where``, ...) that only the Torch analytic reverse
# math emits, so a native runner regressing to Torch trips the gate.
# ---------------------------------------------------------------------------
_REVERSE_REGION_ATEN_WHITELIST = frozenset(
    {
        "aten._local_scalar_dense",
        "aten.add",
        "aten.add_",
        "aten.clamp",
        "aten.clone",
        "aten.complex",
        "aten.copy_",
        "aten.detach",
        "aten.div",
        "aten.empty_like",
        "aten.exp",
        "aten.mul",
        "aten.neg",
        "aten.pow",
        "aten.reciprocal",
        "aten.rsub",
        "aten.select",
        "aten.slice",
        "aten.sub",
        "aten.sum",
        "aten.unsqueeze",
        "aten.view",
        "aten.view_as_real",
        "aten.zeros_like",
    }
)

_NATIVE_CUDA_OP_PREFIX = "witwin_maxwell_fdtd_cuda."


class _DispatchTrace(TorchDispatchMode):
    """Record the overload-packet name of every dispatched op, with multiplicity."""

    def __init__(self):
        self.counts: dict[str, int] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        packet = getattr(func, "overloadpacket", None)
        name = str(packet) if packet is not None else str(func)
        self.counts[name] = self.counts.get(name, 0) + 1
        return func(*args, **(kwargs or {}))

    def aten_ops(self) -> set[str]:
        return {name for name in self.counts if name.startswith("aten.")}

    def native_cuda_ops(self) -> set[str]:
        return {name for name in self.counts if name.startswith(_NATIVE_CUDA_OP_PREFIX)}


# ---------------------------------------------------------------------------
# Native reverse-step case builders: one per differentiable scene class that has
# a fused native CUDA reverse runner. Each returns
# ``(solver, forward_state, adjoint_state, eps_ex, eps_ey, eps_ez, native_label)``.
# The fake reverse solvers are the same canonical per-class reverse setups the
# native CUDA parity suite uses; the Bloch+dispersive case uses a physical
# checkpoint state from a real prepared solver (no fake builder exists for it).
# ---------------------------------------------------------------------------


def _random_state(solver, shapes, *, seed):
    torch.manual_seed(seed)
    return {
        name: torch.randn(shapes[name], device="cuda", dtype=torch.float32)
        for name in checkpoint_schema(solver).state_names
    }


def _eps_leaves(forward_state):
    eps_ex = torch.full_like(forward_state["Ex"], 2.3, requires_grad=True)
    eps_ey = torch.full_like(forward_state["Ey"], 2.7, requires_grad=True)
    eps_ez = torch.full_like(forward_state["Ez"], 3.1, requires_grad=True)
    return eps_ex, eps_ey, eps_ez


def _standard_case(seed):
    solver = _move_solver_tensors_to_cuda(_fake_standard_reverse_solver())
    shapes = {"Ex": (2, 4, 5), "Ey": (3, 3, 5), "Ez": (3, 4, 4), "Hx": (3, 3, 4), "Hy": (2, 4, 4), "Hz": (2, 3, 5)}
    forward_state = _random_state(solver, shapes, seed=seed)
    adjoint_state = {name: torch.randn_like(tensor) for name, tensor in forward_state.items()}
    return (solver, forward_state, adjoint_state, *_eps_leaves(forward_state), "native_standard")


def _state_shape_case(solver, shapes, label, seed):
    forward_state = _random_state(solver, shapes, seed=seed)
    adjoint_state = {name: torch.randn_like(tensor) for name, tensor in forward_state.items()}
    return (solver, forward_state, adjoint_state, *_eps_leaves(forward_state), label)


def _bloch_dispersive_case(seed):
    solver, forward_state, adjoint_state = _physical_reverse_state(seed=seed)
    eps_ex = solver.eps_Ex.detach().clone().requires_grad_(True)
    eps_ey = solver.eps_Ey.detach().clone().requires_grad_(True)
    eps_ez = solver.eps_Ez.detach().clone().requires_grad_(True)
    return (solver, forward_state, adjoint_state, eps_ex, eps_ey, eps_ez, "native_bloch_dispersive")


def _build_native_case(name, *, seed):
    if name == "standard":
        return _standard_case(seed)
    if name == "cpml":
        return _state_shape_case(
            _move_solver_tensors_to_cuda(_fake_cpml_reverse_solver()), _cpml_reverse_state_shapes(), "native_cpml", seed
        )
    if name == "conductive":
        return _state_shape_case(
            _move_solver_tensors_to_cuda(_fake_conductive_cpml_reverse_solver()),
            _cpml_reverse_state_shapes(),
            "native_conductive",
            seed,
        )
    if name == "kerr":
        return _state_shape_case(
            _move_solver_tensors_to_cuda(_fake_kerr_cpml_reverse_solver()), _cpml_reverse_state_shapes(), "native_kerr", seed
        )
    if name == "full_aniso":
        return _state_shape_case(
            _move_solver_tensors_to_cuda(_fake_full_aniso_cpml_reverse_solver()),
            _cpml_reverse_state_shapes(),
            "native_full_aniso",
            seed,
        )
    if name == "bloch":
        return _state_shape_case(
            _move_solver_tensors_to_cuda(_fake_bloch_reverse_solver()), _bloch_reverse_state_shapes(), "native_bloch", seed
        )
    if name == "dispersive_standard":
        return _state_shape_case(
            _move_solver_tensors_to_cuda(_fake_dispersive_standard_reverse_solver()),
            _dispersive_standard_reverse_state_shapes(),
            "native_dispersive",
            seed,
        )
    if name == "dispersive_cpml":
        return _state_shape_case(
            _move_solver_tensors_to_cuda(_fake_dispersive_cpml_reverse_solver()),
            _dispersive_cpml_reverse_state_shapes(),
            "native_dispersive",
            seed,
        )
    if name == "tfsf":
        return _state_shape_case(
            _move_solver_tensors_to_cuda(_fake_tfsf_cpml_reverse_solver()),
            _tfsf_cpml_reverse_state_shapes(),
            "native_tfsf",
            seed,
        )
    if name == "bloch_dispersive":
        return _bloch_dispersive_case(seed)
    raise ValueError(f"Unknown native reverse case {name!r}.")


_NATIVE_CLASSES = (
    "standard",
    "cpml",
    "conductive",
    "kerr",
    "full_aniso",
    "bloch",
    "dispersive_standard",
    "dispersive_cpml",
    "tfsf",
    "bloch_dispersive",
)


def _seed_for(class_name, base):
    """Deterministic per-class seed (``hash`` is per-process randomized)."""
    return base + _NATIVE_CLASSES.index(class_name)


def _run_reverse(monkeypatch, mode, case):
    solver, forward_state, adjoint_state, eps_ex, eps_ey, eps_ez, _label = case
    monkeypatch.setenv(_ENV, mode)
    return reverse_step(
        solver,
        forward_state,
        adjoint_state,
        time_value=0.0,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
    )


# ---------------------------------------------------------------------------
# 1. Hot-path gate: the per-step reverse region runs native math + whitelist only.
# ---------------------------------------------------------------------------


@_CUDA
@pytest.mark.parametrize("class_name", _NATIVE_CLASSES)
def test_hotpath_reverse_region_is_native(monkeypatch, class_name):
    case = _build_native_case(class_name, seed=_seed_for(class_name, 1000))
    _solver, _fs, _adj, _ex, _ey, _ez, native_label = case

    monkeypatch.setenv(_ENV, "native")
    trace = _DispatchTrace()
    with trace:
        result = _run_reverse(monkeypatch, "native", case)
    torch.cuda.synchronize()

    # (a) The step selected the matching NATIVE_* backend.
    assert result.backend == native_label, f"{class_name}: backend {result.backend!r} != {native_label!r}"

    # (b) The fused reverse-math kernels actually ran on native CUDA.
    native_ops = trace.native_cuda_ops()
    assert native_ops, f"{class_name}: no native CUDA reverse kernels dispatched in the reverse region"

    # (c) Every Torch aten op in the reverse region is a whitelisted
    #     orchestration / replay op -- the reverse math itself is never an aten op.
    offending = trace.aten_ops() - _REVERSE_REGION_ATEN_WHITELIST
    assert not offending, (
        f"{class_name}: non-whitelisted aten ops in the native reverse region: {sorted(offending)}. "
        f"Native reverse-math ops present: {sorted(native_ops)}."
    )


@_CUDA
@pytest.mark.parametrize("class_name", ["standard", "cpml", "kerr", "dispersive_cpml", "tfsf", "bloch_dispersive"])
def test_torch_reference_reverse_region_leaves_the_whitelist(monkeypatch, class_name):
    """Calibration guard: the analytic Torch reference reverse trips the whitelist.

    Running the same reverse region under ``torch_reference`` executes the reverse
    math in Torch, which invokes zero native reverse kernels and dispatches aten
    ops outside the calibrated whitelist. This proves the whitelist is tight: it is
    not a trivially-permissive set that a Torch regression could satisfy.
    """
    case = _build_native_case(class_name, seed=_seed_for(class_name, 2000))

    trace = _DispatchTrace()
    with trace:
        result = _run_reverse(monkeypatch, "torch_reference", case)
    torch.cuda.synchronize()

    assert result.backend.startswith("python_reference_"), result.backend
    # The analytic reference runs no fused native reverse-math kernel.
    assert not trace.native_cuda_ops(), (
        f"{class_name}: torch_reference unexpectedly dispatched native CUDA ops {sorted(trace.native_cuda_ops())}"
    )
    # And it leaves the whitelist (allocation / masking ops the native path avoids).
    escaped = trace.aten_ops() - _REVERSE_REGION_ATEN_WHITELIST
    assert escaped, (
        f"{class_name}: torch_reference reverse produced no aten op outside the whitelist; "
        "the whitelist is too permissive to catch a native->Torch regression."
    )


# ---------------------------------------------------------------------------
# 2. Step-level native==reference parity matrix.
# ---------------------------------------------------------------------------


def _l2_relative(actual, expected):
    denom = expected.norm().clamp_min(1e-30)
    return ((actual - expected).norm() / denom).item()


@_CUDA
@pytest.mark.parametrize("class_name", _NATIVE_CLASSES)
def test_native_reverse_matches_reference(monkeypatch, class_name):
    """The fused native reverse reproduces the analytic Torch reference per class.

    Scale-invariant L2-relative parity across every checkpoint-field adjoint and
    the eps (and, for Kerr, chi3) gradient. Analytic (fake-solver) classes hold to
    ~1e-5; the physical Bloch+dispersive checkpoint holds to 1e-4 (the fused native
    accumulation orders the same terms differently in FP, so isolated boundary
    edges where large base/correction contributions cancel carry per-cell error
    while the field/gradient as a whole is bit-close).
    """
    case = _build_native_case(class_name, seed=_seed_for(class_name, 3000))
    forward_state = case[1]
    tol = 1e-4 if class_name == "bloch_dispersive" else 2e-5

    native = _run_reverse(monkeypatch, "native", case)
    reference = _run_reverse(monkeypatch, "torch_reference", case)
    torch.cuda.synchronize()

    assert native.backend == case[-1]
    assert reference.backend.startswith("python_reference_")

    for name in forward_state:
        rel = _l2_relative(native.pre_step_adjoint[name], reference.pre_step_adjoint[name])
        assert rel <= tol, f"{class_name}: pre_step_adjoint[{name}] L2-rel {rel:.3e} > {tol:.1e}"
    for grad_name in ("grad_eps_ex", "grad_eps_ey", "grad_eps_ez"):
        rel = _l2_relative(getattr(native, grad_name), getattr(reference, grad_name))
        assert rel <= tol, f"{class_name}: {grad_name} L2-rel {rel:.3e} > {tol:.1e}"
    if class_name == "kerr":
        for grad_name in ("grad_chi3_ex", "grad_chi3_ey", "grad_chi3_ez"):
            rel = _l2_relative(getattr(native, grad_name), getattr(reference, grad_name))
            assert rel <= tol, f"{class_name}: {grad_name} L2-rel {rel:.3e} > {tol:.1e}"


# ---------------------------------------------------------------------------
# 3. Capability-matrix finite-difference extensions.
# ---------------------------------------------------------------------------


def _abs2(z):
    return (z * z.conj()).real if z.is_complex() else z * z


_FD_DELTA = 1.0e-2


def _central_difference(model, loss_fn, *, delta=_FD_DELTA):
    logits = model.logits
    flat = logits.detach().flatten()
    fd_grad = torch.zeros_like(flat)
    for i in range(flat.numel()):
        with torch.no_grad():
            saved = flat[i].clone()
            flat[i] = saved + delta
            logits.copy_(flat.reshape(logits.shape))
        loss_plus = loss_fn()
        with torch.no_grad():
            flat[i] = saved - delta
            logits.copy_(flat.reshape(logits.shape))
        loss_minus = loss_fn()
        fd_grad[i] = (loss_plus - loss_minus) / (2.0 * delta)
        with torch.no_grad():
            flat[i] = saved
            logits.copy_(flat.reshape(logits.shape))
    return fd_grad.reshape(logits.shape)


_SIGMA_M = 6.0e3


def _sigma_m_scene(sigma_m, *, density=None):
    """Design region embedded in a static magnetically-conductive (sigma_m) slab.

    ``sigma_m`` folds an additive magnetic conduction current ``sigma_m * H`` into
    Faraday's law, discretized semi-implicitly into the magnetic decay/curl pair
    the native standard/CPML reverse already consumes (it carries no separate ADE
    state, so the checkpoint schema is the plain six-field layout). The eps design
    gradient must stay correct through the sigma_m-modified magnetic decay.
    """
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
        grid=mw.GridSpec.uniform(0.12),
        boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
        device="cuda",
    )
    if sigma_m != 0.0:
        scene.add_structure(
            mw.Structure(
                name="magnetically_lossy_slab",
                geometry=mw.Box(position=(0.06, 0.06, 0.18), size=(0.30, 0.30, 0.30)),
                material=mw.Material(eps_r=1.0, sigma_m=sigma_m),
            )
        )
    if density is not None:
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(0.06, 0.06, 0.18), size=(0.24, 0.24, 0.24)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, -0.18),
            polarization="Ez",
            width=0.04,
            source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=50.0),
        )
    )
    scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.18), fields=("Ez",)))
    return scene


class _SigmaMDensityScene(mw.SceneModule):
    def __init__(self, init=0.0, sigma_m=_SIGMA_M):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full((2, 2, 2), float(init), device="cuda"))
        self._sigma_m = sigma_m

    def to_scene(self):
        return _sigma_m_scene(self._sigma_m, density=torch.sigmoid(self.logits))


@_CUDA
def test_additive_sigma_m_gradient_matches_fd():
    """Per-element FD validation of design gradients in an additive magnetic-loss
    (sigma_m) medium: the eps design gradient must be correct through the
    sigma_m-modified native magnetic decay coefficient."""
    model = _SigmaMDensityScene(init=0.0)

    def loss_fn():
        result = mw.Simulation.fdtd(
            model, frequencies=[1e9], run_time=mw.TimeConfig(time_steps=200)
        ).run()
        return _abs2(result.monitor("probe")["data"])

    # The magnetic loss must actually shape the solution, otherwise this only
    # re-validates the lossless path. A realistic sigma_m damps the probe several-fold.
    lossless = _abs2(
        mw.Simulation.fdtd(
            _SigmaMDensityScene(init=0.0, sigma_m=0.0), frequencies=[1e9], run_time=mw.TimeConfig(time_steps=200)
        )
        .run()
        .monitor("probe")["data"]
    ).item()
    lossy = loss_fn().item()
    assert lossy < 0.6 * lossless, f"sigma_m term inactive: lossless={lossless:.6e}, lossy={lossy:.6e}"

    loss = loss_fn()
    loss.backward()
    backward_grad = model.logits.grad.detach().clone()
    model.logits.grad = None
    fd_grad = _central_difference(model, loss_fn)

    assert (fd_grad != 0).any(), "FD gradient is identically zero; test setup is broken."
    dominant = int(torch.abs(fd_grad).flatten().argmax().item())
    dom_rel = abs(
        backward_grad.flatten()[dominant].item() - fd_grad.flatten()[dominant].item()
    ) / abs(fd_grad.flatten()[dominant].item())
    assert dom_rel < 5e-3, f"dominant-element rel err {dom_rel:.3e} exceeds 5e-3."

    scale = torch.abs(fd_grad).max().item()
    assert torch.allclose(backward_grad, fd_grad, rtol=3e-2, atol=scale * 5e-3), (
        f"backward={backward_grad.flatten().tolist()}, fd={fd_grad.flatten().tolist()}"
    )


class _MultiFreqDispersiveScene(mw.SceneModule):
    """Trainable design density inside a Debye host, probed at two frequencies.

    Exercises the native electric-dispersive reverse together with the
    multi-frequency running-DFT seed path: the backward sweep must fold two
    per-frequency monitor cotangents through the same native dispersive reverse.
    """

    def __init__(self, init=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full((1, 1, 1), float(init), device="cuda"))

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.24, 0.24), (-0.24, 0.24), (-0.24, 0.24))),
            grid=mw.GridSpec.uniform(0.12),
            boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
            device="cuda",
        )
        scene.add_structure(
            mw.Structure(
                name="debye_host",
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.24, 0.24, 0.24)),
                material=mw.Material.debye(eps_inf=1.5, delta_eps=2.0, tau=2.0e-10),
            )
        )
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(0.0, 0.0, 0.06), size=(0.12, 0.12, 0.12)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(0.0, 0.0, -0.06),
                polarization="Ez",
                width=0.04,
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.5e9, amplitude=50.0),
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.06), fields=("Ez",)))
        return scene


@_CUDA
def test_multi_frequency_dispersive_gradient_matches_fd():
    """Per-element FD validation of a multi-frequency objective over an
    electric-dispersive design: the native dispersive reverse must correctly fold
    both per-frequency monitor cotangents into one design gradient."""
    model = _MultiFreqDispersiveScene(init=0.0)
    freqs = [0.8e9, 1.5e9]

    def loss_fn():
        result = mw.Simulation.fdtd(
            model, frequencies=freqs, run_time=mw.TimeConfig(time_steps=120)
        ).run()
        d0 = result.monitor("probe", freq_index=0)["data"]
        d1 = result.monitor("probe", freq_index=1)["data"]
        return _abs2(d0) + _abs2(d1)

    loss = loss_fn()
    loss.backward()
    backward_grad = model.logits.grad.detach().clone()
    model.logits.grad = None
    fd_grad = _central_difference(model, loss_fn)

    assert (fd_grad != 0).any(), "FD gradient is identically zero; test setup is broken."
    scale = torch.abs(fd_grad).max().item()
    assert torch.allclose(backward_grad, fd_grad, rtol=3e-2, atol=scale * 5e-3), (
        f"backward={backward_grad.flatten().tolist()}, fd={fd_grad.flatten().tolist()}"
    )


# ---------------------------------------------------------------------------
# 4. Coverage map: the honestly-reported torch-VJP / analytic remainder classes.
#
# These differentiable classes have no fused native reverse runner yet, so the
# gate above does not cover them. They are pinned here (and in the capability
# matrix doc) so the boundary between "native" and "remainder" stays explicit.
# ---------------------------------------------------------------------------


def _has_reverse_step_fn(name: str) -> bool:
    return hasattr(adjoint_reference, name)


@_CUDA
def test_general_nonlinear_reverse_remains_torch_vjp():
    """chi2 / two-photon-absorption (general nonlinear) has no analytic native
    reverse; it stays on the torch-VJP fallback (its dynamic decay+curl rewrite is
    not a fused kernel). Pinned as the documented remainder."""
    from witwin.maxwell.fdtd.adjoint.dispatch import _ReverseBackend, _select_reverse_backend
    from tests.gradients.test_fdtd_adjoint_materials import _general_nonlinear_scene

    solver = (
        mw.Simulation.fdtd(_general_nonlinear_scene(), frequencies=[1e9], run_time=mw.TimeConfig(time_steps=8))
        .prepare()
        .solver
    )
    assert solver.nonlinear_general_enabled
    forward_state = {name: getattr(solver, name) for name in checkpoint_schema(solver).state_names}
    backend = _select_reverse_backend(
        solver,
        forward_state,
        eps_ex=solver.eps_Ex,
        eps_ey=solver.eps_Ey,
        eps_ez=solver.eps_Ez,
        resolved_source_terms=None,
    )
    assert backend is _ReverseBackend.TORCH_VJP, backend
