from __future__ import annotations

import math
import os
import weakref
from collections.abc import Callable
from typing import Any

import torch

from ..boundary.common import (
    BOUNDARY_NONE,
    BOUNDARY_PEC,
    BOUNDARY_PERIODIC,
    BOUNDARY_PMC,
    BOUNDARY_PML,
)


_COMPILED_EXTENSION: Any | None = None


def is_available() -> bool:
    return bool(torch.cuda.is_available())


def get_compiled_extension(*, verbose: bool = False) -> Any:
    global _COMPILED_EXTENSION
    if _COMPILED_EXTENSION is None:
        from .build import build_extension

        _COMPILED_EXTENSION = build_extension(verbose=verbose)
    return _COMPILED_EXTENSION


def _use_compiled_field_kernels() -> bool:
    return os.environ.get("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION") == "1"


_UNIFORM_SCALAR_CACHE: dict[int, tuple[Any, int, int, float | None]] = {}


def _uniform_scalar(tensor: torch.Tensor) -> float | None:
    """Return the scalar value of a spatially uniform coefficient tensor.

    Update coefficients (decay/curl) are static for the duration of a solve,
    so the min==max reduction and its device synchronization run once per
    tensor and the result is cached. Cache entries are validated against the
    tensor identity (weakref), storage pointer, and autograd version counter;
    kernels that mutate a coefficient tensor in place without bumping the
    version counter (the Kerr dynamic-curl update) must call
    _invalidate_uniform_scalar explicitly.
    """
    key = id(tensor)
    entry = _UNIFORM_SCALAR_CACHE.get(key)
    if entry is not None and entry[0]() is tensor:
        if entry[1] is None:  # pinned non-uniform (mutated in place by kernels)
            return None
        if entry[1] == tensor.data_ptr() and entry[2] == tensor._version:
            return entry[3]
    minimum, maximum = torch.aminmax(tensor)
    value = minimum.item() if bool((minimum == maximum).item()) else None
    _UNIFORM_SCALAR_CACHE[key] = (weakref.ref(tensor), tensor.data_ptr(), tensor._version, value)
    return value


def _invalidate_uniform_scalar(tensor: torch.Tensor) -> None:
    # Pin as non-uniform: the tensor is rewritten in place by a kernel every
    # step, so re-deriving a scalar (with its device sync) would be wasted.
    _UNIFORM_SCALAR_CACHE[id(tensor)] = (weakref.ref(tensor), None, None, None)


def build_info() -> dict[str, Any]:
    return {
        "backend": "torch-cuda",
        "compiled_extension_loaded": _COMPILED_EXTENSION is not None,
        "cuda_available": is_available(),
        "torch_version": torch.__version__,
    }


def _require_cuda_tensors(*values: Any) -> None:
    for value in values:
        if torch.is_tensor(value) and value.device.type != "cuda":
            raise ValueError("Native CUDA FDTD backend requires CUDA tensors.")


def _axis_view(values: torch.Tensor, axis: int, ndim: int = 3) -> torch.Tensor:
    shape = [1] * ndim
    shape[axis] = int(values.shape[0])
    return values.reshape(shape)


def debug_linear_indices(
    shape: tuple[int, int, int],
    *,
    device: str | torch.device | None = None,
    use_extension: bool = False,
):
    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda":
        raise ValueError("debug_linear_indices requires a CUDA device.")
    if use_extension:
        return get_compiled_extension().debug_linear_indices([int(value) for value in shape])
    total = int(shape[0]) * int(shape[1]) * int(shape[2])
    linear = torch.arange(total, device=device, dtype=torch.int64).reshape(shape)
    i_index = linear // (int(shape[1]) * int(shape[2]))
    j_index = (linear // int(shape[2])) % int(shape[1])
    k_index = linear % int(shape[2])
    return linear, i_index, j_index, k_index


def _call_with_contiguous(kernel: Callable[..., None], kwargs: dict) -> None:
    # The slangtorch module surface accepts strided tensor views (the adjoint
    # replay passes box slices of adjoint fields and real/imag views), while
    # the compiled kernels require contiguous memory. Materialize contiguous
    # copies and write back afterwards so in-place semantics are preserved.
    write_back = None
    for key, value in kwargs.items():
        if type(value) is torch.Tensor and not value.is_contiguous():
            contiguous = value.contiguous()
            kwargs[key] = contiguous
            if write_back is None:
                write_back = []
            write_back.append((value, contiguous))
    kernel(**kwargs)
    if write_back is not None:
        for original, contiguous in write_back:
            original.copy_(contiguous)


class _Launch:
    __slots__ = ("kernel", "kwargs")

    def __init__(self, kernel: Callable[..., None], kwargs: dict):
        self.kernel = kernel
        self.kwargs = kwargs

    def launchRaw(self, *, blockSize=None, gridSize=None):  # noqa: N802 - mirrors slangtorch API
        del blockSize, gridSize
        _call_with_contiguous(self.kernel, self.kwargs)
        return None


class NativeFDTDModule:
    def __getattr__(self, name: str):
        kernel = _KERNELS.get(name)
        if kernel is None:
            raise AttributeError(f"Native CUDA FDTD backend does not implement kernel {name!r}.")

        def bind(**kwargs):
            return _Launch(kernel, kwargs)

        bind.__name__ = name
        # Cache on the instance so later lookups bypass __getattr__ entirely.
        object.__setattr__(self, name, bind)
        return bind


_NATIVE_MODULE = NativeFDTDModule()


def get_native_fdtd_module() -> NativeFDTDModule:
    if not is_available():
        raise RuntimeError("Native CUDA FDTD backend requires torch.cuda.is_available() to be True.")
    return _NATIVE_MODULE


def _boundary_mode_vector(size: int, low_mode: int, high_mode: int, *, device) -> tuple[torch.Tensor, torch.Tensor]:
    boundary = torch.zeros(size, device=device, dtype=torch.bool)
    mode = torch.full((size,), BOUNDARY_NONE, device=device, dtype=torch.int64)
    if size > 0:
        boundary[0] = True
        mode[0] = int(low_mode)
        if size > 1:
            boundary[-1] = True
            mode[-1] = int(high_mode)
    return boundary, mode


def _boundary_masks(
    shape: tuple[int, int, int],
    specs: tuple[tuple[int, int, int], ...],
    *,
    device,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    pec = torch.zeros(shape, device=device, dtype=torch.bool)
    inactive = torch.zeros(shape, device=device, dtype=torch.bool)
    axis_data: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for axis, low_mode, high_mode in specs:
        boundary, mode = _boundary_mode_vector(shape[axis], low_mode, high_mode, device=device)
        view = _axis_view(boundary, axis).expand(shape)
        mode_view = _axis_view(mode, axis).expand(shape)
        axis_data[axis] = (boundary, mode)
        pec |= view & (mode_view == BOUNDARY_PEC)
        inactive |= view & ((mode_view == BOUNDARY_NONE) | (mode_view == BOUNDARY_PML))
    return pec, inactive, axis_data


def _backward_diff(
    source: torch.Tensor,
    target_shape: tuple[int, int, int],
    axis: int,
    low_mode: int,
    high_mode: int,
    inv_delta: float,
) -> torch.Tensor:
    device = source.device
    result = torch.zeros(target_shape, device=device, dtype=source.dtype)
    size = target_shape[axis]
    if size <= 0:
        return result

    if size > 2:
        dst = [slice(None)] * 3
        dst[axis] = slice(1, size - 1)
        hi = [slice(None)] * 3
        hi[axis] = slice(1, size - 1)
        lo = [slice(None)] * 3
        lo[axis] = slice(0, size - 2)
        result[tuple(dst)] = (source[tuple(hi)] - source[tuple(lo)]) * float(inv_delta)

    low_src = [slice(None)] * 3
    low_src[axis] = 0
    high_src = [slice(None)] * 3
    high_src[axis] = source.shape[axis] - 1

    low_dst = [slice(None)] * 3
    low_dst[axis] = 0
    high_dst = [slice(None)] * 3
    high_dst[axis] = size - 1

    low_value = source[tuple(low_src)]
    high_value = source[tuple(high_src)]
    if low_mode == BOUNDARY_PERIODIC:
        result[tuple(low_dst)] = (low_value - high_value) * float(inv_delta)
    elif low_mode == BOUNDARY_PMC:
        result[tuple(low_dst)] = 2.0 * low_value * float(inv_delta)

    if size > 1:
        if high_mode == BOUNDARY_PERIODIC:
            result[tuple(high_dst)] = (low_value - high_value) * float(inv_delta)
        elif high_mode == BOUNDARY_PMC:
            result[tuple(high_dst)] = -2.0 * high_value * float(inv_delta)
    return result


def _compressed_psi_update(
    psi: torch.Tensor,
    derivative: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    axis: int,
    low_length: int,
    high_start: int,
    high_length: int,
    update_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    full = torch.zeros_like(derivative)

    def update_region(global_start: int, length: int, local_start: int) -> None:
        if length <= 0:
            return
        global_slice = [slice(None)] * 3
        global_slice[axis] = slice(global_start, global_start + length)
        local_slice = [slice(None)] * 3
        local_slice[axis] = slice(local_start, local_start + length)
        coeff_slice = slice(global_start, global_start + length)
        updated = (
            _axis_view(b[coeff_slice], axis) * psi[tuple(local_slice)]
            + _axis_view(c[coeff_slice], axis) * derivative[tuple(global_slice)]
        )
        if update_mask is not None:
            region_mask = update_mask[tuple(global_slice)]
            updated = torch.where(region_mask, updated, psi[tuple(local_slice)])
        psi[tuple(local_slice)].copy_(updated)
        full[tuple(global_slice)].copy_(updated)

    update_region(0, int(low_length), 0)
    update_region(int(high_start), int(high_length), int(low_length))
    return full


def _magnetic_hx_standard(*, Hx, Ey, Ez, HxDecay, HxCurl, invDy, invDz):
    _require_cuda_tensors(Hx, Ey, Ez, HxDecay, HxCurl)
    if _use_compiled_field_kernels():
        get_compiled_extension().update_magnetic_hx_standard(Hx, Ey, Ez, HxDecay, HxCurl, float(invDy), float(invDz))
        return
    curl = (Ez[:, 1:, :] - Ez[:, :-1, :]) * float(invDy) - (Ey[:, :, 1:] - Ey[:, :, :-1]) * float(invDz)
    Hx.copy_(Hx * HxDecay - HxCurl * curl)


def _magnetic_hy_standard(*, Hy, Ex, Ez, HyDecay, HyCurl, invDx, invDz):
    _require_cuda_tensors(Hy, Ex, Ez, HyDecay, HyCurl)
    if _use_compiled_field_kernels():
        get_compiled_extension().update_magnetic_hy_standard(Hy, Ex, Ez, HyDecay, HyCurl, float(invDx), float(invDz))
        return
    curl = (Ex[:, :, 1:] - Ex[:, :, :-1]) * float(invDz) - (Ez[1:, :, :] - Ez[:-1, :, :]) * float(invDx)
    Hy.copy_(Hy * HyDecay - HyCurl * curl)


def _magnetic_hz_standard(*, Hz, Ex, Ey, HzDecay, HzCurl, invDx, invDy):
    _require_cuda_tensors(Hz, Ex, Ey, HzDecay, HzCurl)
    if _use_compiled_field_kernels():
        get_compiled_extension().update_magnetic_hz_standard(Hz, Ex, Ey, HzDecay, HzCurl, float(invDx), float(invDy))
        return
    curl = (Ey[1:, :, :] - Ey[:-1, :, :]) * float(invDx) - (Ex[:, 1:, :] - Ex[:, :-1, :]) * float(invDy)
    Hz.copy_(Hz * HzDecay - HzCurl * curl)


def _magnetic_hx_cpml(
    *,
    Hx,
    Ey,
    Ez,
    HxDecay,
    HxCurl,
    PsiHxY,
    PsiHxZ,
    InvKappaHxY,
    ByHxY,
    CyHxY,
    InvKappaHxZ,
    ByHxZ,
    CyHxZ,
    invDy,
    invDz,
):
    _require_cuda_tensors(
        Hx,
        Ey,
        Ez,
        HxDecay,
        HxCurl,
        PsiHxY,
        PsiHxZ,
        InvKappaHxY,
        ByHxY,
        CyHxY,
        InvKappaHxZ,
        ByHxZ,
        CyHxZ,
    )
    if _use_compiled_field_kernels():
        get_compiled_extension().update_magnetic_hx_cpml(
            Hx,
            Ey,
            Ez,
            HxDecay,
            HxCurl,
            PsiHxY,
            PsiHxZ,
            InvKappaHxY,
            ByHxY,
            CyHxY,
            InvKappaHxZ,
            ByHxZ,
            CyHxZ,
            float(invDy),
            float(invDz),
        )
        return
    d_y = (Ez[:, 1:, :] - Ez[:, :-1, :]) * float(invDy)
    d_z = (Ey[:, :, 1:] - Ey[:, :, :-1]) * float(invDz)
    psi_y = _axis_view(ByHxY, 1) * PsiHxY + _axis_view(CyHxY, 1) * d_y
    psi_z = _axis_view(ByHxZ, 2) * PsiHxZ + _axis_view(CyHxZ, 2) * d_z
    PsiHxY.copy_(psi_y)
    PsiHxZ.copy_(psi_z)
    curl = d_y * _axis_view(InvKappaHxY, 1) + psi_y - d_z * _axis_view(InvKappaHxZ, 2) - psi_z
    Hx.copy_(Hx * HxDecay - HxCurl * curl)


def _magnetic_hy_cpml(
    *,
    Hy,
    Ex,
    Ez,
    HyDecay,
    HyCurl,
    PsiHyX,
    PsiHyZ,
    InvKappaHyX,
    ByHyX,
    CyHyX,
    InvKappaHyZ,
    ByHyZ,
    CyHyZ,
    invDx,
    invDz,
):
    _require_cuda_tensors(
        Hy,
        Ex,
        Ez,
        HyDecay,
        HyCurl,
        PsiHyX,
        PsiHyZ,
        InvKappaHyX,
        ByHyX,
        CyHyX,
        InvKappaHyZ,
        ByHyZ,
        CyHyZ,
    )
    if _use_compiled_field_kernels():
        get_compiled_extension().update_magnetic_hy_cpml(
            Hy,
            Ex,
            Ez,
            HyDecay,
            HyCurl,
            PsiHyX,
            PsiHyZ,
            InvKappaHyX,
            ByHyX,
            CyHyX,
            InvKappaHyZ,
            ByHyZ,
            CyHyZ,
            float(invDx),
            float(invDz),
        )
        return
    d_z = (Ex[:, :, 1:] - Ex[:, :, :-1]) * float(invDz)
    d_x = (Ez[1:, :, :] - Ez[:-1, :, :]) * float(invDx)
    psi_x = _axis_view(ByHyX, 0) * PsiHyX + _axis_view(CyHyX, 0) * d_x
    psi_z = _axis_view(ByHyZ, 2) * PsiHyZ + _axis_view(CyHyZ, 2) * d_z
    PsiHyX.copy_(psi_x)
    PsiHyZ.copy_(psi_z)
    curl = d_z * _axis_view(InvKappaHyZ, 2) + psi_z - d_x * _axis_view(InvKappaHyX, 0) - psi_x
    Hy.copy_(Hy * HyDecay - HyCurl * curl)


def _magnetic_hz_cpml(
    *,
    Hz,
    Ex,
    Ey,
    HzDecay,
    HzCurl,
    PsiHzX,
    PsiHzY,
    InvKappaHzX,
    ByHzX,
    CyHzX,
    InvKappaHzY,
    ByHzY,
    CyHzY,
    invDx,
    invDy,
):
    _require_cuda_tensors(
        Hz,
        Ex,
        Ey,
        HzDecay,
        HzCurl,
        PsiHzX,
        PsiHzY,
        InvKappaHzX,
        ByHzX,
        CyHzX,
        InvKappaHzY,
        ByHzY,
        CyHzY,
    )
    if _use_compiled_field_kernels():
        get_compiled_extension().update_magnetic_hz_cpml(
            Hz,
            Ex,
            Ey,
            HzDecay,
            HzCurl,
            PsiHzX,
            PsiHzY,
            InvKappaHzX,
            ByHzX,
            CyHzX,
            InvKappaHzY,
            ByHzY,
            CyHzY,
            float(invDx),
            float(invDy),
        )
        return
    d_x = (Ey[1:, :, :] - Ey[:-1, :, :]) * float(invDx)
    d_y = (Ex[:, 1:, :] - Ex[:, :-1, :]) * float(invDy)
    psi_x = _axis_view(ByHzX, 0) * PsiHzX + _axis_view(CyHzX, 0) * d_x
    psi_y = _axis_view(ByHzY, 1) * PsiHzY + _axis_view(CyHzY, 1) * d_y
    PsiHzX.copy_(psi_x)
    PsiHzY.copy_(psi_y)
    curl = d_x * _axis_view(InvKappaHzX, 0) + psi_x - d_y * _axis_view(InvKappaHzY, 1) - psi_y
    Hz.copy_(Hz * HzDecay - HzCurl * curl)


def _magnetic_hx_cpml_compressed(
    *,
    Hx,
    Ey,
    Ez,
    HxDecay,
    HxCurl,
    PsiHxY,
    PsiHxZ,
    InvKappaHxY,
    ByHxY,
    CyHxY,
    InvKappaHxZ,
    ByHxZ,
    CyHxZ,
    invDy,
    invDz,
    psiHxYLowLength,
    psiHxYHighStart,
    psiHxYHighLength,
    psiHxZLowLength,
    psiHxZHighStart,
    psiHxZHighLength,
):
    _require_cuda_tensors(Hx, Ey, Ez, PsiHxY, PsiHxZ)
    if _use_compiled_field_kernels():
        get_compiled_extension().update_magnetic_hx_cpml_compressed(
            Hx,
            Ey,
            Ez,
            HxDecay,
            HxCurl,
            PsiHxY,
            PsiHxZ,
            InvKappaHxY,
            ByHxY,
            CyHxY,
            InvKappaHxZ,
            ByHxZ,
            CyHxZ,
            float(invDy),
            float(invDz),
            int(psiHxYLowLength),
            int(psiHxYHighStart),
            int(psiHxYHighLength),
            int(psiHxZLowLength),
            int(psiHxZHighStart),
            int(psiHxZHighLength),
            _uniform_scalar(HxDecay),
            _uniform_scalar(HxCurl),
        )
        return
    d_y = (Ez[:, 1:, :] - Ez[:, :-1, :]) * float(invDy)
    d_z = (Ey[:, :, 1:] - Ey[:, :, :-1]) * float(invDz)
    psi_y = _compressed_psi_update(
        PsiHxY,
        d_y,
        ByHxY,
        CyHxY,
        axis=1,
        low_length=psiHxYLowLength,
        high_start=psiHxYHighStart,
        high_length=psiHxYHighLength,
    )
    psi_z = _compressed_psi_update(
        PsiHxZ,
        d_z,
        ByHxZ,
        CyHxZ,
        axis=2,
        low_length=psiHxZLowLength,
        high_start=psiHxZHighStart,
        high_length=psiHxZHighLength,
    )
    curl = d_y * _axis_view(InvKappaHxY, 1) + psi_y - d_z * _axis_view(InvKappaHxZ, 2) - psi_z
    Hx.copy_(Hx * HxDecay - HxCurl * curl)


def _magnetic_hy_cpml_compressed(
    *,
    Hy,
    Ex,
    Ez,
    HyDecay,
    HyCurl,
    PsiHyX,
    PsiHyZ,
    InvKappaHyX,
    ByHyX,
    CyHyX,
    InvKappaHyZ,
    ByHyZ,
    CyHyZ,
    invDx,
    invDz,
    psiHyXLowLength,
    psiHyXHighStart,
    psiHyXHighLength,
    psiHyZLowLength,
    psiHyZHighStart,
    psiHyZHighLength,
):
    _require_cuda_tensors(Hy, Ex, Ez, PsiHyX, PsiHyZ)
    if _use_compiled_field_kernels():
        get_compiled_extension().update_magnetic_hy_cpml_compressed(
            Hy,
            Ex,
            Ez,
            HyDecay,
            HyCurl,
            PsiHyX,
            PsiHyZ,
            InvKappaHyX,
            ByHyX,
            CyHyX,
            InvKappaHyZ,
            ByHyZ,
            CyHyZ,
            float(invDx),
            float(invDz),
            int(psiHyXLowLength),
            int(psiHyXHighStart),
            int(psiHyXHighLength),
            int(psiHyZLowLength),
            int(psiHyZHighStart),
            int(psiHyZHighLength),
            _uniform_scalar(HyDecay),
            _uniform_scalar(HyCurl),
        )
        return
    d_z = (Ex[:, :, 1:] - Ex[:, :, :-1]) * float(invDz)
    d_x = (Ez[1:, :, :] - Ez[:-1, :, :]) * float(invDx)
    psi_x = _compressed_psi_update(
        PsiHyX,
        d_x,
        ByHyX,
        CyHyX,
        axis=0,
        low_length=psiHyXLowLength,
        high_start=psiHyXHighStart,
        high_length=psiHyXHighLength,
    )
    psi_z = _compressed_psi_update(
        PsiHyZ,
        d_z,
        ByHyZ,
        CyHyZ,
        axis=2,
        low_length=psiHyZLowLength,
        high_start=psiHyZHighStart,
        high_length=psiHyZHighLength,
    )
    curl = d_z * _axis_view(InvKappaHyZ, 2) + psi_z - d_x * _axis_view(InvKappaHyX, 0) - psi_x
    Hy.copy_(Hy * HyDecay - HyCurl * curl)


def _magnetic_hz_cpml_compressed(
    *,
    Hz,
    Ex,
    Ey,
    HzDecay,
    HzCurl,
    PsiHzX,
    PsiHzY,
    InvKappaHzX,
    ByHzX,
    CyHzX,
    InvKappaHzY,
    ByHzY,
    CyHzY,
    invDx,
    invDy,
    psiHzXLowLength,
    psiHzXHighStart,
    psiHzXHighLength,
    psiHzYLowLength,
    psiHzYHighStart,
    psiHzYHighLength,
):
    _require_cuda_tensors(Hz, Ex, Ey, PsiHzX, PsiHzY)
    if _use_compiled_field_kernels():
        get_compiled_extension().update_magnetic_hz_cpml_compressed(
            Hz,
            Ex,
            Ey,
            HzDecay,
            HzCurl,
            PsiHzX,
            PsiHzY,
            InvKappaHzX,
            ByHzX,
            CyHzX,
            InvKappaHzY,
            ByHzY,
            CyHzY,
            float(invDx),
            float(invDy),
            int(psiHzXLowLength),
            int(psiHzXHighStart),
            int(psiHzXHighLength),
            int(psiHzYLowLength),
            int(psiHzYHighStart),
            int(psiHzYHighLength),
            _uniform_scalar(HzDecay),
            _uniform_scalar(HzCurl),
        )
        return
    d_x = (Ey[1:, :, :] - Ey[:-1, :, :]) * float(invDx)
    d_y = (Ex[:, 1:, :] - Ex[:, :-1, :]) * float(invDy)
    psi_x = _compressed_psi_update(
        PsiHzX,
        d_x,
        ByHzX,
        CyHzX,
        axis=0,
        low_length=psiHzXLowLength,
        high_start=psiHzXHighStart,
        high_length=psiHzXHighLength,
    )
    psi_y = _compressed_psi_update(
        PsiHzY,
        d_y,
        ByHzY,
        CyHzY,
        axis=1,
        low_length=psiHzYLowLength,
        high_start=psiHzYHighStart,
        high_length=psiHzYHighLength,
    )
    curl = d_x * _axis_view(InvKappaHzX, 0) + psi_x - d_y * _axis_view(InvKappaHzY, 1) - psi_y
    Hz.copy_(Hz * HzDecay - HzCurl * curl)


def _electric_standard(
    field: torch.Tensor,
    first_derivative: torch.Tensor,
    second_derivative: torch.Tensor,
    decay: torch.Tensor,
    curl: torch.Tensor,
    specs: tuple[tuple[int, int, int], tuple[int, int, int]],
) -> None:
    pec, inactive, _ = _boundary_masks(tuple(field.shape), specs, device=field.device)
    active = ~(pec | inactive)
    updated = field * decay + curl * (first_derivative - second_derivative)
    field.copy_(torch.where(pec, torch.zeros_like(field), torch.where(active, updated, field)))


def _electric_ex_standard(
    *,
    Ex,
    Hy,
    Hz,
    ExDecay,
    ExCurl,
    invDy,
    invDz,
    yLowBoundaryMode,
    yHighBoundaryMode,
    zLowBoundaryMode,
    zHighBoundaryMode,
):
    _require_cuda_tensors(Ex, Hy, Hz, ExDecay, ExCurl)
    if _use_compiled_field_kernels():
        get_compiled_extension().update_electric_ex_standard(
            Ex,
            Hy,
            Hz,
            ExDecay,
            ExCurl,
            float(invDy),
            float(invDz),
            int(yLowBoundaryMode),
            int(yHighBoundaryMode),
            int(zLowBoundaryMode),
            int(zHighBoundaryMode),
        )
        return
    d_y = _backward_diff(Hz, tuple(Ex.shape), 1, yLowBoundaryMode, yHighBoundaryMode, invDy)
    d_z = _backward_diff(Hy, tuple(Ex.shape), 2, zLowBoundaryMode, zHighBoundaryMode, invDz)
    _electric_standard(
        Ex,
        d_y,
        d_z,
        ExDecay,
        ExCurl,
        ((1, yLowBoundaryMode, yHighBoundaryMode), (2, zLowBoundaryMode, zHighBoundaryMode)),
    )


def _electric_ey_standard(
    *,
    Ey,
    Hx,
    Hz,
    EyDecay,
    EyCurl,
    invDx,
    invDz,
    xLowBoundaryMode,
    xHighBoundaryMode,
    zLowBoundaryMode,
    zHighBoundaryMode,
):
    _require_cuda_tensors(Ey, Hx, Hz, EyDecay, EyCurl)
    if _use_compiled_field_kernels():
        get_compiled_extension().update_electric_ey_standard(
            Ey,
            Hx,
            Hz,
            EyDecay,
            EyCurl,
            float(invDx),
            float(invDz),
            int(xLowBoundaryMode),
            int(xHighBoundaryMode),
            int(zLowBoundaryMode),
            int(zHighBoundaryMode),
        )
        return
    d_z = _backward_diff(Hx, tuple(Ey.shape), 2, zLowBoundaryMode, zHighBoundaryMode, invDz)
    d_x = _backward_diff(Hz, tuple(Ey.shape), 0, xLowBoundaryMode, xHighBoundaryMode, invDx)
    _electric_standard(
        Ey,
        d_z,
        d_x,
        EyDecay,
        EyCurl,
        ((0, xLowBoundaryMode, xHighBoundaryMode), (2, zLowBoundaryMode, zHighBoundaryMode)),
    )


def _electric_ez_standard(
    *,
    Ez,
    Hx,
    Hy,
    EzDecay,
    EzCurl,
    invDx,
    invDy,
    xLowBoundaryMode,
    xHighBoundaryMode,
    yLowBoundaryMode,
    yHighBoundaryMode,
):
    _require_cuda_tensors(Ez, Hx, Hy, EzDecay, EzCurl)
    if _use_compiled_field_kernels():
        get_compiled_extension().update_electric_ez_standard(
            Ez,
            Hx,
            Hy,
            EzDecay,
            EzCurl,
            float(invDx),
            float(invDy),
            int(xLowBoundaryMode),
            int(xHighBoundaryMode),
            int(yLowBoundaryMode),
            int(yHighBoundaryMode),
        )
        return
    d_x = _backward_diff(Hy, tuple(Ez.shape), 0, xLowBoundaryMode, xHighBoundaryMode, invDx)
    d_y = _backward_diff(Hx, tuple(Ez.shape), 1, yLowBoundaryMode, yHighBoundaryMode, invDy)
    _electric_standard(
        Ez,
        d_x,
        d_y,
        EzDecay,
        EzCurl,
        ((0, xLowBoundaryMode, xHighBoundaryMode), (1, yLowBoundaryMode, yHighBoundaryMode)),
    )


def _electric_cpml(
    field: torch.Tensor,
    first_derivative: torch.Tensor,
    second_derivative: torch.Tensor,
    decay: torch.Tensor,
    curl: torch.Tensor,
    first_psi: torch.Tensor,
    second_psi: torch.Tensor,
    first_inv_kappa: torch.Tensor,
    second_inv_kappa: torch.Tensor,
    first_b: torch.Tensor,
    second_b: torch.Tensor,
    first_c: torch.Tensor,
    second_c: torch.Tensor,
    *,
    first_axis: int,
    second_axis: int,
    specs: tuple[tuple[int, int, int], tuple[int, int, int]],
) -> None:
    pec, inactive, _ = _boundary_masks(tuple(field.shape), specs, device=field.device)
    active = ~(pec | inactive)
    psi_first = _axis_view(first_b, first_axis) * first_psi + _axis_view(first_c, first_axis) * first_derivative
    psi_second = _axis_view(second_b, second_axis) * second_psi + _axis_view(second_c, second_axis) * second_derivative
    first_psi.copy_(torch.where(active, psi_first, first_psi))
    second_psi.copy_(torch.where(active, psi_second, second_psi))
    curl_value = (
        first_derivative * _axis_view(first_inv_kappa, first_axis)
        + psi_first
        - second_derivative * _axis_view(second_inv_kappa, second_axis)
        - psi_second
    )
    updated = field * decay + curl * curl_value
    field.copy_(torch.where(pec, torch.zeros_like(field), torch.where(active, updated, field)))


def _electric_cpml_compressed(
    field: torch.Tensor,
    first_derivative: torch.Tensor,
    second_derivative: torch.Tensor,
    decay: torch.Tensor,
    curl: torch.Tensor,
    first_psi: torch.Tensor,
    second_psi: torch.Tensor,
    first_inv_kappa: torch.Tensor,
    second_inv_kappa: torch.Tensor,
    first_b: torch.Tensor,
    second_b: torch.Tensor,
    first_c: torch.Tensor,
    second_c: torch.Tensor,
    *,
    first_axis: int,
    second_axis: int,
    first_low_length: int,
    first_high_start: int,
    first_high_length: int,
    second_low_length: int,
    second_high_start: int,
    second_high_length: int,
    specs: tuple[tuple[int, int, int], tuple[int, int, int]],
) -> None:
    pec, inactive, _ = _boundary_masks(tuple(field.shape), specs, device=field.device)
    active = ~(pec | inactive)
    masked_first = torch.where(active, first_derivative, torch.zeros_like(first_derivative))
    masked_second = torch.where(active, second_derivative, torch.zeros_like(second_derivative))
    psi_first = _compressed_psi_update(
        first_psi,
        masked_first,
        first_b,
        first_c,
        axis=first_axis,
        low_length=first_low_length,
        high_start=first_high_start,
        high_length=first_high_length,
        update_mask=active,
    )
    psi_second = _compressed_psi_update(
        second_psi,
        masked_second,
        second_b,
        second_c,
        axis=second_axis,
        low_length=second_low_length,
        high_start=second_high_start,
        high_length=second_high_length,
        update_mask=active,
    )
    curl_value = (
        first_derivative * _axis_view(first_inv_kappa, first_axis)
        + psi_first
        - second_derivative * _axis_view(second_inv_kappa, second_axis)
        - psi_second
    )
    updated = field * decay + curl * curl_value
    field.copy_(torch.where(pec, torch.zeros_like(field), torch.where(active, updated, field)))


def _electric_ex_cpml(
    *,
    Ex,
    Hy,
    Hz,
    ExDecay,
    ExCurl,
    PsiExY,
    PsiExZ,
    InvKappaExY,
    BExY,
    CExY,
    InvKappaExZ,
    BExZ,
    CExZ,
    invDy,
    invDz,
    yLowBoundaryMode,
    yHighBoundaryMode,
    zLowBoundaryMode,
    zHighBoundaryMode,
):
    _require_cuda_tensors(
        Ex,
        Hy,
        Hz,
        ExDecay,
        ExCurl,
        PsiExY,
        PsiExZ,
        InvKappaExY,
        BExY,
        CExY,
        InvKappaExZ,
        BExZ,
        CExZ,
    )
    if _use_compiled_field_kernels():
        get_compiled_extension().update_electric_ex_cpml(
            Ex,
            Hy,
            Hz,
            ExDecay,
            ExCurl,
            PsiExY,
            PsiExZ,
            InvKappaExY,
            BExY,
            CExY,
            InvKappaExZ,
            BExZ,
            CExZ,
            float(invDy),
            float(invDz),
            int(yLowBoundaryMode),
            int(yHighBoundaryMode),
            int(zLowBoundaryMode),
            int(zHighBoundaryMode),
        )
        return
    d_y = _backward_diff(Hz, tuple(Ex.shape), 1, yLowBoundaryMode, yHighBoundaryMode, invDy)
    d_z = _backward_diff(Hy, tuple(Ex.shape), 2, zLowBoundaryMode, zHighBoundaryMode, invDz)
    _electric_cpml(
        Ex,
        d_y,
        d_z,
        ExDecay,
        ExCurl,
        PsiExY,
        PsiExZ,
        InvKappaExY,
        InvKappaExZ,
        BExY,
        BExZ,
        CExY,
        CExZ,
        first_axis=1,
        second_axis=2,
        specs=((1, yLowBoundaryMode, yHighBoundaryMode), (2, zLowBoundaryMode, zHighBoundaryMode)),
    )


def _electric_ey_cpml(
    *,
    Ey,
    Hx,
    Hz,
    EyDecay,
    EyCurl,
    PsiEyX,
    PsiEyZ,
    InvKappaEyX,
    BEyX,
    CEyX,
    InvKappaEyZ,
    BEyZ,
    CEyZ,
    invDx,
    invDz,
    xLowBoundaryMode,
    xHighBoundaryMode,
    zLowBoundaryMode,
    zHighBoundaryMode,
):
    _require_cuda_tensors(
        Ey,
        Hx,
        Hz,
        EyDecay,
        EyCurl,
        PsiEyX,
        PsiEyZ,
        InvKappaEyX,
        BEyX,
        CEyX,
        InvKappaEyZ,
        BEyZ,
        CEyZ,
    )
    if _use_compiled_field_kernels():
        get_compiled_extension().update_electric_ey_cpml(
            Ey,
            Hx,
            Hz,
            EyDecay,
            EyCurl,
            PsiEyX,
            PsiEyZ,
            InvKappaEyX,
            BEyX,
            CEyX,
            InvKappaEyZ,
            BEyZ,
            CEyZ,
            float(invDx),
            float(invDz),
            int(xLowBoundaryMode),
            int(xHighBoundaryMode),
            int(zLowBoundaryMode),
            int(zHighBoundaryMode),
        )
        return
    d_z = _backward_diff(Hx, tuple(Ey.shape), 2, zLowBoundaryMode, zHighBoundaryMode, invDz)
    d_x = _backward_diff(Hz, tuple(Ey.shape), 0, xLowBoundaryMode, xHighBoundaryMode, invDx)
    _electric_cpml(
        Ey,
        d_z,
        d_x,
        EyDecay,
        EyCurl,
        PsiEyZ,
        PsiEyX,
        InvKappaEyZ,
        InvKappaEyX,
        BEyZ,
        BEyX,
        CEyZ,
        CEyX,
        first_axis=2,
        second_axis=0,
        specs=((0, xLowBoundaryMode, xHighBoundaryMode), (2, zLowBoundaryMode, zHighBoundaryMode)),
    )


def _electric_ez_cpml(
    *,
    Ez,
    Hx,
    Hy,
    EzDecay,
    EzCurl,
    PsiEzX,
    PsiEzY,
    InvKappaEzX,
    BEzX,
    CEzX,
    InvKappaEzY,
    BEzY,
    CEzY,
    invDx,
    invDy,
    xLowBoundaryMode,
    xHighBoundaryMode,
    yLowBoundaryMode,
    yHighBoundaryMode,
):
    _require_cuda_tensors(
        Ez,
        Hx,
        Hy,
        EzDecay,
        EzCurl,
        PsiEzX,
        PsiEzY,
        InvKappaEzX,
        BEzX,
        CEzX,
        InvKappaEzY,
        BEzY,
        CEzY,
    )
    if _use_compiled_field_kernels():
        get_compiled_extension().update_electric_ez_cpml(
            Ez,
            Hx,
            Hy,
            EzDecay,
            EzCurl,
            PsiEzX,
            PsiEzY,
            InvKappaEzX,
            BEzX,
            CEzX,
            InvKappaEzY,
            BEzY,
            CEzY,
            float(invDx),
            float(invDy),
            int(xLowBoundaryMode),
            int(xHighBoundaryMode),
            int(yLowBoundaryMode),
            int(yHighBoundaryMode),
        )
        return
    d_x = _backward_diff(Hy, tuple(Ez.shape), 0, xLowBoundaryMode, xHighBoundaryMode, invDx)
    d_y = _backward_diff(Hx, tuple(Ez.shape), 1, yLowBoundaryMode, yHighBoundaryMode, invDy)
    _electric_cpml(
        Ez,
        d_x,
        d_y,
        EzDecay,
        EzCurl,
        PsiEzX,
        PsiEzY,
        InvKappaEzX,
        InvKappaEzY,
        BEzX,
        BEzY,
        CEzX,
        CEzY,
        first_axis=0,
        second_axis=1,
        specs=((0, xLowBoundaryMode, xHighBoundaryMode), (1, yLowBoundaryMode, yHighBoundaryMode)),
    )


def _electric_ex_cpml_compressed(
    *,
    Ex,
    Hy,
    Hz,
    ExDecay,
    ExCurl,
    PsiExY,
    PsiExZ,
    InvKappaExY,
    BExY,
    CExY,
    InvKappaExZ,
    BExZ,
    CExZ,
    invDy,
    invDz,
    yLowBoundaryMode,
    yHighBoundaryMode,
    zLowBoundaryMode,
    zHighBoundaryMode,
    psiExYLowLength,
    psiExYHighStart,
    psiExYHighLength,
    psiExZLowLength,
    psiExZHighStart,
    psiExZHighLength,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().update_electric_ex_cpml_compressed(
            Ex,
            Hy,
            Hz,
            ExDecay,
            ExCurl,
            PsiExY,
            PsiExZ,
            InvKappaExY,
            BExY,
            CExY,
            InvKappaExZ,
            BExZ,
            CExZ,
            float(invDy),
            float(invDz),
            int(yLowBoundaryMode),
            int(yHighBoundaryMode),
            int(zLowBoundaryMode),
            int(zHighBoundaryMode),
            int(psiExYLowLength),
            int(psiExYHighStart),
            int(psiExYHighLength),
            int(psiExZLowLength),
            int(psiExZHighStart),
            int(psiExZHighLength),
            _uniform_scalar(ExDecay),
            _uniform_scalar(ExCurl),
        )
        return
    d_y = _backward_diff(Hz, tuple(Ex.shape), 1, yLowBoundaryMode, yHighBoundaryMode, invDy)
    d_z = _backward_diff(Hy, tuple(Ex.shape), 2, zLowBoundaryMode, zHighBoundaryMode, invDz)
    _electric_cpml_compressed(
        Ex,
        d_y,
        d_z,
        ExDecay,
        ExCurl,
        PsiExY,
        PsiExZ,
        InvKappaExY,
        InvKappaExZ,
        BExY,
        BExZ,
        CExY,
        CExZ,
        first_axis=1,
        second_axis=2,
        first_low_length=psiExYLowLength,
        first_high_start=psiExYHighStart,
        first_high_length=psiExYHighLength,
        second_low_length=psiExZLowLength,
        second_high_start=psiExZHighStart,
        second_high_length=psiExZHighLength,
        specs=((1, yLowBoundaryMode, yHighBoundaryMode), (2, zLowBoundaryMode, zHighBoundaryMode)),
    )


def _electric_ey_cpml_compressed(
    *,
    Ey,
    Hx,
    Hz,
    EyDecay,
    EyCurl,
    PsiEyX,
    PsiEyZ,
    InvKappaEyX,
    BEyX,
    CEyX,
    InvKappaEyZ,
    BEyZ,
    CEyZ,
    invDx,
    invDz,
    xLowBoundaryMode,
    xHighBoundaryMode,
    zLowBoundaryMode,
    zHighBoundaryMode,
    psiEyXLowLength,
    psiEyXHighStart,
    psiEyXHighLength,
    psiEyZLowLength,
    psiEyZHighStart,
    psiEyZHighLength,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().update_electric_ey_cpml_compressed(
            Ey,
            Hx,
            Hz,
            EyDecay,
            EyCurl,
            PsiEyX,
            PsiEyZ,
            InvKappaEyX,
            BEyX,
            CEyX,
            InvKappaEyZ,
            BEyZ,
            CEyZ,
            float(invDx),
            float(invDz),
            int(xLowBoundaryMode),
            int(xHighBoundaryMode),
            int(zLowBoundaryMode),
            int(zHighBoundaryMode),
            int(psiEyXLowLength),
            int(psiEyXHighStart),
            int(psiEyXHighLength),
            int(psiEyZLowLength),
            int(psiEyZHighStart),
            int(psiEyZHighLength),
            _uniform_scalar(EyDecay),
            _uniform_scalar(EyCurl),
        )
        return
    d_z = _backward_diff(Hx, tuple(Ey.shape), 2, zLowBoundaryMode, zHighBoundaryMode, invDz)
    d_x = _backward_diff(Hz, tuple(Ey.shape), 0, xLowBoundaryMode, xHighBoundaryMode, invDx)
    _electric_cpml_compressed(
        Ey,
        d_z,
        d_x,
        EyDecay,
        EyCurl,
        PsiEyZ,
        PsiEyX,
        InvKappaEyZ,
        InvKappaEyX,
        BEyZ,
        BEyX,
        CEyZ,
        CEyX,
        first_axis=2,
        second_axis=0,
        first_low_length=psiEyZLowLength,
        first_high_start=psiEyZHighStart,
        first_high_length=psiEyZHighLength,
        second_low_length=psiEyXLowLength,
        second_high_start=psiEyXHighStart,
        second_high_length=psiEyXHighLength,
        specs=((0, xLowBoundaryMode, xHighBoundaryMode), (2, zLowBoundaryMode, zHighBoundaryMode)),
    )


def _electric_ez_cpml_compressed(
    *,
    Ez,
    Hx,
    Hy,
    EzDecay,
    EzCurl,
    PsiEzX,
    PsiEzY,
    InvKappaEzX,
    BEzX,
    CEzX,
    InvKappaEzY,
    BEzY,
    CEzY,
    invDx,
    invDy,
    xLowBoundaryMode,
    xHighBoundaryMode,
    yLowBoundaryMode,
    yHighBoundaryMode,
    psiEzXLowLength,
    psiEzXHighStart,
    psiEzXHighLength,
    psiEzYLowLength,
    psiEzYHighStart,
    psiEzYHighLength,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().update_electric_ez_cpml_compressed(
            Ez,
            Hx,
            Hy,
            EzDecay,
            EzCurl,
            PsiEzX,
            PsiEzY,
            InvKappaEzX,
            BEzX,
            CEzX,
            InvKappaEzY,
            BEzY,
            CEzY,
            float(invDx),
            float(invDy),
            int(xLowBoundaryMode),
            int(xHighBoundaryMode),
            int(yLowBoundaryMode),
            int(yHighBoundaryMode),
            int(psiEzXLowLength),
            int(psiEzXHighStart),
            int(psiEzXHighLength),
            int(psiEzYLowLength),
            int(psiEzYHighStart),
            int(psiEzYHighLength),
            _uniform_scalar(EzDecay),
            _uniform_scalar(EzCurl),
        )
        return
    d_x = _backward_diff(Hy, tuple(Ez.shape), 0, xLowBoundaryMode, xHighBoundaryMode, invDx)
    d_y = _backward_diff(Hx, tuple(Ez.shape), 1, yLowBoundaryMode, yHighBoundaryMode, invDy)
    _electric_cpml_compressed(
        Ez,
        d_x,
        d_y,
        EzDecay,
        EzCurl,
        PsiEzX,
        PsiEzY,
        InvKappaEzX,
        InvKappaEzY,
        BEzX,
        BEzY,
        CEzX,
        CEzY,
        first_axis=0,
        second_axis=1,
        first_low_length=psiEzXLowLength,
        first_high_start=psiEzXHighStart,
        first_high_length=psiEzXHighLength,
        second_low_length=psiEzYLowLength,
        second_high_start=psiEzYHighStart,
        second_high_length=psiEzYHighLength,
        specs=((0, xLowBoundaryMode, xHighBoundaryMode), (1, yLowBoundaryMode, yHighBoundaryMode)),
    )


def _phase_positive(real: torch.Tensor, imag: torch.Tensor, phase_cos: float, phase_sin: float):
    return (
        float(phase_cos) * real - float(phase_sin) * imag,
        float(phase_sin) * real + float(phase_cos) * imag,
    )


def _phase_negative(real: torch.Tensor, imag: torch.Tensor, phase_cos: float, phase_sin: float):
    return (
        float(phase_cos) * real + float(phase_sin) * imag,
        float(phase_cos) * imag - float(phase_sin) * real,
    )


def _bloch_backward_diff(
    real: torch.Tensor,
    imag: torch.Tensor,
    target_shape: tuple[int, int, int],
    axis: int,
    phase_cos: float,
    phase_sin: float,
    inv_delta: float,
):
    real_diff = torch.zeros(target_shape, device=real.device, dtype=real.dtype)
    imag_diff = torch.zeros(target_shape, device=real.device, dtype=real.dtype)
    size = target_shape[axis]
    if size > 2:
        dst = [slice(None)] * 3
        dst[axis] = slice(1, size - 1)
        hi = [slice(None)] * 3
        hi[axis] = slice(1, size - 1)
        lo = [slice(None)] * 3
        lo[axis] = slice(0, size - 2)
        real_diff[tuple(dst)] = (real[tuple(hi)] - real[tuple(lo)]) * float(inv_delta)
        imag_diff[tuple(dst)] = (imag[tuple(hi)] - imag[tuple(lo)]) * float(inv_delta)

    low = [slice(None)] * 3
    low[axis] = 0
    high = [slice(None)] * 3
    high[axis] = real.shape[axis] - 1
    low_dst = [slice(None)] * 3
    low_dst[axis] = 0
    high_dst = [slice(None)] * 3
    high_dst[axis] = size - 1

    neg_high_r, neg_high_i = _phase_negative(real[tuple(high)], imag[tuple(high)], phase_cos, phase_sin)
    real_diff[tuple(low_dst)] = (real[tuple(low)] - neg_high_r) * float(inv_delta)
    imag_diff[tuple(low_dst)] = (imag[tuple(low)] - neg_high_i) * float(inv_delta)
    if size > 1:
        pos_low_r, pos_low_i = _phase_positive(real[tuple(low)], imag[tuple(low)], phase_cos, phase_sin)
        real_diff[tuple(high_dst)] = (pos_low_r - real[tuple(high)]) * float(inv_delta)
        imag_diff[tuple(high_dst)] = (pos_low_i - imag[tuple(high)]) * float(inv_delta)
    return real_diff, imag_diff


def _electric_ex_bloch(
    *,
    ExReal,
    ExImag,
    HyReal,
    HyImag,
    HzReal,
    HzImag,
    ExDecay,
    ExCurl,
    phaseCosY,
    phaseSinY,
    phaseCosZ,
    phaseSinZ,
    invDy,
    invDz,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().update_electric_ex_bloch(
            ExReal,
            ExImag,
            HyReal,
            HyImag,
            HzReal,
            HzImag,
            ExDecay,
            ExCurl,
            float(phaseCosY),
            float(phaseSinY),
            float(phaseCosZ),
            float(phaseSinZ),
            float(invDy),
            float(invDz),
        )
        return
    d_y_r, d_y_i = _bloch_backward_diff(HzReal, HzImag, tuple(ExReal.shape), 1, phaseCosY, phaseSinY, invDy)
    d_z_r, d_z_i = _bloch_backward_diff(HyReal, HyImag, tuple(ExReal.shape), 2, phaseCosZ, phaseSinZ, invDz)
    ExReal.copy_(ExReal * ExDecay + ExCurl * (d_y_r - d_z_r))
    ExImag.copy_(ExImag * ExDecay + ExCurl * (d_y_i - d_z_i))


def _electric_ey_bloch(
    *,
    EyReal,
    EyImag,
    HxReal,
    HxImag,
    HzReal,
    HzImag,
    EyDecay,
    EyCurl,
    phaseCosX,
    phaseSinX,
    phaseCosZ,
    phaseSinZ,
    invDx,
    invDz,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().update_electric_ey_bloch(
            EyReal,
            EyImag,
            HxReal,
            HxImag,
            HzReal,
            HzImag,
            EyDecay,
            EyCurl,
            float(phaseCosX),
            float(phaseSinX),
            float(phaseCosZ),
            float(phaseSinZ),
            float(invDx),
            float(invDz),
        )
        return
    d_z_r, d_z_i = _bloch_backward_diff(HxReal, HxImag, tuple(EyReal.shape), 2, phaseCosZ, phaseSinZ, invDz)
    d_x_r, d_x_i = _bloch_backward_diff(HzReal, HzImag, tuple(EyReal.shape), 0, phaseCosX, phaseSinX, invDx)
    EyReal.copy_(EyReal * EyDecay + EyCurl * (d_z_r - d_x_r))
    EyImag.copy_(EyImag * EyDecay + EyCurl * (d_z_i - d_x_i))


def _electric_ez_bloch(
    *,
    EzReal,
    EzImag,
    HxReal,
    HxImag,
    HyReal,
    HyImag,
    EzDecay,
    EzCurl,
    phaseCosX,
    phaseSinX,
    phaseCosY,
    phaseSinY,
    invDx,
    invDy,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().update_electric_ez_bloch(
            EzReal,
            EzImag,
            HxReal,
            HxImag,
            HyReal,
            HyImag,
            EzDecay,
            EzCurl,
            float(phaseCosX),
            float(phaseSinX),
            float(phaseCosY),
            float(phaseSinY),
            float(invDx),
            float(invDy),
        )
        return
    d_x_r, d_x_i = _bloch_backward_diff(HyReal, HyImag, tuple(EzReal.shape), 0, phaseCosX, phaseSinX, invDx)
    d_y_r, d_y_i = _bloch_backward_diff(HxReal, HxImag, tuple(EzReal.shape), 1, phaseCosY, phaseSinY, invDy)
    EzReal.copy_(EzReal * EzDecay + EzCurl * (d_x_r - d_y_r))
    EzImag.copy_(EzImag * EzDecay + EzCurl * (d_x_i - d_y_i))


def _patch_coords_and_mask(field: torch.Tensor, patch: torch.Tensor, offsets):
    grids = torch.meshgrid(
        torch.arange(patch.shape[0], device=field.device, dtype=torch.long),
        torch.arange(patch.shape[1], device=field.device, dtype=torch.long),
        torch.arange(patch.shape[2], device=field.device, dtype=torch.long),
        indexing="ij",
    )
    coords = [grids[axis] + int(offsets[axis]) for axis in range(3)]
    mask = (
        (coords[0] >= 0)
        & (coords[0] < field.shape[0])
        & (coords[1] >= 0)
        & (coords[1] < field.shape[1])
        & (coords[2] >= 0)
        & (coords[2] < field.shape[2])
    )
    return coords, mask


def _scatter_values(field: torch.Tensor, coords, mask: torch.Tensor, values: torch.Tensor) -> None:
    if not bool(mask.any().item()):
        return
    field.index_put_(
        (coords[0][mask], coords[1][mask], coords[2][mask]),
        values[mask],
        accumulate=True,
    )


def _scatter_patch(field: torch.Tensor, patch: torch.Tensor, offsets, values: torch.Tensor) -> None:
    _require_cuda_tensors(field, patch, values)
    coords, mask = _patch_coords_and_mask(field, patch, offsets)
    _scatter_values(field, coords, mask, values)


def _boundary_duplicate_coords(field: torch.Tensor, patch: torch.Tensor, offsets, axes: tuple[int, ...]):
    coords, mask = _patch_coords_and_mask(field, patch, offsets)
    dst_coords = list(coords)
    low_masks = []
    for axis in axes:
        low = coords[axis] == 0
        high = coords[axis] + 1 >= field.shape[axis]
        mask = mask & (low | high)
        low_masks.append(low)
        dst_coords[axis] = torch.where(low, torch.full_like(coords[axis], field.shape[axis] - 1), torch.zeros_like(coords[axis]))
    return dst_coords, mask, tuple(low_masks)


def _scatter_periodic_duplicate(
    field: torch.Tensor,
    patch: torch.Tensor,
    offsets,
    values: torch.Tensor,
    axes: tuple[int, ...],
) -> None:
    dst_coords, mask, _ = _boundary_duplicate_coords(field, patch, offsets, axes)
    _scatter_values(field, dst_coords, mask, values)


def _scatter_bloch_duplicate(
    real_field: torch.Tensor,
    imag_field: torch.Tensor,
    patch: torch.Tensor,
    offsets,
    real_values: torch.Tensor,
    imag_values: torch.Tensor,
    axes_and_phases: tuple[tuple[int, float, float], ...],
) -> None:
    axes = tuple(axis for axis, _, _ in axes_and_phases)
    dst_coords, mask, low_masks = _boundary_duplicate_coords(real_field, patch, offsets, axes)
    out_real = real_values
    out_imag = imag_values
    for low, (_, phase_cos, phase_sin) in zip(low_masks, axes_and_phases):
        pos_real, pos_imag = _phase_positive(out_real, out_imag, phase_cos, phase_sin)
        neg_real, neg_imag = _phase_negative(out_real, out_imag, phase_cos, phase_sin)
        out_real = torch.where(low, pos_real, neg_real)
        out_imag = torch.where(low, pos_imag, neg_imag)
    _scatter_values(real_field, dst_coords, mask, out_real)
    _scatter_values(imag_field, dst_coords, mask, out_imag)


def _add_source_patch(*, field, sourcePatch, offsetI, offsetJ, offsetK, signal):
    if _use_compiled_field_kernels():
        get_compiled_extension().add_source_patch(
            field,
            sourcePatch,
            int(offsetI),
            int(offsetJ),
            int(offsetK),
            float(signal),
        )
        return
    _scatter_patch(field, sourcePatch, (offsetI, offsetJ, offsetK), sourcePatch * float(signal))


def _add_cw_phased_source_patch(
    *,
    field,
    sourcePatchCos,
    sourcePatchSin,
    offsetI,
    offsetJ,
    offsetK,
    signalCos,
    signalSin,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().add_cw_phased_source_patch(
            field,
            sourcePatchCos,
            sourcePatchSin,
            int(offsetI),
            int(offsetJ),
            int(offsetK),
            float(signalCos),
            float(signalSin),
        )
        return
    values = sourcePatchCos * float(signalCos) + sourcePatchSin * float(signalSin)
    _scatter_patch(field, sourcePatchCos, (offsetI, offsetJ, offsetK), values)


def _evaluate_source_time_sample(timeKind, sample_time, frequency, fwidth, amplitude, phase, delay):
    two_pi = 2.0 * math.pi
    if int(timeKind) == 0:
        return float(amplitude) * torch.cos(two_pi * float(frequency) * sample_time + float(phase))
    if int(timeKind) == 1:
        sigma_t = 1.0 / max(two_pi * float(fwidth), 1.0e-30)
        tau = sample_time - float(delay)
        envelope = torch.exp(-0.5 * (tau / sigma_t) ** 2)
        return float(amplitude) * envelope * torch.cos(two_pi * float(frequency) * tau + float(phase))
    tau = sample_time - float(delay)
    alpha = math.pi * float(frequency) * tau
    alpha_sq = alpha * alpha
    return float(amplitude) * (1.0 - 2.0 * alpha_sq) * torch.exp(-alpha_sq)


def _add_time_shifted_source_patch(
    *,
    field,
    sourcePatch,
    delayPatch,
    activationDelayPatch,
    offsetI,
    offsetJ,
    offsetK,
    timeKind,
    time,
    frequency,
    fwidth,
    amplitude,
    phase,
    delay,
    causalGate,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().add_time_shifted_source_patch(
            field,
            sourcePatch,
            delayPatch,
            activationDelayPatch,
            int(offsetI),
            int(offsetJ),
            int(offsetK),
            int(timeKind),
            float(time),
            float(frequency),
            float(fwidth),
            float(amplitude),
            float(phase),
            float(delay),
            int(causalGate),
        )
        return
    sample_time = float(time) - delayPatch
    signal = _evaluate_source_time_sample(timeKind, sample_time, frequency, fwidth, amplitude, phase, delay)
    values = signal * sourcePatch
    if int(causalGate) != 0:
        values = torch.where(float(time) >= activationDelayPatch, values, torch.zeros_like(values))
    _scatter_patch(field, sourcePatch, (offsetI, offsetJ, offsetK), values)


def _add_source_patch_periodic(*, signal, sourcePatch, offsetI, offsetJ, offsetK, wrapAxisA, wrapAxisB, **kwargs):
    if "Ex" in kwargs:
        field = kwargs["Ex"]
        extension_method = "add_source_patch_ex_periodic"
    elif "Ey" in kwargs:
        field = kwargs["Ey"]
        extension_method = "add_source_patch_ey_periodic"
    else:
        field = kwargs["Ez"]
        extension_method = "add_source_patch_ez_periodic"
    if _use_compiled_field_kernels():
        getattr(get_compiled_extension(), extension_method)(
            field,
            sourcePatch,
            int(offsetI),
            int(offsetJ),
            int(offsetK),
            float(signal),
            int(wrapAxisA),
            int(wrapAxisB),
        )
        return
    offsets = (int(offsetI), int(offsetJ), int(offsetK))
    values = sourcePatch * float(signal)
    _scatter_patch(field, sourcePatch, offsets, values)
    tangential = (1, 2) if "Ex" in kwargs else ((0, 2) if "Ey" in kwargs else (0, 1))
    duplicate_axes = tuple(
        axis for enabled, axis in ((int(wrapAxisA) != 0, tangential[0]), (int(wrapAxisB) != 0, tangential[1])) if enabled
    )
    for axis in duplicate_axes:
        _scatter_periodic_duplicate(field, sourcePatch, offsets, values, (axis,))
    if len(duplicate_axes) == 2:
        _scatter_periodic_duplicate(field, sourcePatch, offsets, values, duplicate_axes)


def _add_source_patch_bloch(
    *,
    ExReal,
    ExImag,
    EyReal,
    EyImag,
    EzReal,
    EzImag,
    sourcePatch,
    offsetI,
    offsetJ,
    offsetK,
    signalReal,
    signalImag,
    axisCode,
    phaseCosA,
    phaseSinA,
    phaseCosB,
    phaseSinB,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().add_source_patch_bloch(
            ExReal,
            ExImag,
            EyReal,
            EyImag,
            EzReal,
            EzImag,
            sourcePatch,
            int(offsetI),
            int(offsetJ),
            int(offsetK),
            float(signalReal),
            float(signalImag),
            int(axisCode),
            float(phaseCosA),
            float(phaseSinA),
            float(phaseCosB),
            float(phaseSinB),
        )
        return
    fields = ((ExReal, ExImag), (EyReal, EyImag), (EzReal, EzImag))[int(axisCode)]
    offsets = (int(offsetI), int(offsetJ), int(offsetK))
    real_values = sourcePatch * float(signalReal)
    imag_values = sourcePatch * float(signalImag)
    _scatter_patch(fields[0], sourcePatch, offsets, real_values)
    _scatter_patch(fields[1], sourcePatch, offsets, imag_values)
    tangential = ((1, 2), (0, 2), (0, 1))[int(axisCode)]
    phase_axes = (
        (tangential[0], float(phaseCosA), float(phaseSinA)),
        (tangential[1], float(phaseCosB), float(phaseSinB)),
    )
    _scatter_bloch_duplicate(fields[0], fields[1], sourcePatch, offsets, real_values, imag_values, (phase_axes[0],))
    _scatter_bloch_duplicate(fields[0], fields[1], sourcePatch, offsets, real_values, imag_values, (phase_axes[1],))
    _scatter_bloch_duplicate(fields[0], fields[1], sourcePatch, offsets, real_values, imag_values, phase_axes)


def _add_scaled_slice_source_patch(*, field, sourcePatch, incidentField, sampleIndex, offsetI, offsetJ, offsetK, scale):
    if _use_compiled_field_kernels():
        get_compiled_extension().add_scaled_slice_source_patch(
            field,
            sourcePatch,
            incidentField,
            int(sampleIndex),
            int(offsetI),
            int(offsetJ),
            int(offsetK),
            float(scale),
        )
        return
    values = sourcePatch * incidentField[int(sampleIndex)] * float(scale)
    _scatter_patch(field, sourcePatch, (offsetI, offsetJ, offsetK), values)


def _add_scaled_line_source_patch(
    *,
    field,
    coeffPatch,
    incidentField,
    sampleIndices,
    sampleAxisCode,
    offsetI,
    offsetJ,
    offsetK,
    scale,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().add_scaled_line_source_patch(
            field,
            coeffPatch,
            incidentField,
            sampleIndices.to(dtype=torch.int32).contiguous(),
            int(sampleAxisCode),
            int(offsetI),
            int(offsetJ),
            int(offsetK),
            float(scale),
        )
        return
    axis = int(sampleAxisCode)
    index_shape = [1, 1, 1]
    index_shape[axis] = int(coeffPatch.shape[axis])
    samples = incidentField[sampleIndices.to(dtype=torch.long)].reshape(index_shape)
    values = coeffPatch * samples * float(scale)
    _scatter_patch(field, coeffPatch, (offsetI, offsetJ, offsetK), values)


def _add_interpolated_source_patch(
    *,
    field,
    coeffPatch,
    incidentField,
    samplePositions,
    origin,
    ds,
    offsetI,
    offsetJ,
    offsetK,
    scale,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().add_interpolated_source_patch(
            field,
            coeffPatch,
            incidentField,
            samplePositions,
            float(origin),
            float(ds),
            int(offsetI),
            int(offsetJ),
            int(offsetK),
            float(scale),
        )
        return
    coord = torch.clamp((samplePositions - float(origin)) / float(ds), 0.0, float(incidentField.numel() - 1))
    lower = torch.floor(coord).to(dtype=torch.long)
    upper = torch.clamp(lower + 1, max=incidentField.numel() - 1)
    frac = coord - lower.to(dtype=coord.dtype)
    incident = incidentField[lower] + frac * (incidentField[upper] - incidentField[lower])
    _scatter_patch(field, coeffPatch, (offsetI, offsetJ, offsetK), coeffPatch * incident * float(scale))


def _legacy_batched_flat_metadata(
    *,
    coeff_count: int,
    termStarts,
    termShapes,
    termOffsets,
    fieldCodes,
    fieldShapes,
    sampleAxisCodes=None,
    sampleIndexStarts=None,
    sampleIndices=None,
):
    device = termStarts.device
    term_starts_cpu = termStarts.detach().to(device="cpu", dtype=torch.long)
    term_shapes_cpu = termShapes.detach().to(device="cpu", dtype=torch.long)
    term_offsets_cpu = termOffsets.detach().to(device="cpu", dtype=torch.long)
    field_codes_cpu = fieldCodes.detach().to(device="cpu", dtype=torch.long)
    sample_axis_cpu = (
        sampleAxisCodes.detach().to(device="cpu", dtype=torch.long)
        if sampleAxisCodes is not None
        else None
    )
    sample_starts_cpu = (
        sampleIndexStarts.detach().to(device="cpu", dtype=torch.long)
        if sampleIndexStarts is not None
        else None
    )
    field_offset_dtype = (
        torch.int32
        if all(int(shape[0]) * int(shape[1]) * int(shape[2]) <= torch.iinfo(torch.int32).max for shape in fieldShapes)
        else torch.long
    )
    field_codes_flat = torch.empty(coeff_count, device=device, dtype=torch.int32)
    field_offsets = torch.empty(coeff_count, device=device, dtype=field_offset_dtype)
    sample_indices_flat = (
        torch.empty(coeff_count, device=device, dtype=torch.int32)
        if sampleIndices is not None
        else None
    )
    term_count = int(term_starts_cpu.numel())
    for term_index in range(term_count):
        start = int(term_starts_cpu[term_index].item())
        end = int(term_starts_cpu[term_index + 1].item()) if term_index + 1 < term_count else coeff_count
        sx = int(term_shapes_cpu[term_index, 0].item())
        sy = int(term_shapes_cpu[term_index, 1].item())
        sz = int(term_shapes_cpu[term_index, 2].item())
        count = end - start
        local_linear = torch.arange(count, device=device, dtype=torch.int32)
        stride_i = sy * sz
        local_i = local_linear // stride_i
        remainder = local_linear - local_i * stride_i
        local_j = remainder // sz
        local_k = remainder - local_j * sz
        field_codes_flat[start:end] = int(field_codes_cpu[term_index].item())
        field_i = local_i + int(term_offsets_cpu[term_index, 0].item())
        field_j = local_j + int(term_offsets_cpu[term_index, 1].item())
        field_k = local_k + int(term_offsets_cpu[term_index, 2].item())
        field_code = int(field_codes_cpu[term_index].item())
        field_shape = fieldShapes[field_code]
        field_offsets[start:end] = (
            field_i.to(dtype=torch.long) * (int(field_shape[1]) * int(field_shape[2]))
            + field_j.to(dtype=torch.long) * int(field_shape[2])
            + field_k.to(dtype=torch.long)
        ).to(dtype=field_offset_dtype)
        if sample_indices_flat is not None:
            axis = int(sample_axis_cpu[term_index].item())
            sample_linear = local_i if axis == 0 else (local_j if axis == 1 else local_k)
            sample_start = int(sample_starts_cpu[term_index].item())
            sample_indices_flat[start:end] = sampleIndices[sample_start + sample_linear.to(dtype=torch.long)]
    return field_codes_flat, field_offsets, sample_indices_flat


def _add_batched_reference_source_patches(
    *,
    fieldX,
    fieldY,
    fieldZ,
    coeffData,
    incidentField,
    termStarts,
    termShapes,
    termOffsets,
    fieldCodes,
    sampleAxisCodes,
    sampleIndexStarts,
    sampleIndices,
    fieldCodesPerCoeff=None,
    fieldOffsets=None,
    sampleIndicesPerCoeff=None,
):
    if fieldCodesPerCoeff is None or fieldOffsets is None or sampleIndicesPerCoeff is None:
        fieldCodesPerCoeff, fieldOffsets, sampleIndicesPerCoeff = _legacy_batched_flat_metadata(
            coeff_count=int(coeffData.numel()),
            termStarts=termStarts,
            termShapes=termShapes,
            termOffsets=termOffsets,
            fieldCodes=fieldCodes,
            fieldShapes=(fieldX.shape, fieldY.shape, fieldZ.shape),
            sampleAxisCodes=sampleAxisCodes,
            sampleIndexStarts=sampleIndexStarts,
            sampleIndices=sampleIndices,
        )
    if _use_compiled_field_kernels():
        get_compiled_extension().add_batched_reference_source_patches(
            fieldX,
            fieldY,
            fieldZ,
            coeffData,
            incidentField,
            fieldCodesPerCoeff.to(dtype=torch.int32).contiguous(),
            fieldOffsets.contiguous(),
            sampleIndicesPerCoeff.to(dtype=torch.int32).contiguous(),
        )
        return
    fields = (fieldX, fieldY, fieldZ)
    term_count = int(termStarts.numel())
    total = int(coeffData.numel())
    for term_index in range(term_count):
        start = int(termStarts[term_index].item())
        end = int(termStarts[term_index + 1].item()) if term_index + 1 < term_count else total
        shape = tuple(int(value.item()) for value in termShapes[term_index])
        offsets = tuple(int(value.item()) for value in termOffsets[term_index])
        axis = int(sampleAxisCodes[term_index].item())
        sample_start = int(sampleIndexStarts[term_index].item())
        sample_count = shape[axis]
        indices = sampleIndices[sample_start : sample_start + sample_count].to(dtype=torch.long)
        view_shape = [1, 1, 1]
        view_shape[axis] = sample_count
        values = coeffData[start:end].reshape(shape) * incidentField[indices].reshape(view_shape)
        _scatter_patch(fields[int(fieldCodes[term_index].item())], values, offsets, values)


def _add_batched_interpolated_source_patches(
    *,
    fieldX,
    fieldY,
    fieldZ,
    coeffData,
    incidentField,
    samplePositions,
    termStarts,
    termShapes,
    termOffsets,
    fieldCodes,
    fieldCodesPerCoeff=None,
    fieldOffsets=None,
    origin,
    ds,
):
    if fieldCodesPerCoeff is None or fieldOffsets is None:
        fieldCodesPerCoeff, fieldOffsets, _ = _legacy_batched_flat_metadata(
            coeff_count=int(coeffData.numel()),
            termStarts=termStarts,
            termShapes=termShapes,
            termOffsets=termOffsets,
            fieldCodes=fieldCodes,
            fieldShapes=(fieldX.shape, fieldY.shape, fieldZ.shape),
        )
    if _use_compiled_field_kernels():
        get_compiled_extension().add_batched_interpolated_source_patches(
            fieldX,
            fieldY,
            fieldZ,
            coeffData,
            incidentField,
            samplePositions,
            fieldCodesPerCoeff.to(dtype=torch.int32).contiguous(),
            fieldOffsets.contiguous(),
            float(origin),
            float(ds),
        )
        return
    fields = (fieldX, fieldY, fieldZ)
    term_count = int(termStarts.numel())
    total = int(coeffData.numel())
    for term_index in range(term_count):
        start = int(termStarts[term_index].item())
        end = int(termStarts[term_index + 1].item()) if term_index + 1 < term_count else total
        shape = tuple(int(value.item()) for value in termShapes[term_index])
        offsets = tuple(int(value.item()) for value in termOffsets[term_index])
        coeff = coeffData[start:end].reshape(shape)
        positions = samplePositions[start:end].reshape(shape)
        coord = torch.clamp((positions - float(origin)) / float(ds), 0.0, float(incidentField.numel() - 1))
        lower = torch.floor(coord).to(dtype=torch.long)
        upper = torch.clamp(lower + 1, max=incidentField.numel() - 1)
        frac = coord - lower.to(dtype=coord.dtype)
        incident = incidentField[lower] + frac * (incidentField[upper] - incidentField[lower])
        values = coeff * incident
        _scatter_patch(fields[int(fieldCodes[term_index].item())], values, offsets, values)


def _update_auxiliary_magnetic(*, Magnetic, Electric, MagneticDecay, MagneticCurl):
    if _use_compiled_field_kernels():
        get_compiled_extension().update_auxiliary_magnetic(
            Magnetic,
            Electric,
            MagneticDecay,
            MagneticCurl,
        )
        return
    Magnetic.copy_(MagneticDecay * Magnetic - MagneticCurl * (Electric[1:] - Electric[:-1]))


def _update_auxiliary_electric(*, Electric, Magnetic, ElectricDecay, ElectricCurl, sourceIndex, sourceValue):
    if _use_compiled_field_kernels():
        get_compiled_extension().update_auxiliary_electric(
            Electric,
            Magnetic,
            ElectricDecay,
            ElectricCurl,
            int(sourceIndex),
            float(sourceValue),
        )
        return
    updated = Electric.clone()
    if Electric.numel() > 0:
        updated[-1] = 0.0
    if Electric.numel() > 2:
        interior = torch.arange(1, Electric.numel() - 1, device=Electric.device)
        mask = interior != int(sourceIndex)
        active = interior[mask]
        updated[active] = ElectricDecay[active] * Electric[active] - ElectricCurl[active] * (
            Magnetic[active] - Magnetic[active - 1]
        )
    if 0 <= int(sourceIndex) < Electric.numel():
        updated[int(sourceIndex)] = float(sourceValue)
    Electric.copy_(updated)


def _accumulate_dft_batched(
    *,
    Ex,
    Ey,
    Ez,
    ExRealAccum,
    ExImagAccum,
    EyRealAccum,
    EyImagAccum,
    EzRealAccum,
    EzImagAccum,
    weightedCos,
    weightedSin,
):
    _require_cuda_tensors(Ex, Ey, Ez, weightedCos, weightedSin)
    if _use_compiled_field_kernels():
        get_compiled_extension().accumulate_dft_batched(
            Ex,
            Ey,
            Ez,
            ExRealAccum,
            ExImagAccum,
            EyRealAccum,
            EyImagAccum,
            EzRealAccum,
            EzImagAccum,
            weightedCos,
            weightedSin,
        )
        return
    view_shape = (weightedCos.shape[0],) + (1, 1, 1)
    cos = weightedCos.reshape(view_shape)
    sin = weightedSin.reshape(view_shape)
    ExRealAccum.add_(Ex.unsqueeze(0) * cos)
    ExImagAccum.add_(Ex.unsqueeze(0) * sin)
    EyRealAccum.add_(Ey.unsqueeze(0) * cos)
    EyImagAccum.add_(Ey.unsqueeze(0) * sin)
    EzRealAccum.add_(Ez.unsqueeze(0) * cos)
    EzImagAccum.add_(Ez.unsqueeze(0) * sin)


def _accumulate_point_observers(
    *,
    field,
    pointI,
    pointJ,
    pointK,
    realAccum,
    imagAccum,
    weightedCos,
    weightedSin,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().accumulate_point_observers(
            field,
            pointI.contiguous(),
            pointJ.contiguous(),
            pointK.contiguous(),
            realAccum,
            imagAccum,
            float(weightedCos),
            float(weightedSin),
        )
        return
    values = field[pointI.to(torch.long), pointJ.to(torch.long), pointK.to(torch.long)]
    realAccum.add_(values * float(weightedCos))
    imagAccum.add_(values * float(weightedSin))


def _accumulate_plane_observer(*, field, planeRealAccum, planeImagAccum, axisCode, planeIndex, weightedCos, weightedSin):
    if _use_compiled_field_kernels():
        get_compiled_extension().accumulate_plane_observer(
            field,
            planeRealAccum,
            planeImagAccum,
            int(axisCode),
            int(planeIndex),
            float(weightedCos),
            float(weightedSin),
        )
        return
    axis = int(axisCode)
    if axis == 0:
        values = field[int(planeIndex), :, :]
    elif axis == 1:
        values = field[:, int(planeIndex), :]
    else:
        values = field[:, :, int(planeIndex)]
    planeRealAccum.add_(values * float(weightedCos))
    planeImagAccum.add_(values * float(weightedSin))


def _update_debye_current(*, ElectricField, Polarization, PolarizationCurrent, DebyeDrive, decay, dt):
    if _use_compiled_field_kernels():
        get_compiled_extension().update_debye_current(
            ElectricField,
            Polarization,
            PolarizationCurrent,
            DebyeDrive,
            float(decay),
            float(dt),
        )
        return
    previous = Polarization.clone()
    next_polarization = float(decay) * previous + DebyeDrive * ElectricField
    Polarization.copy_(next_polarization)
    PolarizationCurrent.copy_((next_polarization - previous) / float(dt))


def _update_drude_current(*, ElectricField, PolarizationCurrent, DrudeDrive, decay):
    if _use_compiled_field_kernels():
        get_compiled_extension().update_drude_current(
            ElectricField,
            PolarizationCurrent,
            DrudeDrive,
            float(decay),
        )
        return
    PolarizationCurrent.copy_(float(decay) * PolarizationCurrent + DrudeDrive * ElectricField)


def _update_lorentz_current(*, ElectricField, Polarization, PolarizationCurrent, LorentzDrive, decay, restoring, dt):
    if _use_compiled_field_kernels():
        get_compiled_extension().update_lorentz_current(
            ElectricField,
            Polarization,
            PolarizationCurrent,
            LorentzDrive,
            float(decay),
            float(restoring),
            float(dt),
        )
        return
    next_current = (
        float(decay) * PolarizationCurrent
        - float(restoring) * Polarization
        + LorentzDrive * ElectricField
    )
    PolarizationCurrent.copy_(next_current)
    Polarization.add_(float(dt) * next_current)


def _apply_polarization_current(*, ElectricField, PolarizationCurrent, InvPermittivity, dt):
    if _use_compiled_field_kernels():
        get_compiled_extension().apply_polarization_current(
            ElectricField,
            PolarizationCurrent,
            InvPermittivity,
            float(dt),
        )
        return
    ElectricField.sub_(float(dt) * PolarizationCurrent * InvPermittivity)


def _sample_clamped(field: torch.Tensor, i, j, k):
    ii = torch.clamp(i, 0, field.shape[0] - 1)
    jj = torch.clamp(j, 0, field.shape[1] - 1)
    kk = torch.clamp(k, 0, field.shape[2] - 1)
    return field[ii, jj, kk]


def _kerr_indices(shape, device):
    return torch.meshgrid(
        torch.arange(shape[0], device=device),
        torch.arange(shape[1], device=device),
        torch.arange(shape[2], device=device),
        indexing="ij",
    )


def _update_kerr_ex(*, DynamicCurl, Ex, Ey, Ez, LinearPermittivity, ExDecay, KerrChi3, dt, eps0):
    if _use_compiled_field_kernels():
        _invalidate_uniform_scalar(DynamicCurl)
        get_compiled_extension().update_kerr_ex_curl(
            DynamicCurl,
            Ex,
            Ey,
            Ez,
            LinearPermittivity,
            ExDecay,
            KerrChi3,
            float(dt),
            float(eps0),
        )
        return
    i, j, k = _kerr_indices(tuple(DynamicCurl.shape), DynamicCurl.device)
    ey = 0.25 * (
        _sample_clamped(Ey, i, j - 1, k) + _sample_clamped(Ey, i, j, k)
        + _sample_clamped(Ey, i + 1, j - 1, k) + _sample_clamped(Ey, i + 1, j, k)
    )
    ez = 0.25 * (
        _sample_clamped(Ez, i, j, k - 1) + _sample_clamped(Ez, i, j, k)
        + _sample_clamped(Ez, i + 1, j, k - 1) + _sample_clamped(Ez, i + 1, j, k)
    )
    effective = torch.clamp(LinearPermittivity + float(eps0) * KerrChi3 * (Ex * Ex + ey * ey + ez * ez), min=1.0e-12 * float(eps0))
    DynamicCurl.copy_((float(dt) / effective) * ExDecay)


def _update_kerr_ey(*, DynamicCurl, Ex, Ey, Ez, LinearPermittivity, EyDecay, KerrChi3, dt, eps0):
    if _use_compiled_field_kernels():
        _invalidate_uniform_scalar(DynamicCurl)
        get_compiled_extension().update_kerr_ey_curl(
            DynamicCurl,
            Ex,
            Ey,
            Ez,
            LinearPermittivity,
            EyDecay,
            KerrChi3,
            float(dt),
            float(eps0),
        )
        return
    i, j, k = _kerr_indices(tuple(DynamicCurl.shape), DynamicCurl.device)
    ex = 0.25 * (
        _sample_clamped(Ex, i - 1, j, k) + _sample_clamped(Ex, i, j, k)
        + _sample_clamped(Ex, i - 1, j + 1, k) + _sample_clamped(Ex, i, j + 1, k)
    )
    ez = 0.25 * (
        _sample_clamped(Ez, i, j, k - 1) + _sample_clamped(Ez, i, j, k)
        + _sample_clamped(Ez, i, j + 1, k - 1) + _sample_clamped(Ez, i, j + 1, k)
    )
    effective = torch.clamp(LinearPermittivity + float(eps0) * KerrChi3 * (ex * ex + Ey * Ey + ez * ez), min=1.0e-12 * float(eps0))
    DynamicCurl.copy_((float(dt) / effective) * EyDecay)


def _update_kerr_ez(*, DynamicCurl, Ex, Ey, Ez, LinearPermittivity, EzDecay, KerrChi3, dt, eps0):
    if _use_compiled_field_kernels():
        _invalidate_uniform_scalar(DynamicCurl)
        get_compiled_extension().update_kerr_ez_curl(
            DynamicCurl,
            Ex,
            Ey,
            Ez,
            LinearPermittivity,
            EzDecay,
            KerrChi3,
            float(dt),
            float(eps0),
        )
        return
    i, j, k = _kerr_indices(tuple(DynamicCurl.shape), DynamicCurl.device)
    ex = 0.25 * (
        _sample_clamped(Ex, i - 1, j, k) + _sample_clamped(Ex, i, j, k)
        + _sample_clamped(Ex, i - 1, j, k + 1) + _sample_clamped(Ex, i, j, k + 1)
    )
    ey = 0.25 * (
        _sample_clamped(Ey, i, j - 1, k) + _sample_clamped(Ey, i, j, k)
        + _sample_clamped(Ey, i, j - 1, k + 1) + _sample_clamped(Ey, i, j, k + 1)
    )
    effective = torch.clamp(LinearPermittivity + float(eps0) * KerrChi3 * (ex * ex + ey * ey + Ez * Ez), min=1.0e-12 * float(eps0))
    DynamicCurl.copy_((float(dt) / effective) * EzDecay)


def _idx_active(shape, i, j, k, field_name):
    nx, ny, nz = shape
    if i < 0 or i >= nx or j < 0 or j >= ny or k < 0 or k >= nz:
        return False
    if field_name == "Ex":
        return 0 < j < ny - 1 and 0 < k < nz - 1
    if field_name == "Ey":
        return 0 < i < nx - 1 and 0 < k < nz - 1
    if field_name == "Ez":
        return 0 < i < nx - 1 and 0 < j < ny - 1
    raise ValueError(f"unsupported field {field_name!r}")


def _electric_cell_status(coord_a, size_a, low_a, high_a, coord_b, size_b, low_b, high_b):
    def status_for(coord, size, low, high):
        if coord != 0 and coord + 1 != size:
            return False, False
        mode = int(low) if coord == 0 else int(high)
        return mode == BOUNDARY_PEC, mode in {BOUNDARY_NONE, BOUNDARY_PML}

    pec_a, inactive_a = status_for(int(coord_a), int(size_a), low_a, high_a)
    pec_b, inactive_b = status_for(int(coord_b), int(size_b), low_b, high_b)
    pec = pec_a or pec_b
    inactive = (not pec) and (inactive_a or inactive_b)
    return (not pec and not inactive), inactive, pec


def _reverse_electric_hx_standard(*, AdjHxMid, AdjHxPost, AdjEyPost, AdjEzPost, EyCurl, EzCurl, invDy, invDz):
    if _use_compiled_field_kernels():
        get_compiled_extension().reverse_electric_adjoint_to_hx_standard(
            AdjHxMid, AdjHxPost, AdjEyPost, AdjEzPost, EyCurl, EzCurl, float(invDy), float(invDz)
        )
        return
    nx, ny, nz = AdjHxMid.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                adjoint = AdjHxPost[i, j, k]
                if _idx_active(AdjEyPost.shape, i, j, k, "Ey"):
                    adjoint = adjoint + EyCurl[i, j, k] * float(invDz) * AdjEyPost[i, j, k]
                if _idx_active(AdjEyPost.shape, i, j, k + 1, "Ey"):
                    adjoint = adjoint - EyCurl[i, j, k + 1] * float(invDz) * AdjEyPost[i, j, k + 1]
                if _idx_active(AdjEzPost.shape, i, j, k, "Ez"):
                    adjoint = adjoint - EzCurl[i, j, k] * float(invDy) * AdjEzPost[i, j, k]
                if _idx_active(AdjEzPost.shape, i, j + 1, k, "Ez"):
                    adjoint = adjoint + EzCurl[i, j + 1, k] * float(invDy) * AdjEzPost[i, j + 1, k]
                AdjHxMid[i, j, k] = adjoint


def _reverse_electric_hy_standard(*, AdjHyMid, AdjHyPost, AdjExPost, AdjEzPost, ExCurl, EzCurl, invDx, invDz):
    if _use_compiled_field_kernels():
        get_compiled_extension().reverse_electric_adjoint_to_hy_standard(
            AdjHyMid, AdjHyPost, AdjExPost, AdjEzPost, ExCurl, EzCurl, float(invDx), float(invDz)
        )
        return
    nx, ny, nz = AdjHyMid.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                adjoint = AdjHyPost[i, j, k]
                if _idx_active(AdjExPost.shape, i, j, k, "Ex"):
                    adjoint = adjoint - ExCurl[i, j, k] * float(invDz) * AdjExPost[i, j, k]
                if _idx_active(AdjExPost.shape, i, j, k + 1, "Ex"):
                    adjoint = adjoint + ExCurl[i, j, k + 1] * float(invDz) * AdjExPost[i, j, k + 1]
                if _idx_active(AdjEzPost.shape, i, j, k, "Ez"):
                    adjoint = adjoint + EzCurl[i, j, k] * float(invDx) * AdjEzPost[i, j, k]
                if _idx_active(AdjEzPost.shape, i + 1, j, k, "Ez"):
                    adjoint = adjoint - EzCurl[i + 1, j, k] * float(invDx) * AdjEzPost[i + 1, j, k]
                AdjHyMid[i, j, k] = adjoint


def _reverse_electric_hz_standard(*, AdjHzMid, AdjHzPost, AdjExPost, AdjEyPost, ExCurl, EyCurl, invDx, invDy):
    if _use_compiled_field_kernels():
        get_compiled_extension().reverse_electric_adjoint_to_hz_standard(
            AdjHzMid, AdjHzPost, AdjExPost, AdjEyPost, ExCurl, EyCurl, float(invDx), float(invDy)
        )
        return
    nx, ny, nz = AdjHzMid.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                adjoint = AdjHzPost[i, j, k]
                if _idx_active(AdjExPost.shape, i, j, k, "Ex"):
                    adjoint = adjoint + ExCurl[i, j, k] * float(invDy) * AdjExPost[i, j, k]
                if _idx_active(AdjExPost.shape, i, j + 1, k, "Ex"):
                    adjoint = adjoint - ExCurl[i, j + 1, k] * float(invDy) * AdjExPost[i, j + 1, k]
                if _idx_active(AdjEyPost.shape, i, j, k, "Ey"):
                    adjoint = adjoint - EyCurl[i, j, k] * float(invDx) * AdjEyPost[i, j, k]
                if _idx_active(AdjEyPost.shape, i + 1, j, k, "Ey"):
                    adjoint = adjoint + EyCurl[i + 1, j, k] * float(invDx) * AdjEyPost[i + 1, j, k]
                AdjHzMid[i, j, k] = adjoint


def _accumulate_bloch_backward_diff_adjoint_complex(
    field_grad_real: torch.Tensor,
    field_grad_imag: torch.Tensor,
    diff_grad_real: torch.Tensor,
    diff_grad_imag: torch.Tensor,
    *,
    axis: int,
    phase_cos: float,
    phase_sin: float,
    inv_delta: float,
) -> None:
    scale = float(inv_delta)
    if diff_grad_real.shape[axis] > 2:
        interior = [slice(None)] * 3
        interior[axis] = slice(1, -1)
        field_lo = [slice(None)] * 3
        field_hi = [slice(None)] * 3
        field_lo[axis] = slice(0, -1)
        field_hi[axis] = slice(1, None)
        field_grad_real[tuple(field_lo)].sub_(scale * diff_grad_real[tuple(interior)])
        field_grad_imag[tuple(field_lo)].sub_(scale * diff_grad_imag[tuple(interior)])
        field_grad_real[tuple(field_hi)].add_(scale * diff_grad_real[tuple(interior)])
        field_grad_imag[tuple(field_hi)].add_(scale * diff_grad_imag[tuple(interior)])

    low_grad_real = diff_grad_real.select(axis, 0)
    low_grad_imag = diff_grad_imag.select(axis, 0)
    high_grad_real = diff_grad_real.select(axis, int(diff_grad_real.shape[axis] - 1))
    high_grad_imag = diff_grad_imag.select(axis, int(diff_grad_imag.shape[axis] - 1))
    neg_high_real, neg_high_imag = _phase_negative(high_grad_real, high_grad_imag, phase_cos, phase_sin)
    pos_low_real, pos_low_imag = _phase_positive(low_grad_real, low_grad_imag, phase_cos, phase_sin)
    field_grad_real.select(axis, 0).add_(scale * low_grad_real + scale * neg_high_real)
    field_grad_imag.select(axis, 0).add_(scale * low_grad_imag + scale * neg_high_imag)
    field_grad_real.select(axis, int(field_grad_real.shape[axis] - 1)).add_(
        -scale * pos_low_real - scale * high_grad_real
    )
    field_grad_imag.select(axis, int(field_grad_imag.shape[axis] - 1)).add_(
        -scale * pos_low_imag - scale * high_grad_imag
    )


def _bloch_backward_diff_adjoint(
    target_shape: tuple[int, int, int],
    adj_real: torch.Tensor,
    adj_imag: torch.Tensor,
    curl: torch.Tensor,
    *,
    axis: int,
    sign: float,
    phase_cos: float,
    phase_sin: float,
    inv_delta: float,
):
    grad_real = torch.zeros(target_shape, device=adj_real.device, dtype=adj_real.dtype)
    grad_imag = torch.zeros(target_shape, device=adj_imag.device, dtype=adj_imag.dtype)
    scale = float(sign)
    _accumulate_bloch_backward_diff_adjoint_complex(
        grad_real,
        grad_imag,
        scale * curl * adj_real,
        scale * curl * adj_imag,
        axis=axis,
        phase_cos=phase_cos,
        phase_sin=phase_sin,
        inv_delta=inv_delta,
    )
    return grad_real, grad_imag


def _reverse_electric_hx_bloch(
    *,
    AdjHxMidReal,
    AdjHxMidImag,
    AdjHxPostReal,
    AdjHxPostImag,
    AdjEyPostReal,
    AdjEyPostImag,
    AdjEzPostReal,
    AdjEzPostImag,
    EyCurl,
    EzCurl,
    phaseCosY,
    phaseSinY,
    phaseCosZ,
    phaseSinZ,
    invDy,
    invDz,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().reverse_electric_adjoint_to_hx_bloch(
            AdjHxMidReal, AdjHxMidImag, AdjHxPostReal, AdjHxPostImag,
            AdjEyPostReal, AdjEyPostImag, AdjEzPostReal, AdjEzPostImag,
            EyCurl, EzCurl, float(phaseCosY), float(phaseSinY), float(phaseCosZ), float(phaseSinZ),
            float(invDy), float(invDz),
        )
        return
    ey_real, ey_imag = _bloch_backward_diff_adjoint(
        tuple(AdjHxMidReal.shape), AdjEyPostReal, AdjEyPostImag, EyCurl,
        axis=2, sign=1.0, phase_cos=phaseCosZ, phase_sin=phaseSinZ, inv_delta=invDz,
    )
    ez_real, ez_imag = _bloch_backward_diff_adjoint(
        tuple(AdjHxMidReal.shape), AdjEzPostReal, AdjEzPostImag, EzCurl,
        axis=1, sign=-1.0, phase_cos=phaseCosY, phase_sin=phaseSinY, inv_delta=invDy,
    )
    AdjHxMidReal.copy_(AdjHxPostReal + ey_real + ez_real)
    AdjHxMidImag.copy_(AdjHxPostImag + ey_imag + ez_imag)


def _reverse_electric_hy_bloch(
    *,
    AdjHyMidReal,
    AdjHyMidImag,
    AdjHyPostReal,
    AdjHyPostImag,
    AdjExPostReal,
    AdjExPostImag,
    AdjEzPostReal,
    AdjEzPostImag,
    ExCurl,
    EzCurl,
    phaseCosX,
    phaseSinX,
    phaseCosZ,
    phaseSinZ,
    invDx,
    invDz,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().reverse_electric_adjoint_to_hy_bloch(
            AdjHyMidReal, AdjHyMidImag, AdjHyPostReal, AdjHyPostImag,
            AdjExPostReal, AdjExPostImag, AdjEzPostReal, AdjEzPostImag,
            ExCurl, EzCurl, float(phaseCosX), float(phaseSinX), float(phaseCosZ), float(phaseSinZ),
            float(invDx), float(invDz),
        )
        return
    ex_real, ex_imag = _bloch_backward_diff_adjoint(
        tuple(AdjHyMidReal.shape), AdjExPostReal, AdjExPostImag, ExCurl,
        axis=2, sign=-1.0, phase_cos=phaseCosZ, phase_sin=phaseSinZ, inv_delta=invDz,
    )
    ez_real, ez_imag = _bloch_backward_diff_adjoint(
        tuple(AdjHyMidReal.shape), AdjEzPostReal, AdjEzPostImag, EzCurl,
        axis=0, sign=1.0, phase_cos=phaseCosX, phase_sin=phaseSinX, inv_delta=invDx,
    )
    AdjHyMidReal.copy_(AdjHyPostReal + ex_real + ez_real)
    AdjHyMidImag.copy_(AdjHyPostImag + ex_imag + ez_imag)


def _reverse_electric_hz_bloch(
    *,
    AdjHzMidReal,
    AdjHzMidImag,
    AdjHzPostReal,
    AdjHzPostImag,
    AdjExPostReal,
    AdjExPostImag,
    AdjEyPostReal,
    AdjEyPostImag,
    ExCurl,
    EyCurl,
    phaseCosX,
    phaseSinX,
    phaseCosY,
    phaseSinY,
    invDx,
    invDy,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().reverse_electric_adjoint_to_hz_bloch(
            AdjHzMidReal, AdjHzMidImag, AdjHzPostReal, AdjHzPostImag,
            AdjExPostReal, AdjExPostImag, AdjEyPostReal, AdjEyPostImag,
            ExCurl, EyCurl, float(phaseCosX), float(phaseSinX), float(phaseCosY), float(phaseSinY),
            float(invDx), float(invDy),
        )
        return
    ex_real, ex_imag = _bloch_backward_diff_adjoint(
        tuple(AdjHzMidReal.shape), AdjExPostReal, AdjExPostImag, ExCurl,
        axis=1, sign=1.0, phase_cos=phaseCosY, phase_sin=phaseSinY, inv_delta=invDy,
    )
    ey_real, ey_imag = _bloch_backward_diff_adjoint(
        tuple(AdjHzMidReal.shape), AdjEyPostReal, AdjEyPostImag, EyCurl,
        axis=0, sign=-1.0, phase_cos=phaseCosX, phase_sin=phaseSinX, inv_delta=invDx,
    )
    AdjHzMidReal.copy_(AdjHzPostReal + ex_real + ey_real)
    AdjHzMidImag.copy_(AdjHzPostImag + ex_imag + ey_imag)


def _reverse_magnetic_ex_standard(
    *,
    AdjExPrev,
    GradEpsEx,
    AdjExPost,
    AdjHyMid,
    AdjHzMid,
    ExDecay,
    ExCurl,
    EpsEx,
    HyMid,
    HzMid,
    HyCurl,
    HzCurl,
    invDy,
    invDz,
    yLowBoundaryMode,
    yHighBoundaryMode,
    zLowBoundaryMode,
    zHighBoundaryMode,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().reverse_magnetic_adjoint_to_ex_standard(
            AdjExPrev,
            GradEpsEx,
            AdjExPost,
            AdjHyMid,
            AdjHzMid,
            ExDecay,
            ExCurl,
            EpsEx,
            HyMid,
            HzMid,
            HyCurl,
            HzCurl,
            float(invDy),
            float(invDz),
            int(yLowBoundaryMode),
            int(yHighBoundaryMode),
            int(zLowBoundaryMode),
            int(zHighBoundaryMode),
        )
        return
    nx, ny, nz = AdjExPrev.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                active, inactive, _ = _electric_cell_status(j, ny, yLowBoundaryMode, yHighBoundaryMode, k, nz, zLowBoundaryMode, zHighBoundaryMode)
                adjoint = AdjExPost.new_zeros(())
                grad = AdjExPost.new_zeros(())
                if inactive:
                    adjoint = AdjExPost[i, j, k]
                elif active:
                    curl_h = (HzMid[i, j, k] - HzMid[i, j - 1, k]) * float(invDy) - (HyMid[i, j, k] - HyMid[i, j, k - 1]) * float(invDz)
                    adjoint = AdjExPost[i, j, k] * ExDecay[i, j, k]
                    grad = -AdjExPost[i, j, k] * ExCurl[i, j, k] * curl_h / EpsEx[i, j, k]
                if k < AdjHyMid.shape[2]:
                    adjoint = adjoint + HyCurl[i, j, k] * float(invDz) * AdjHyMid[i, j, k]
                if k > 0:
                    adjoint = adjoint - HyCurl[i, j, k - 1] * float(invDz) * AdjHyMid[i, j, k - 1]
                if j < AdjHzMid.shape[1]:
                    adjoint = adjoint - HzCurl[i, j, k] * float(invDy) * AdjHzMid[i, j, k]
                if j > 0:
                    adjoint = adjoint + HzCurl[i, j - 1, k] * float(invDy) * AdjHzMid[i, j - 1, k]
                AdjExPrev[i, j, k] = adjoint
                GradEpsEx[i, j, k] = grad


def _reverse_magnetic_ey_standard(
    *,
    AdjEyPrev,
    GradEpsEy,
    AdjEyPost,
    AdjHxMid,
    AdjHzMid,
    EyDecay,
    EyCurl,
    EpsEy,
    HxMid,
    HzMid,
    HxCurl,
    HzCurl,
    invDx,
    invDz,
    xLowBoundaryMode,
    xHighBoundaryMode,
    zLowBoundaryMode,
    zHighBoundaryMode,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().reverse_magnetic_adjoint_to_ey_standard(
            AdjEyPrev,
            GradEpsEy,
            AdjEyPost,
            AdjHxMid,
            AdjHzMid,
            EyDecay,
            EyCurl,
            EpsEy,
            HxMid,
            HzMid,
            HxCurl,
            HzCurl,
            float(invDx),
            float(invDz),
            int(xLowBoundaryMode),
            int(xHighBoundaryMode),
            int(zLowBoundaryMode),
            int(zHighBoundaryMode),
        )
        return
    nx, ny, nz = AdjEyPrev.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                active, inactive, _ = _electric_cell_status(i, nx, xLowBoundaryMode, xHighBoundaryMode, k, nz, zLowBoundaryMode, zHighBoundaryMode)
                adjoint = AdjEyPost.new_zeros(())
                grad = AdjEyPost.new_zeros(())
                if inactive:
                    adjoint = AdjEyPost[i, j, k]
                elif active:
                    curl_h = (HxMid[i, j, k] - HxMid[i, j, k - 1]) * float(invDz) - (HzMid[i, j, k] - HzMid[i - 1, j, k]) * float(invDx)
                    adjoint = AdjEyPost[i, j, k] * EyDecay[i, j, k]
                    grad = -AdjEyPost[i, j, k] * EyCurl[i, j, k] * curl_h / EpsEy[i, j, k]
                if k < AdjHxMid.shape[2]:
                    adjoint = adjoint - HxCurl[i, j, k] * float(invDz) * AdjHxMid[i, j, k]
                if k > 0:
                    adjoint = adjoint + HxCurl[i, j, k - 1] * float(invDz) * AdjHxMid[i, j, k - 1]
                if i < AdjHzMid.shape[0]:
                    adjoint = adjoint + HzCurl[i, j, k] * float(invDx) * AdjHzMid[i, j, k]
                if i > 0:
                    adjoint = adjoint - HzCurl[i - 1, j, k] * float(invDx) * AdjHzMid[i - 1, j, k]
                AdjEyPrev[i, j, k] = adjoint
                GradEpsEy[i, j, k] = grad


def _reverse_magnetic_ez_standard(
    *,
    AdjEzPrev,
    GradEpsEz,
    AdjEzPost,
    AdjHxMid,
    AdjHyMid,
    EzDecay,
    EzCurl,
    EpsEz,
    HxMid,
    HyMid,
    HxCurl,
    HyCurl,
    invDx,
    invDy,
    xLowBoundaryMode,
    xHighBoundaryMode,
    yLowBoundaryMode,
    yHighBoundaryMode,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().reverse_magnetic_adjoint_to_ez_standard(
            AdjEzPrev,
            GradEpsEz,
            AdjEzPost,
            AdjHxMid,
            AdjHyMid,
            EzDecay,
            EzCurl,
            EpsEz,
            HxMid,
            HyMid,
            HxCurl,
            HyCurl,
            float(invDx),
            float(invDy),
            int(xLowBoundaryMode),
            int(xHighBoundaryMode),
            int(yLowBoundaryMode),
            int(yHighBoundaryMode),
        )
        return
    nx, ny, nz = AdjEzPrev.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                active, inactive, _ = _electric_cell_status(i, nx, xLowBoundaryMode, xHighBoundaryMode, j, ny, yLowBoundaryMode, yHighBoundaryMode)
                adjoint = AdjEzPost.new_zeros(())
                grad = AdjEzPost.new_zeros(())
                if inactive:
                    adjoint = AdjEzPost[i, j, k]
                elif active:
                    curl_h = (HyMid[i, j, k] - HyMid[i - 1, j, k]) * float(invDx) - (HxMid[i, j, k] - HxMid[i, j - 1, k]) * float(invDy)
                    adjoint = AdjEzPost[i, j, k] * EzDecay[i, j, k]
                    grad = -AdjEzPost[i, j, k] * EzCurl[i, j, k] * curl_h / EpsEz[i, j, k]
                if j < AdjHxMid.shape[1]:
                    adjoint = adjoint + HxCurl[i, j, k] * float(invDy) * AdjHxMid[i, j, k]
                if j > 0:
                    adjoint = adjoint - HxCurl[i, j - 1, k] * float(invDy) * AdjHxMid[i, j - 1, k]
                if i < AdjHyMid.shape[0]:
                    adjoint = adjoint - HyCurl[i, j, k] * float(invDx) * AdjHyMid[i, j, k]
                if i > 0:
                    adjoint = adjoint + HyCurl[i - 1, j, k] * float(invDx) * AdjHyMid[i - 1, j, k]
                AdjEzPrev[i, j, k] = adjoint
                GradEpsEz[i, j, k] = grad


def _reverse_magnetic_ex_bloch(
    *,
    AdjExPrevReal,
    AdjExPrevImag,
    GradEpsEx,
    AdjExPostReal,
    AdjExPostImag,
    AdjHyMidReal,
    AdjHyMidImag,
    AdjHzMidReal,
    AdjHzMidImag,
    ExDecay,
    ExCurl,
    EpsEx,
    HyMidReal,
    HyMidImag,
    HzMidReal,
    HzMidImag,
    HyCurl,
    HzCurl,
    phaseCosY,
    phaseSinY,
    phaseCosZ,
    phaseSinZ,
    invDy,
    invDz,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().reverse_magnetic_adjoint_to_ex_bloch(
            AdjExPrevReal, AdjExPrevImag, GradEpsEx, AdjExPostReal, AdjExPostImag,
            AdjHyMidReal, AdjHyMidImag, AdjHzMidReal, AdjHzMidImag,
            ExDecay, ExCurl, EpsEx, HyMidReal, HyMidImag, HzMidReal, HzMidImag,
            HyCurl, HzCurl, float(phaseCosY), float(phaseSinY), float(phaseCosZ), float(phaseSinZ),
            float(invDy), float(invDz),
        )
        return
    d_hz_dy_r, d_hz_dy_i = _bloch_backward_diff(
        HzMidReal, HzMidImag, tuple(AdjExPrevReal.shape), 1, phaseCosY, phaseSinY, invDy,
    )
    d_hy_dz_r, d_hy_dz_i = _bloch_backward_diff(
        HyMidReal, HyMidImag, tuple(AdjExPrevReal.shape), 2, phaseCosZ, phaseSinZ, invDz,
    )
    curl_h_r = d_hz_dy_r - d_hy_dz_r
    curl_h_i = d_hz_dy_i - d_hy_dz_i
    AdjExPrevReal.copy_(AdjExPostReal * ExDecay)
    AdjExPrevImag.copy_(AdjExPostImag * ExDecay)
    GradEpsEx.copy_(-ExCurl * (AdjExPostReal * curl_h_r + AdjExPostImag * curl_h_i) / EpsEx)
    AdjExPrevReal[:, :, : HyMidReal.shape[2]].add_(HyCurl * float(invDz) * AdjHyMidReal)
    AdjExPrevImag[:, :, : HyMidImag.shape[2]].add_(HyCurl * float(invDz) * AdjHyMidImag)
    AdjExPrevReal[:, :, 1:].sub_(HyCurl * float(invDz) * AdjHyMidReal)
    AdjExPrevImag[:, :, 1:].sub_(HyCurl * float(invDz) * AdjHyMidImag)
    AdjExPrevReal[:, : HzMidReal.shape[1], :].sub_(HzCurl * float(invDy) * AdjHzMidReal)
    AdjExPrevImag[:, : HzMidImag.shape[1], :].sub_(HzCurl * float(invDy) * AdjHzMidImag)
    AdjExPrevReal[:, 1:, :].add_(HzCurl * float(invDy) * AdjHzMidReal)
    AdjExPrevImag[:, 1:, :].add_(HzCurl * float(invDy) * AdjHzMidImag)


def _reverse_magnetic_ey_bloch(
    *,
    AdjEyPrevReal,
    AdjEyPrevImag,
    GradEpsEy,
    AdjEyPostReal,
    AdjEyPostImag,
    AdjHxMidReal,
    AdjHxMidImag,
    AdjHzMidReal,
    AdjHzMidImag,
    EyDecay,
    EyCurl,
    EpsEy,
    HxMidReal,
    HxMidImag,
    HzMidReal,
    HzMidImag,
    HxCurl,
    HzCurl,
    phaseCosX,
    phaseSinX,
    phaseCosZ,
    phaseSinZ,
    invDx,
    invDz,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().reverse_magnetic_adjoint_to_ey_bloch(
            AdjEyPrevReal, AdjEyPrevImag, GradEpsEy, AdjEyPostReal, AdjEyPostImag,
            AdjHxMidReal, AdjHxMidImag, AdjHzMidReal, AdjHzMidImag,
            EyDecay, EyCurl, EpsEy, HxMidReal, HxMidImag, HzMidReal, HzMidImag,
            HxCurl, HzCurl, float(phaseCosX), float(phaseSinX), float(phaseCosZ), float(phaseSinZ),
            float(invDx), float(invDz),
        )
        return
    d_hx_dz_r, d_hx_dz_i = _bloch_backward_diff(
        HxMidReal, HxMidImag, tuple(AdjEyPrevReal.shape), 2, phaseCosZ, phaseSinZ, invDz,
    )
    d_hz_dx_r, d_hz_dx_i = _bloch_backward_diff(
        HzMidReal, HzMidImag, tuple(AdjEyPrevReal.shape), 0, phaseCosX, phaseSinX, invDx,
    )
    curl_h_r = d_hx_dz_r - d_hz_dx_r
    curl_h_i = d_hx_dz_i - d_hz_dx_i
    AdjEyPrevReal.copy_(AdjEyPostReal * EyDecay)
    AdjEyPrevImag.copy_(AdjEyPostImag * EyDecay)
    GradEpsEy.copy_(-EyCurl * (AdjEyPostReal * curl_h_r + AdjEyPostImag * curl_h_i) / EpsEy)
    AdjEyPrevReal[:, :, : HxMidReal.shape[2]].sub_(HxCurl * float(invDz) * AdjHxMidReal)
    AdjEyPrevImag[:, :, : HxMidImag.shape[2]].sub_(HxCurl * float(invDz) * AdjHxMidImag)
    AdjEyPrevReal[:, :, 1:].add_(HxCurl * float(invDz) * AdjHxMidReal)
    AdjEyPrevImag[:, :, 1:].add_(HxCurl * float(invDz) * AdjHxMidImag)
    AdjEyPrevReal[: HzMidReal.shape[0], :, :].add_(HzCurl * float(invDx) * AdjHzMidReal)
    AdjEyPrevImag[: HzMidImag.shape[0], :, :].add_(HzCurl * float(invDx) * AdjHzMidImag)
    AdjEyPrevReal[1:, :, :].sub_(HzCurl * float(invDx) * AdjHzMidReal)
    AdjEyPrevImag[1:, :, :].sub_(HzCurl * float(invDx) * AdjHzMidImag)


def _reverse_magnetic_ez_bloch(
    *,
    AdjEzPrevReal,
    AdjEzPrevImag,
    GradEpsEz,
    AdjEzPostReal,
    AdjEzPostImag,
    AdjHxMidReal,
    AdjHxMidImag,
    AdjHyMidReal,
    AdjHyMidImag,
    EzDecay,
    EzCurl,
    EpsEz,
    HxMidReal,
    HxMidImag,
    HyMidReal,
    HyMidImag,
    HxCurl,
    HyCurl,
    phaseCosX,
    phaseSinX,
    phaseCosY,
    phaseSinY,
    invDx,
    invDy,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().reverse_magnetic_adjoint_to_ez_bloch(
            AdjEzPrevReal, AdjEzPrevImag, GradEpsEz, AdjEzPostReal, AdjEzPostImag,
            AdjHxMidReal, AdjHxMidImag, AdjHyMidReal, AdjHyMidImag,
            EzDecay, EzCurl, EpsEz, HxMidReal, HxMidImag, HyMidReal, HyMidImag,
            HxCurl, HyCurl, float(phaseCosX), float(phaseSinX), float(phaseCosY), float(phaseSinY),
            float(invDx), float(invDy),
        )
        return
    d_hy_dx_r, d_hy_dx_i = _bloch_backward_diff(
        HyMidReal, HyMidImag, tuple(AdjEzPrevReal.shape), 0, phaseCosX, phaseSinX, invDx,
    )
    d_hx_dy_r, d_hx_dy_i = _bloch_backward_diff(
        HxMidReal, HxMidImag, tuple(AdjEzPrevReal.shape), 1, phaseCosY, phaseSinY, invDy,
    )
    curl_h_r = d_hy_dx_r - d_hx_dy_r
    curl_h_i = d_hy_dx_i - d_hx_dy_i
    AdjEzPrevReal.copy_(AdjEzPostReal * EzDecay)
    AdjEzPrevImag.copy_(AdjEzPostImag * EzDecay)
    GradEpsEz.copy_(-EzCurl * (AdjEzPostReal * curl_h_r + AdjEzPostImag * curl_h_i) / EpsEz)
    AdjEzPrevReal[:, : HxMidReal.shape[1], :].add_(HxCurl * float(invDy) * AdjHxMidReal)
    AdjEzPrevImag[:, : HxMidImag.shape[1], :].add_(HxCurl * float(invDy) * AdjHxMidImag)
    AdjEzPrevReal[:, 1:, :].sub_(HxCurl * float(invDy) * AdjHxMidReal)
    AdjEzPrevImag[:, 1:, :].sub_(HxCurl * float(invDy) * AdjHxMidImag)
    AdjEzPrevReal[: HyMidReal.shape[0], :, :].sub_(HyCurl * float(invDx) * AdjHyMidReal)
    AdjEzPrevImag[: HyMidImag.shape[0], :, :].sub_(HyCurl * float(invDx) * AdjHyMidImag)
    AdjEzPrevReal[1:, :, :].add_(HyCurl * float(invDx) * AdjHyMidReal)
    AdjEzPrevImag[1:, :, :].add_(HyCurl * float(invDx) * AdjHyMidImag)


def _accumulate_diff_adjoint(field_grad, diff_grad, axis, inv_delta, *, forward):
    if _use_compiled_field_kernels():
        method = (
            get_compiled_extension().accumulate_forward_diff_adjoint
            if forward
            else get_compiled_extension().accumulate_backward_diff_adjoint
        )
        method(field_grad, diff_grad, int(axis), float(inv_delta))
        return
    field_shape = tuple(int(size) for size in field_grad.shape)
    diff_shape = tuple(int(size) for size in diff_grad.shape)
    for i in range(field_shape[0]):
        for j in range(field_shape[1]):
            for k in range(field_shape[2]):
                coords = [i, j, k]
                value = field_grad.new_zeros(())
                if forward:
                    if coords[axis] < diff_shape[axis] and all(coords[d] < diff_shape[d] for d in range(3)):
                        value = value - float(inv_delta) * diff_grad[i, j, k]
                    if coords[axis] > 0:
                        prev = list(coords)
                        prev[axis] -= 1
                        if all(prev[d] < diff_shape[d] for d in range(3)):
                            value = value + float(inv_delta) * diff_grad[tuple(prev)]
                else:
                    if coords[axis] > 0 and all(coords[d] < diff_shape[d] for d in range(3)):
                        value = value + float(inv_delta) * diff_grad[i, j, k]
                    if coords[axis] + 1 < field_shape[axis]:
                        next_coords = list(coords)
                        next_coords[axis] += 1
                        if all(next_coords[d] < diff_shape[d] for d in range(3)):
                            value = value - float(inv_delta) * diff_grad[tuple(next_coords)]
                field_grad[i, j, k].add_(value)


def _accumulate_forward_diff_x(*, FieldGrad, DiffGrad, invDx):
    _accumulate_diff_adjoint(FieldGrad, DiffGrad, 0, invDx, forward=True)


def _accumulate_forward_diff_y(*, FieldGrad, DiffGrad, invDy):
    _accumulate_diff_adjoint(FieldGrad, DiffGrad, 1, invDy, forward=True)


def _accumulate_forward_diff_z(*, FieldGrad, DiffGrad, invDz):
    _accumulate_diff_adjoint(FieldGrad, DiffGrad, 2, invDz, forward=True)


def _accumulate_backward_diff_x(*, FieldGrad, DiffGrad, invDx):
    _accumulate_diff_adjoint(FieldGrad, DiffGrad, 0, invDx, forward=False)


def _accumulate_backward_diff_y(*, FieldGrad, DiffGrad, invDy):
    _accumulate_diff_adjoint(FieldGrad, DiffGrad, 1, invDy, forward=False)


def _accumulate_backward_diff_z(*, FieldGrad, DiffGrad, invDz):
    _accumulate_diff_adjoint(FieldGrad, DiffGrad, 2, invDz, forward=False)


def _reverse_electric_cpml_torch(
    component,
    adj_prev,
    grad_eps,
    adj_psi_pos_prev,
    adj_psi_neg_prev,
    adj_d_pos,
    adj_d_neg,
    adj_post,
    adj_psi_pos_post,
    adj_psi_neg_post,
    decay,
    curl,
    eps,
    psi_pos,
    psi_neg,
    b_pos,
    c_pos,
    inv_kappa_pos,
    b_neg,
    c_neg,
    inv_kappa_neg,
    h_pos_mid,
    h_neg_mid,
    inv_pos,
    inv_neg,
    low_mode_a,
    high_mode_a,
    low_mode_b,
    high_mode_b,
):
    nx, ny, nz = adj_prev.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if component == 0:
                    coord_a, size_a, coord_b, size_b = j, ny, k, nz
                    pos_idx, neg_idx = j, k
                elif component == 1:
                    coord_a, size_a, coord_b, size_b = i, nx, k, nz
                    pos_idx, neg_idx = k, i
                else:
                    coord_a, size_a, coord_b, size_b = i, nx, j, ny
                    pos_idx, neg_idx = i, j
                active, inactive, _ = _electric_cell_status(coord_a, size_a, low_mode_a, high_mode_a, coord_b, size_b, low_mode_b, high_mode_b)
                adjoint = adj_post.new_zeros(())
                grad = adj_post.new_zeros(())
                adj_psi_pos = adj_psi_pos_post[i, j, k]
                adj_psi_neg = adj_psi_neg_post[i, j, k]
                out_adj_d_pos = adj_post.new_zeros(())
                out_adj_d_neg = adj_post.new_zeros(())
                if inactive:
                    adjoint = adj_post[i, j, k]
                elif active:
                    if component == 0:
                        d_pos = (h_pos_mid[i, j, k] - h_pos_mid[i, j - 1, k]) * float(inv_pos)
                        d_neg = (h_neg_mid[i, j, k] - h_neg_mid[i, j, k - 1]) * float(inv_neg)
                    elif component == 1:
                        d_pos = (h_pos_mid[i, j, k] - h_pos_mid[i, j, k - 1]) * float(inv_pos)
                        d_neg = (h_neg_mid[i, j, k] - h_neg_mid[i - 1, j, k]) * float(inv_neg)
                    else:
                        d_pos = (h_pos_mid[i, j, k] - h_pos_mid[i - 1, j, k]) * float(inv_pos)
                        d_neg = (h_neg_mid[i, j, k] - h_neg_mid[i, j - 1, k]) * float(inv_neg)
                    psi_pos_candidate = b_pos[pos_idx] * psi_pos[i, j, k] + c_pos[pos_idx] * d_pos
                    psi_neg_candidate = b_neg[neg_idx] * psi_neg[i, j, k] + c_neg[neg_idx] * d_neg
                    curl_h = (d_pos * inv_kappa_pos[pos_idx] + psi_pos_candidate) - (d_neg * inv_kappa_neg[neg_idx] + psi_neg_candidate)
                    adj_curl_h = adj_post[i, j, k] * curl[i, j, k]
                    adjoint = adj_post[i, j, k] * decay[i, j, k]
                    grad = -adj_post[i, j, k] * curl[i, j, k] * curl_h / eps[i, j, k]
                    adj_psi_pos = b_pos[pos_idx] * (adj_psi_pos_post[i, j, k] + adj_curl_h)
                    adj_psi_neg = b_neg[neg_idx] * (adj_psi_neg_post[i, j, k] - adj_curl_h)
                    out_adj_d_pos = inv_kappa_pos[pos_idx] * adj_curl_h + c_pos[pos_idx] * (adj_psi_pos_post[i, j, k] + adj_curl_h)
                    out_adj_d_neg = -inv_kappa_neg[neg_idx] * adj_curl_h + c_neg[neg_idx] * (adj_psi_neg_post[i, j, k] - adj_curl_h)
                adj_prev[i, j, k] = adjoint
                grad_eps[i, j, k] = grad
                adj_psi_pos_prev[i, j, k] = adj_psi_pos
                adj_psi_neg_prev[i, j, k] = adj_psi_neg
                adj_d_pos[i, j, k] = out_adj_d_pos
                adj_d_neg[i, j, k] = out_adj_d_neg


def _reverse_magnetic_cpml_torch(
    component,
    adj_prev,
    adj_psi_pos_prev,
    adj_psi_neg_prev,
    adj_d_pos,
    adj_d_neg,
    adj_post,
    adj_psi_pos_post,
    adj_psi_neg_post,
    decay,
    curl,
    b_pos,
    c_pos,
    inv_kappa_pos,
    b_neg,
    c_neg,
    inv_kappa_neg,
):
    nx, ny, nz = adj_prev.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if component == 0:
                    pos_idx, neg_idx = j, k
                elif component == 1:
                    pos_idx, neg_idx = k, i
                else:
                    pos_idx, neg_idx = i, j
                adj_curl_e = -curl[i, j, k] * adj_post[i, j, k]
                adj_psi_pos_candidate = adj_psi_pos_post[i, j, k] + adj_curl_e
                adj_psi_neg_candidate = adj_psi_neg_post[i, j, k] - adj_curl_e
                adj_prev[i, j, k] = adj_post[i, j, k] * decay[i, j, k]
                adj_psi_pos_prev[i, j, k] = b_pos[pos_idx] * adj_psi_pos_candidate
                adj_psi_neg_prev[i, j, k] = b_neg[neg_idx] * adj_psi_neg_candidate
                adj_d_pos[i, j, k] = inv_kappa_pos[pos_idx] * adj_curl_e + c_pos[pos_idx] * adj_psi_pos_candidate
                adj_d_neg[i, j, k] = -inv_kappa_neg[neg_idx] * adj_curl_e + c_neg[neg_idx] * adj_psi_neg_candidate


def _reverse_electric_cpml_ex(
    *,
    AdjExPrev,
    GradEpsEx,
    AdjPsiPosPrev,
    AdjPsiNegPrev,
    AdjDPos,
    AdjDNeg,
    AdjExPost,
    AdjPsiPosPost,
    AdjPsiNegPost,
    ExDecay,
    ExCurl,
    EpsEx,
    PsiPos,
    PsiNeg,
    BPos,
    CPos,
    InvKappaPos,
    BNeg,
    CNeg,
    InvKappaNeg,
    HyMid,
    HzMid,
    invDy,
    invDz,
    yLowBoundaryMode,
    yHighBoundaryMode,
    zLowBoundaryMode,
    zHighBoundaryMode,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().reverse_electric_component_ex_cpml(
            AdjExPrev, GradEpsEx, AdjPsiPosPrev, AdjPsiNegPrev, AdjDPos, AdjDNeg,
            AdjExPost, AdjPsiPosPost, AdjPsiNegPost, ExDecay, ExCurl, EpsEx,
            PsiPos, PsiNeg, BPos, CPos, InvKappaPos, BNeg, CNeg, InvKappaNeg,
            HyMid, HzMid, float(invDy), float(invDz), int(yLowBoundaryMode), int(yHighBoundaryMode), int(zLowBoundaryMode), int(zHighBoundaryMode),
        )
        return
    _reverse_electric_cpml_torch(
        0, AdjExPrev, GradEpsEx, AdjPsiPosPrev, AdjPsiNegPrev, AdjDPos, AdjDNeg,
        AdjExPost, AdjPsiPosPost, AdjPsiNegPost, ExDecay, ExCurl, EpsEx, PsiPos, PsiNeg,
        BPos, CPos, InvKappaPos, BNeg, CNeg, InvKappaNeg, HzMid, HyMid, invDy, invDz,
        yLowBoundaryMode, yHighBoundaryMode, zLowBoundaryMode, zHighBoundaryMode,
    )


def _reverse_electric_cpml_ey(
    *,
    AdjEyPrev,
    GradEpsEy,
    AdjPsiPosPrev,
    AdjPsiNegPrev,
    AdjDPos,
    AdjDNeg,
    AdjEyPost,
    AdjPsiPosPost,
    AdjPsiNegPost,
    EyDecay,
    EyCurl,
    EpsEy,
    PsiPos,
    PsiNeg,
    BPos,
    CPos,
    InvKappaPos,
    BNeg,
    CNeg,
    InvKappaNeg,
    HxMid,
    HzMid,
    invDx,
    invDz,
    xLowBoundaryMode,
    xHighBoundaryMode,
    zLowBoundaryMode,
    zHighBoundaryMode,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().reverse_electric_component_ey_cpml(
            AdjEyPrev, GradEpsEy, AdjPsiPosPrev, AdjPsiNegPrev, AdjDPos, AdjDNeg,
            AdjEyPost, AdjPsiPosPost, AdjPsiNegPost, EyDecay, EyCurl, EpsEy,
            PsiPos, PsiNeg, BPos, CPos, InvKappaPos, BNeg, CNeg, InvKappaNeg,
            HxMid, HzMid, float(invDx), float(invDz), int(xLowBoundaryMode), int(xHighBoundaryMode), int(zLowBoundaryMode), int(zHighBoundaryMode),
        )
        return
    _reverse_electric_cpml_torch(
        1, AdjEyPrev, GradEpsEy, AdjPsiPosPrev, AdjPsiNegPrev, AdjDPos, AdjDNeg,
        AdjEyPost, AdjPsiPosPost, AdjPsiNegPost, EyDecay, EyCurl, EpsEy, PsiPos, PsiNeg,
        BPos, CPos, InvKappaPos, BNeg, CNeg, InvKappaNeg, HxMid, HzMid, invDz, invDx,
        xLowBoundaryMode, xHighBoundaryMode, zLowBoundaryMode, zHighBoundaryMode,
    )


def _reverse_electric_cpml_ez(
    *,
    AdjEzPrev,
    GradEpsEz,
    AdjPsiPosPrev,
    AdjPsiNegPrev,
    AdjDPos,
    AdjDNeg,
    AdjEzPost,
    AdjPsiPosPost,
    AdjPsiNegPost,
    EzDecay,
    EzCurl,
    EpsEz,
    PsiPos,
    PsiNeg,
    BPos,
    CPos,
    InvKappaPos,
    BNeg,
    CNeg,
    InvKappaNeg,
    HxMid,
    HyMid,
    invDx,
    invDy,
    xLowBoundaryMode,
    xHighBoundaryMode,
    yLowBoundaryMode,
    yHighBoundaryMode,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().reverse_electric_component_ez_cpml(
            AdjEzPrev, GradEpsEz, AdjPsiPosPrev, AdjPsiNegPrev, AdjDPos, AdjDNeg,
            AdjEzPost, AdjPsiPosPost, AdjPsiNegPost, EzDecay, EzCurl, EpsEz,
            PsiPos, PsiNeg, BPos, CPos, InvKappaPos, BNeg, CNeg, InvKappaNeg,
            HxMid, HyMid, float(invDx), float(invDy), int(xLowBoundaryMode), int(xHighBoundaryMode), int(yLowBoundaryMode), int(yHighBoundaryMode),
        )
        return
    _reverse_electric_cpml_torch(
        2, AdjEzPrev, GradEpsEz, AdjPsiPosPrev, AdjPsiNegPrev, AdjDPos, AdjDNeg,
        AdjEzPost, AdjPsiPosPost, AdjPsiNegPost, EzDecay, EzCurl, EpsEz, PsiPos, PsiNeg,
        BPos, CPos, InvKappaPos, BNeg, CNeg, InvKappaNeg, HyMid, HxMid, invDx, invDy,
        xLowBoundaryMode, xHighBoundaryMode, yLowBoundaryMode, yHighBoundaryMode,
    )


def _reverse_magnetic_cpml_hx(**kwargs):
    if _use_compiled_field_kernels():
        get_compiled_extension().reverse_magnetic_component_hx_cpml(
            kwargs["AdjHxPrev"], kwargs["AdjPsiPosPrev"], kwargs["AdjPsiNegPrev"], kwargs["AdjDPos"], kwargs["AdjDNeg"],
            kwargs["AdjHxPost"], kwargs["AdjPsiPosPost"], kwargs["AdjPsiNegPost"], kwargs["HxDecay"], kwargs["HxCurl"],
            kwargs["BPos"], kwargs["CPos"], kwargs["InvKappaPos"], kwargs["BNeg"], kwargs["CNeg"], kwargs["InvKappaNeg"],
        )
        return
    _reverse_magnetic_cpml_torch(
        0, kwargs["AdjHxPrev"], kwargs["AdjPsiPosPrev"], kwargs["AdjPsiNegPrev"], kwargs["AdjDPos"], kwargs["AdjDNeg"],
        kwargs["AdjHxPost"], kwargs["AdjPsiPosPost"], kwargs["AdjPsiNegPost"], kwargs["HxDecay"], kwargs["HxCurl"],
        kwargs["BPos"], kwargs["CPos"], kwargs["InvKappaPos"], kwargs["BNeg"], kwargs["CNeg"], kwargs["InvKappaNeg"],
    )


def _reverse_magnetic_cpml_hy(**kwargs):
    if _use_compiled_field_kernels():
        get_compiled_extension().reverse_magnetic_component_hy_cpml(
            kwargs["AdjHyPrev"], kwargs["AdjPsiPosPrev"], kwargs["AdjPsiNegPrev"], kwargs["AdjDPos"], kwargs["AdjDNeg"],
            kwargs["AdjHyPost"], kwargs["AdjPsiPosPost"], kwargs["AdjPsiNegPost"], kwargs["HyDecay"], kwargs["HyCurl"],
            kwargs["BPos"], kwargs["CPos"], kwargs["InvKappaPos"], kwargs["BNeg"], kwargs["CNeg"], kwargs["InvKappaNeg"],
        )
        return
    _reverse_magnetic_cpml_torch(
        1, kwargs["AdjHyPrev"], kwargs["AdjPsiPosPrev"], kwargs["AdjPsiNegPrev"], kwargs["AdjDPos"], kwargs["AdjDNeg"],
        kwargs["AdjHyPost"], kwargs["AdjPsiPosPost"], kwargs["AdjPsiNegPost"], kwargs["HyDecay"], kwargs["HyCurl"],
        kwargs["BPos"], kwargs["CPos"], kwargs["InvKappaPos"], kwargs["BNeg"], kwargs["CNeg"], kwargs["InvKappaNeg"],
    )


def _reverse_magnetic_cpml_hz(**kwargs):
    if _use_compiled_field_kernels():
        get_compiled_extension().reverse_magnetic_component_hz_cpml(
            kwargs["AdjHzPrev"], kwargs["AdjPsiPosPrev"], kwargs["AdjPsiNegPrev"], kwargs["AdjDPos"], kwargs["AdjDNeg"],
            kwargs["AdjHzPost"], kwargs["AdjPsiPosPost"], kwargs["AdjPsiNegPost"], kwargs["HzDecay"], kwargs["HzCurl"],
            kwargs["BPos"], kwargs["CPos"], kwargs["InvKappaPos"], kwargs["BNeg"], kwargs["CNeg"], kwargs["InvKappaNeg"],
        )
        return
    _reverse_magnetic_cpml_torch(
        2, kwargs["AdjHzPrev"], kwargs["AdjPsiPosPrev"], kwargs["AdjPsiNegPrev"], kwargs["AdjDPos"], kwargs["AdjDNeg"],
        kwargs["AdjHzPost"], kwargs["AdjPsiPosPost"], kwargs["AdjPsiNegPost"], kwargs["HzDecay"], kwargs["HzCurl"],
        kwargs["BPos"], kwargs["CPos"], kwargs["InvKappaPos"], kwargs["BNeg"], kwargs["CNeg"], kwargs["InvKappaNeg"],
    )


def _reverse_magnetic_hx_decay(*, AdjHxPrev, AdjHxMid, HxDecay):
    if _use_compiled_field_kernels():
        get_compiled_extension().reverse_magnetic_adjoint_decay(AdjHxPrev, AdjHxMid, HxDecay)
        return
    AdjHxPrev.copy_(AdjHxMid * HxDecay)


def _reverse_magnetic_hy_decay(*, AdjHyPrev, AdjHyMid, HyDecay):
    if _use_compiled_field_kernels():
        get_compiled_extension().reverse_magnetic_adjoint_decay(AdjHyPrev, AdjHyMid, HyDecay)
        return
    AdjHyPrev.copy_(AdjHyMid * HyDecay)


def _reverse_magnetic_hz_decay(*, AdjHzPrev, AdjHzMid, HzDecay):
    if _use_compiled_field_kernels():
        get_compiled_extension().reverse_magnetic_adjoint_decay(AdjHzPrev, AdjHzMid, HzDecay)
        return
    AdjHzPrev.copy_(AdjHzMid * HzDecay)


def _reverse_debye_current(
    *,
    AdjElectricPrev,
    AdjPolarizationPrev,
    AdjPolarizationPost,
    AdjCurrentPost,
    DebyeDrive,
    decay,
    dt,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().reverse_debye_current(
            AdjElectricPrev,
            AdjPolarizationPrev,
            AdjPolarizationPost,
            AdjCurrentPost,
            DebyeDrive,
            float(decay),
            float(dt),
        )
        return
    adj_internal = AdjPolarizationPost + AdjCurrentPost / float(dt)
    AdjElectricPrev.add_(DebyeDrive * adj_internal)
    AdjPolarizationPrev.add_(float(decay) * adj_internal - AdjCurrentPost / float(dt))


def _reverse_drude_current(*, AdjElectricPrev, AdjCurrentPrev, AdjCurrentPost, DrudeDrive, decay):
    if _use_compiled_field_kernels():
        get_compiled_extension().reverse_drude_current(
            AdjElectricPrev,
            AdjCurrentPrev,
            AdjCurrentPost,
            DrudeDrive,
            float(decay),
        )
        return
    AdjElectricPrev.add_(DrudeDrive * AdjCurrentPost)
    AdjCurrentPrev.add_(float(decay) * AdjCurrentPost)


def _reverse_lorentz_current(
    *,
    AdjElectricPrev,
    AdjPolarizationPrev,
    AdjCurrentPrev,
    AdjPolarizationPost,
    AdjCurrentPost,
    LorentzDrive,
    decay,
    restoring,
    dt,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().reverse_lorentz_current(
            AdjElectricPrev,
            AdjPolarizationPrev,
            AdjCurrentPrev,
            AdjPolarizationPost,
            AdjCurrentPost,
            LorentzDrive,
            float(decay),
            float(restoring),
            float(dt),
        )
        return
    adj_internal = AdjCurrentPost + float(dt) * AdjPolarizationPost
    AdjElectricPrev.add_(LorentzDrive * adj_internal)
    AdjPolarizationPrev.add_(AdjPolarizationPost - float(restoring) * adj_internal)
    AdjCurrentPrev.add_(float(decay) * adj_internal)


def _accumulate_tfsf_scalar_sample_adjoint(*, AdjAuxField, AdjFieldPatch, CoeffPatch, sampleIndex, componentScale):
    if _use_compiled_field_kernels():
        get_compiled_extension().accumulate_tfsf_scalar_sample_adjoint(
            AdjAuxField,
            AdjFieldPatch,
            CoeffPatch,
            int(sampleIndex),
            float(componentScale),
        )
        return
    AdjAuxField[int(sampleIndex)].add_(float(componentScale) * torch.sum(AdjFieldPatch * CoeffPatch))


def _accumulate_tfsf_line_sample_adjoint(
    *,
    AdjAuxField,
    AdjFieldPatch,
    CoeffPatch,
    SampleIndices,
    sampleAxisCode,
    componentScale,
):
    sample_indices = SampleIndices.to(device=AdjAuxField.device, dtype=torch.int32).contiguous()
    if _use_compiled_field_kernels():
        get_compiled_extension().accumulate_tfsf_line_sample_adjoint(
            AdjAuxField,
            AdjFieldPatch,
            CoeffPatch,
            sample_indices,
            int(sampleAxisCode),
            float(componentScale),
        )
        return
    axis = int(sampleAxisCode)
    weighted = (float(componentScale) * AdjFieldPatch * CoeffPatch).movedim(axis, 0).reshape(sample_indices.numel(), -1)
    AdjAuxField.index_add_(0, sample_indices.to(dtype=torch.long), weighted.sum(dim=1))


def _accumulate_tfsf_interpolated_sample_adjoint(
    *,
    AdjAuxField,
    AdjFieldPatch,
    CoeffPatch,
    SamplePositions,
    origin,
    ds,
    componentScale,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().accumulate_tfsf_interpolated_sample_adjoint(
            AdjAuxField,
            AdjFieldPatch,
            CoeffPatch,
            SamplePositions,
            float(origin),
            float(ds),
            float(componentScale),
        )
        return
    if AdjAuxField.numel() == 0:
        return
    coord = (SamplePositions.to(device=AdjAuxField.device, dtype=AdjAuxField.dtype) - float(origin)) / float(ds) if float(ds) > 0.0 else torch.zeros_like(SamplePositions)
    coord = torch.clamp(coord, min=0.0, max=max(AdjAuxField.numel() - 1, 0))
    lower = torch.floor(coord).to(dtype=torch.long).reshape(-1)
    upper = torch.clamp(lower + 1, max=max(AdjAuxField.numel() - 1, 0))
    frac = (coord.reshape(-1) - lower.to(dtype=AdjAuxField.dtype))
    value = (float(componentScale) * AdjFieldPatch * CoeffPatch).reshape(-1)
    AdjAuxField.index_add_(0, lower, value * (1.0 - frac))
    upper_weight = value * frac
    if upper_weight.numel() > 0:
        AdjAuxField.index_add_(0, upper, upper_weight)


def _reverse_tfsf_auxiliary_electric(
    *,
    AdjElectricPrev,
    AdjMagneticAfter,
    AdjElectricPost,
    ElectricDecay,
    ElectricCurl,
    sourceIndex,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().reverse_tfsf_auxiliary_electric(
            AdjElectricPrev,
            AdjMagneticAfter,
            AdjElectricPost,
            ElectricDecay,
            ElectricCurl,
            int(sourceIndex),
        )
        return
    overwritten = torch.zeros_like(AdjElectricPost, dtype=torch.bool)
    if 0 <= int(sourceIndex) < AdjElectricPost.numel():
        overwritten[int(sourceIndex)] = True
    if AdjElectricPost.numel() > 0:
        overwritten[-1] = True
        if not bool(overwritten[0].item()):
            AdjElectricPrev[0].add_(AdjElectricPost[0])
    if AdjElectricPost.numel() > 2:
        indices = torch.arange(1, AdjElectricPost.numel() - 1, device=AdjElectricPost.device)
        active = indices[~overwritten[indices]]
        if active.numel() > 0:
            adjoint = AdjElectricPost[active]
            AdjElectricPrev[active].add_(ElectricDecay[active] * adjoint)
            values = ElectricCurl[active] * adjoint
            AdjMagneticAfter.index_add_(0, active - 1, values)
            AdjMagneticAfter.index_add_(0, active, -values)


def _reverse_tfsf_auxiliary_magnetic(
    *,
    AdjElectricPrev,
    AdjMagneticPrev,
    AdjMagneticAfter,
    MagneticDecay,
    MagneticCurl,
):
    if _use_compiled_field_kernels():
        get_compiled_extension().reverse_tfsf_auxiliary_magnetic(
            AdjElectricPrev,
            AdjMagneticPrev,
            AdjMagneticAfter,
            MagneticDecay,
            MagneticCurl,
        )
        return
    AdjMagneticPrev.copy_(MagneticDecay * AdjMagneticAfter)
    values = MagneticCurl * AdjMagneticAfter
    indices = torch.arange(AdjMagneticAfter.numel(), device=AdjMagneticAfter.device)
    AdjElectricPrev.index_add_(0, indices, values)
    AdjElectricPrev.index_add_(0, indices + 1, -values)


def _clamp_field_face(*, field, axis, side):
    if _use_compiled_field_kernels():
        get_compiled_extension().clamp_field_face(field, int(axis), int(side))
        return
    index = 0 if int(side) == 0 else field.shape[int(axis)] - 1
    field.select(int(axis), index).zero_()


def _clamp_pec_boundary(*, field, axisA, axisB):
    if _use_compiled_field_kernels():
        get_compiled_extension().clamp_pec_boundary(field, int(axisA), int(axisB))
        return
    mask = torch.zeros_like(field, dtype=torch.bool)
    for axis in (int(axisA), int(axisB)):
        low = [slice(None)] * 3
        low[axis] = 0
        high = [slice(None)] * 3
        high[axis] = field.shape[axis] - 1
        mask[tuple(low)] = True
        mask[tuple(high)] = True
    field.masked_fill_(mask, 0.0)


def _project_periodic_boundary(*, field, axis):
    if _use_compiled_field_kernels():
        get_compiled_extension().project_periodic_boundary(field, int(axis))
        return
    axis = int(axis)
    low = [slice(None)] * 3
    high = [slice(None)] * 3
    low[axis] = 0
    high[axis] = field.shape[axis] - 1
    average = 0.5 * (field[tuple(low)] + field[tuple(high)])
    field[tuple(low)].copy_(average)
    field[tuple(high)].copy_(average)


def _project_bloch_boundary(*, fieldReal, fieldImag, axis, phaseCos, phaseSin):
    if _use_compiled_field_kernels():
        get_compiled_extension().project_bloch_boundary(
            fieldReal,
            fieldImag,
            int(axis),
            float(phaseCos),
            float(phaseSin),
        )
        return
    axis = int(axis)
    low = [slice(None)] * 3
    high = [slice(None)] * 3
    low[axis] = 0
    high[axis] = fieldReal.shape[axis] - 1
    projected_low_r, projected_low_i = _phase_negative(fieldReal[tuple(high)], fieldImag[tuple(high)], phaseCos, phaseSin)
    projected_low_r = 0.5 * (fieldReal[tuple(low)] + projected_low_r)
    projected_low_i = 0.5 * (fieldImag[tuple(low)] + projected_low_i)
    projected_high_r, projected_high_i = _phase_positive(projected_low_r, projected_low_i, phaseCos, phaseSin)
    fieldReal[tuple(low)].copy_(projected_low_r)
    fieldImag[tuple(low)].copy_(projected_low_i)
    fieldReal[tuple(high)].copy_(projected_high_r)
    fieldImag[tuple(high)].copy_(projected_high_i)


_KERNELS: dict[str, Callable[..., None]] = {
    "updateMagneticFieldHxStandard3D": _magnetic_hx_standard,
    "updateMagneticFieldHyStandard3D": _magnetic_hy_standard,
    "updateMagneticFieldHzStandard3D": _magnetic_hz_standard,
    "updateMagneticFieldHx3D": _magnetic_hx_cpml,
    "updateMagneticFieldHy3D": _magnetic_hy_cpml,
    "updateMagneticFieldHz3D": _magnetic_hz_cpml,
    "updateMagneticFieldHxCpmlCompressed3D": _magnetic_hx_cpml_compressed,
    "updateMagneticFieldHyCpmlCompressed3D": _magnetic_hy_cpml_compressed,
    "updateMagneticFieldHzCpmlCompressed3D": _magnetic_hz_cpml_compressed,
    "updateElectricFieldExStandard3D": _electric_ex_standard,
    "updateElectricFieldEyStandard3D": _electric_ey_standard,
    "updateElectricFieldEzStandard3D": _electric_ez_standard,
    "updateElectricFieldExCpml3D": _electric_ex_cpml,
    "updateElectricFieldEyCpml3D": _electric_ey_cpml,
    "updateElectricFieldEzCpml3D": _electric_ez_cpml,
    "updateElectricFieldExCpmlCompressed3D": _electric_ex_cpml_compressed,
    "updateElectricFieldEyCpmlCompressed3D": _electric_ey_cpml_compressed,
    "updateElectricFieldEzCpmlCompressed3D": _electric_ez_cpml_compressed,
    "updateElectricFieldExBloch3D": _electric_ex_bloch,
    "updateElectricFieldEyBloch3D": _electric_ey_bloch,
    "updateElectricFieldEzBloch3D": _electric_ez_bloch,
    "addSourcePatch3D": _add_source_patch,
    "addCwPhasedSourcePatch3D": _add_cw_phased_source_patch,
    "addTimeShiftedSourcePatch3D": _add_time_shifted_source_patch,
    "addSourcePatchExPeriodic3D": _add_source_patch_periodic,
    "addSourcePatchEyPeriodic3D": _add_source_patch_periodic,
    "addSourcePatchEzPeriodic3D": _add_source_patch_periodic,
    "addSourcePatchBloch3D": _add_source_patch_bloch,
    "addScaledSliceSourcePatch3D": _add_scaled_slice_source_patch,
    "addScaledLineSourcePatch3D": _add_scaled_line_source_patch,
    "addInterpolatedSourcePatch3D": _add_interpolated_source_patch,
    "addBatchedReferenceSourcePatches3D": _add_batched_reference_source_patches,
    "addBatchedInterpolatedSourcePatches3D": _add_batched_interpolated_source_patches,
    "updateAuxiliaryMagnetic1D": _update_auxiliary_magnetic,
    "updateAuxiliaryElectric1D": _update_auxiliary_electric,
    "accumulateRunningDftYee3DBatched": _accumulate_dft_batched,
    "accumulatePointObservers3D": _accumulate_point_observers,
    "accumulatePlaneObserver3D": _accumulate_plane_observer,
    "updateDebyeCurrent3D": _update_debye_current,
    "updateDrudeCurrent3D": _update_drude_current,
    "updateLorentzCurrent3D": _update_lorentz_current,
    "applyPolarizationCurrent3D": _apply_polarization_current,
    "updateKerrElectricFieldExCurl3D": _update_kerr_ex,
    "updateKerrElectricFieldEyCurl3D": _update_kerr_ey,
    "updateKerrElectricFieldEzCurl3D": _update_kerr_ez,
    "reverseElectricAdjointToHxStandard3D": _reverse_electric_hx_standard,
    "reverseElectricAdjointToHyStandard3D": _reverse_electric_hy_standard,
    "reverseElectricAdjointToHzStandard3D": _reverse_electric_hz_standard,
    "reverseMagneticAdjointToExStandard3D": _reverse_magnetic_ex_standard,
    "reverseMagneticAdjointToEyStandard3D": _reverse_magnetic_ey_standard,
    "reverseMagneticAdjointToEzStandard3D": _reverse_magnetic_ez_standard,
    "reverseMagneticAdjointToHxStandard3D": _reverse_magnetic_hx_decay,
    "reverseMagneticAdjointToHyStandard3D": _reverse_magnetic_hy_decay,
    "reverseMagneticAdjointToHzStandard3D": _reverse_magnetic_hz_decay,
    "reverseElectricAdjointToHxBloch3D": _reverse_electric_hx_bloch,
    "reverseElectricAdjointToHyBloch3D": _reverse_electric_hy_bloch,
    "reverseElectricAdjointToHzBloch3D": _reverse_electric_hz_bloch,
    "reverseMagneticAdjointToExBloch3D": _reverse_magnetic_ex_bloch,
    "reverseMagneticAdjointToEyBloch3D": _reverse_magnetic_ey_bloch,
    "reverseMagneticAdjointToEzBloch3D": _reverse_magnetic_ez_bloch,
    "accumulateForwardDiffAdjointX3D": _accumulate_forward_diff_x,
    "accumulateForwardDiffAdjointY3D": _accumulate_forward_diff_y,
    "accumulateForwardDiffAdjointZ3D": _accumulate_forward_diff_z,
    "accumulateBackwardDiffAdjointX3D": _accumulate_backward_diff_x,
    "accumulateBackwardDiffAdjointY3D": _accumulate_backward_diff_y,
    "accumulateBackwardDiffAdjointZ3D": _accumulate_backward_diff_z,
    "reverseElectricComponentExCpml3D": _reverse_electric_cpml_ex,
    "reverseElectricComponentEyCpml3D": _reverse_electric_cpml_ey,
    "reverseElectricComponentEzCpml3D": _reverse_electric_cpml_ez,
    "reverseMagneticComponentHxCpml3D": _reverse_magnetic_cpml_hx,
    "reverseMagneticComponentHyCpml3D": _reverse_magnetic_cpml_hy,
    "reverseMagneticComponentHzCpml3D": _reverse_magnetic_cpml_hz,
    "reverseDebyeCurrent3D": _reverse_debye_current,
    "reverseDrudeCurrent3D": _reverse_drude_current,
    "reverseLorentzCurrent3D": _reverse_lorentz_current,
    "accumulateTfsfScalarSampleAdjoint3D": _accumulate_tfsf_scalar_sample_adjoint,
    "accumulateTfsfLineSampleAdjoint3D": _accumulate_tfsf_line_sample_adjoint,
    "accumulateTfsfInterpolatedSampleAdjoint3D": _accumulate_tfsf_interpolated_sample_adjoint,
    "reverseTfsfAuxiliaryElectric1D": _reverse_tfsf_auxiliary_electric,
    "reverseTfsfAuxiliaryMagnetic1D": _reverse_tfsf_auxiliary_magnetic,
    "clampFieldFace3D": _clamp_field_face,
    "clampPecBoundary3D": _clamp_pec_boundary,
    "projectPeriodicBoundary3D": _project_periodic_boundary,
    "projectBlochBoundary3D": _project_bloch_boundary,
}
