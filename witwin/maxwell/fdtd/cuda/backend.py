from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import Any

import torch


_COMPILED_EXTENSION: Any | None = None


def is_available() -> bool:
    return bool(torch.cuda.is_available())


def get_compiled_extension(*, verbose: bool = False) -> Any:
    global _COMPILED_EXTENSION
    if _COMPILED_EXTENSION is None:
        from .build import build_extension

        _COMPILED_EXTENSION = build_extension(verbose=verbose)
    return _COMPILED_EXTENSION


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
        with torch.cuda.device(device):
            return get_compiled_extension().debug_linear_indices([int(value) for value in shape])
    total = int(shape[0]) * int(shape[1]) * int(shape[2])
    linear = torch.arange(total, device=device, dtype=torch.int64).reshape(shape)
    i_index = linear // (int(shape[1]) * int(shape[2]))
    j_index = (linear // int(shape[2])) % int(shape[1])
    k_index = linear % int(shape[2])
    return linear, i_index, j_index, k_index


def _call_with_contiguous(kernel: Callable[..., None], kwargs: dict) -> None:
    # Callers may pass strided tensor views (the adjoint replay passes box
    # slices of adjoint fields and real/imag views), while the compiled kernels
    # require contiguous memory. Materialize contiguous copies and write back
    # afterwards so in-place semantics are preserved.
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

    def launchRaw(self, *, blockSize=None, gridSize=None):  # noqa: N802 - kernel launch entry point
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


def _magnetic_hx_standard(*, Hx, Ey, Ez, HxDecay, HxCurl, invDy, invDz):
    _require_cuda_tensors(Hx, Ey, Ez, HxDecay, HxCurl, invDy, invDz)
    get_compiled_extension().update_magnetic_hx_standard(Hx, Ey, Ez, HxDecay, HxCurl, invDy, invDz)


def _magnetic_hy_standard(*, Hy, Ex, Ez, HyDecay, HyCurl, invDx, invDz):
    _require_cuda_tensors(Hy, Ex, Ez, HyDecay, HyCurl, invDx, invDz)
    get_compiled_extension().update_magnetic_hy_standard(Hy, Ex, Ez, HyDecay, HyCurl, invDx, invDz)


def _magnetic_hz_standard(*, Hz, Ex, Ey, HzDecay, HzCurl, invDx, invDy):
    _require_cuda_tensors(Hz, Ex, Ey, HzDecay, HzCurl, invDx, invDy)
    get_compiled_extension().update_magnetic_hz_standard(Hz, Ex, Ey, HzDecay, HzCurl, invDx, invDy)


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
        invDy,
        invDz,
    )
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
                invDy,
                invDz,
            )


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
        invDx,
        invDz,
    )
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
                invDx,
                invDz,
            )


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
        invDx,
        invDy,
    )
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
                invDx,
                invDy,
            )


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
    _require_cuda_tensors(Hx, Ey, Ez, PsiHxY, PsiHxZ, invDy, invDz)
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
                invDy,
                invDz,
                int(psiHxYLowLength),
                int(psiHxYHighStart),
                int(psiHxYHighLength),
                int(psiHxZLowLength),
                int(psiHxZHighStart),
                int(psiHxZHighLength),
                _uniform_scalar(HxDecay),
                _uniform_scalar(HxCurl),
            )


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
    _require_cuda_tensors(Hy, Ex, Ez, PsiHyX, PsiHyZ, invDx, invDz)
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
                invDx,
                invDz,
                int(psiHyXLowLength),
                int(psiHyXHighStart),
                int(psiHyXHighLength),
                int(psiHyZLowLength),
                int(psiHyZHighStart),
                int(psiHyZHighLength),
                _uniform_scalar(HyDecay),
                _uniform_scalar(HyCurl),
            )


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
    _require_cuda_tensors(Hz, Ex, Ey, PsiHzX, PsiHzY, invDx, invDy)
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
                invDx,
                invDy,
                int(psiHzXLowLength),
                int(psiHzXHighStart),
                int(psiHzXHighLength),
                int(psiHzYLowLength),
                int(psiHzYHighStart),
                int(psiHzYHighLength),
                _uniform_scalar(HzDecay),
                _uniform_scalar(HzCurl),
            )


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
    _require_cuda_tensors(Ex, Hy, Hz, ExDecay, ExCurl, invDy, invDz)
    get_compiled_extension().update_electric_ex_standard(
                Ex,
                Hy,
                Hz,
                ExDecay,
                ExCurl,
                invDy,
                invDz,
                int(yLowBoundaryMode),
                int(yHighBoundaryMode),
                int(zLowBoundaryMode),
                int(zHighBoundaryMode),
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
    _require_cuda_tensors(Ey, Hx, Hz, EyDecay, EyCurl, invDx, invDz)
    get_compiled_extension().update_electric_ey_standard(
                Ey,
                Hx,
                Hz,
                EyDecay,
                EyCurl,
                invDx,
                invDz,
                int(xLowBoundaryMode),
                int(xHighBoundaryMode),
                int(zLowBoundaryMode),
                int(zHighBoundaryMode),
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
    _require_cuda_tensors(Ez, Hx, Hy, EzDecay, EzCurl, invDx, invDy)
    get_compiled_extension().update_electric_ez_standard(
                Ez,
                Hx,
                Hy,
                EzDecay,
                EzCurl,
                invDx,
                invDy,
                int(xLowBoundaryMode),
                int(xHighBoundaryMode),
                int(yLowBoundaryMode),
                int(yHighBoundaryMode),
            )


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
        invDy,
        invDz,
    )
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
                invDy,
                invDz,
                int(yLowBoundaryMode),
                int(yHighBoundaryMode),
                int(zLowBoundaryMode),
                int(zHighBoundaryMode),
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
        invDx,
        invDz,
    )
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
                invDx,
                invDz,
                int(xLowBoundaryMode),
                int(xHighBoundaryMode),
                int(zLowBoundaryMode),
                int(zHighBoundaryMode),
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
        invDx,
        invDy,
    )
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
                invDx,
                invDy,
                int(xLowBoundaryMode),
                int(xHighBoundaryMode),
                int(yLowBoundaryMode),
                int(yHighBoundaryMode),
            )


def _apply_electric_ex_cpml_z_correction(
    *,
    Ex,
    Hy,
    ExCurl,
    PsiExZ,
    InvKappaExZ,
    BExZ,
    CExZ,
    invDz,
    offsetI,
    offsetJ,
    offsetK,
    yLowBoundaryMode,
    yHighBoundaryMode,
    fullSizeY,
    fullSizeZ,
):
    _require_cuda_tensors(Ex, Hy, ExCurl, PsiExZ, InvKappaExZ, BExZ, CExZ, invDz)
    get_compiled_extension().apply_electric_ex_cpml_z_correction(
                Ex, Hy, ExCurl, PsiExZ, InvKappaExZ, BExZ, CExZ, invDz,
                int(offsetI), int(offsetJ), int(offsetK),
                int(yLowBoundaryMode), int(yHighBoundaryMode),
                int(fullSizeY), int(fullSizeZ),
            )


def _apply_electric_ey_cpml_z_correction(
    *,
    Ey,
    Hx,
    EyCurl,
    PsiEyZ,
    InvKappaEyZ,
    BEyZ,
    CEyZ,
    invDz,
    offsetI,
    offsetJ,
    offsetK,
    xLowBoundaryMode,
    xHighBoundaryMode,
    fullSizeX,
    fullSizeZ,
):
    _require_cuda_tensors(Ey, Hx, EyCurl, PsiEyZ, InvKappaEyZ, BEyZ, CEyZ, invDz)
    get_compiled_extension().apply_electric_ey_cpml_z_correction(
                Ey, Hx, EyCurl, PsiEyZ, InvKappaEyZ, BEyZ, CEyZ, invDz,
                int(offsetI), int(offsetJ), int(offsetK),
                int(xLowBoundaryMode), int(xHighBoundaryMode),
                int(fullSizeX), int(fullSizeZ),
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
                invDy,
                invDz,
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
                invDx,
                invDz,
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
                invDx,
                invDy,
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
                invDy,
                invDz,
            )


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
                invDx,
                invDz,
            )


def _electric_ex_bloch_y_standard_z(
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
    invDy,
    invDz,
    zLowBoundaryMode,
    zHighBoundaryMode,
):
    get_compiled_extension().update_electric_ex_bloch_y_standard_z(
                ExReal, ExImag, HyReal, HyImag, HzReal, HzImag, ExDecay, ExCurl,
                float(phaseCosY), float(phaseSinY), invDy, invDz,
                int(zLowBoundaryMode), int(zHighBoundaryMode),
            )


def _electric_ey_bloch_x_standard_z(
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
    invDx,
    invDz,
    zLowBoundaryMode,
    zHighBoundaryMode,
):
    get_compiled_extension().update_electric_ey_bloch_x_standard_z(
                EyReal, EyImag, HxReal, HxImag, HzReal, HzImag, EyDecay, EyCurl,
                float(phaseCosX), float(phaseSinX), invDx, invDz,
                int(zLowBoundaryMode), int(zHighBoundaryMode),
            )


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
                invDx,
                invDy,
            )


def _add_source_patch(*, field, sourcePatch, offsetI, offsetJ, offsetK, signal):
    get_compiled_extension().add_source_patch(
                field,
                sourcePatch,
                int(offsetI),
                int(offsetJ),
                int(offsetK),
                float(signal),
            )


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
    wrapAxisA=1,
    wrapAxisB=1,
):
    get_compiled_extension().add_source_patch_bloch(
        ExReal, ExImag, EyReal, EyImag, EzReal, EzImag, sourcePatch,
        int(offsetI), int(offsetJ), int(offsetK),
        float(signalReal), float(signalImag), int(axisCode),
        float(phaseCosA), float(phaseSinA), float(phaseCosB), float(phaseSinB),
        int(wrapAxisA), int(wrapAxisB),
    )


def _add_cw_phased_source_patch_bloch(
    *,
    ExReal,
    ExImag,
    EyReal,
    EyImag,
    EzReal,
    EzImag,
    sourcePatchCos,
    sourcePatchSin,
    offsetI,
    offsetJ,
    offsetK,
    signalCos,
    signalSin,
    axisCode,
    phaseCosA,
    phaseSinA,
    phaseCosB,
    phaseSinB,
    wrapAxisA=1,
    wrapAxisB=1,
):
    real_values = sourcePatchCos * float(signalCos) + sourcePatchSin * float(signalSin)
    imag_values = sourcePatchCos * float(signalSin) - sourcePatchSin * float(signalCos)
    extension = get_compiled_extension()
    extension.add_source_patch_bloch(
        ExReal, ExImag, EyReal, EyImag, EzReal, EzImag, real_values,
        int(offsetI), int(offsetJ), int(offsetK), 1.0, 0.0, int(axisCode),
        float(phaseCosA), float(phaseSinA), float(phaseCosB), float(phaseSinB),
        int(wrapAxisA), int(wrapAxisB),
    )
    extension.add_source_patch_bloch(
        ExReal, ExImag, EyReal, EyImag, EzReal, EzImag, imag_values,
        int(offsetI), int(offsetJ), int(offsetK), 0.0, 1.0, int(axisCode),
        float(phaseCosA), float(phaseSinA), float(phaseCosB), float(phaseSinB),
        int(wrapAxisA), int(wrapAxisB),
    )


def _add_scaled_slice_source_patch(*, field, sourcePatch, incidentField, sampleIndex, offsetI, offsetJ, offsetK, scale):
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


def _update_auxiliary_magnetic(*, Magnetic, Electric, MagneticDecay, MagneticCurl):
    get_compiled_extension().update_auxiliary_magnetic(
                Magnetic,
                Electric,
                MagneticDecay,
                MagneticCurl,
            )


def _update_auxiliary_electric(*, Electric, Magnetic, ElectricDecay, ElectricCurl, sourceIndex, sourceValue):
    get_compiled_extension().update_auxiliary_electric(
                Electric,
                Magnetic,
                ElectricDecay,
                ElectricCurl,
                int(sourceIndex),
                float(sourceValue),
            )


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


def _accumulate_plane_observer(*, field, planeRealAccum, planeImagAccum, axisCode, planeIndex, weightedCos, weightedSin):
    get_compiled_extension().accumulate_plane_observer(
                field,
                planeRealAccum,
                planeImagAccum,
                int(axisCode),
                int(planeIndex),
                float(weightedCos),
                float(weightedSin),
            )


def _plane_flux_reduce(*, ea, eb, ha, hb, weights, out, outIndex, scale):
    _require_cuda_tensors(ea, eb, ha, hb, weights, out)
    get_compiled_extension().plane_flux_reduce(
        ea,
        eb,
        ha,
        hb,
        weights,
        out,
        int(outIndex),
        float(scale),
    )


def _update_debye_current(*, ElectricField, Polarization, PolarizationCurrent, DebyeDrive, decay, dt):
    get_compiled_extension().update_debye_current(
                ElectricField,
                Polarization,
                PolarizationCurrent,
                DebyeDrive,
                float(decay),
                float(dt),
            )


def _update_drude_current(*, ElectricField, PolarizationCurrent, DrudeDrive, decay):
    get_compiled_extension().update_drude_current(
                ElectricField,
                PolarizationCurrent,
                DrudeDrive,
                float(decay),
            )


def _update_lorentz_current(*, ElectricField, Polarization, PolarizationCurrent, LorentzDrive, decay, restoring, dt):
    get_compiled_extension().update_lorentz_current(
                ElectricField,
                Polarization,
                PolarizationCurrent,
                LorentzDrive,
                float(decay),
                float(restoring),
                float(dt),
            )


def _apply_polarization_current(*, ElectricField, PolarizationCurrent, InvPermittivity, dt):
    get_compiled_extension().apply_polarization_current(
                ElectricField,
                PolarizationCurrent,
                InvPermittivity,
                float(dt),
            )


def _update_kerr_ex(*, DynamicCurl, Ex, Ey, Ez, LinearPermittivity, ExDecay, KerrChi3, dt, eps0):
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


def _update_kerr_ey(*, DynamicCurl, Ex, Ey, Ez, LinearPermittivity, EyDecay, KerrChi3, dt, eps0):
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


def _update_kerr_ez(*, DynamicCurl, Ex, Ey, Ez, LinearPermittivity, EzDecay, KerrChi3, dt, eps0):
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


_NONLINEAR_COMPONENT_CODES = {"Ex": 0, "Ey": 1, "Ez": 2}


def _update_nonlinear_coefficients(
    *,
    DynamicDecay,
    DynamicCurl,
    Ex,
    Ey,
    Ez,
    LinearPermittivity,
    ExternalDecay,
    SigmaStatic,
    Chi2,
    Chi3,
    TpaSigma,
    component,
    dt,
    eps0,
):
    _invalidate_uniform_scalar(DynamicDecay)
    _invalidate_uniform_scalar(DynamicCurl)
    get_compiled_extension().update_nonlinear_coefficients(
                DynamicDecay,
                DynamicCurl,
                Ex,
                Ey,
                Ez,
                LinearPermittivity,
                ExternalDecay,
                SigmaStatic,
                Chi2,
                Chi3,
                TpaSigma,
                _NONLINEAR_COMPONENT_CODES[component],
                float(dt),
                float(eps0),
            )


def _electric_ex_full_aniso(*, Ex, Hx, Hy, Hz, CoeffY, CoeffZ, invDx, invDy, invDz, periodicX, periodicY, periodicZ):
    _require_cuda_tensors(Ex, Hx, Hy, Hz, CoeffY, CoeffZ, invDx, invDy, invDz)
    get_compiled_extension().update_electric_ex_full_aniso(
                Ex, Hx, Hy, Hz, CoeffY, CoeffZ, invDx, invDy, invDz,
                int(periodicX), int(periodicY), int(periodicZ),
            )


def _electric_ey_full_aniso(*, Ey, Hx, Hy, Hz, CoeffX, CoeffZ, invDx, invDy, invDz, periodicX, periodicY, periodicZ):
    _require_cuda_tensors(Ey, Hx, Hy, Hz, CoeffX, CoeffZ, invDx, invDy, invDz)
    get_compiled_extension().update_electric_ey_full_aniso(
                Ey, Hx, Hy, Hz, CoeffX, CoeffZ, invDx, invDy, invDz,
                int(periodicX), int(periodicY), int(periodicZ),
            )


def _electric_ez_full_aniso(*, Ez, Hx, Hy, Hz, CoeffX, CoeffY, invDx, invDy, invDz, periodicX, periodicY, periodicZ):
    _require_cuda_tensors(Ez, Hx, Hy, Hz, CoeffX, CoeffY, invDx, invDy, invDz)
    get_compiled_extension().update_electric_ez_full_aniso(
                Ez, Hx, Hy, Hz, CoeffX, CoeffY, invDx, invDy, invDz,
                int(periodicX), int(periodicY), int(periodicZ),
            )


def _reverse_electric_hx_standard(*, AdjHxMid, AdjHxPost, AdjEyPost, AdjEzPost, EyCurl, EzCurl, invDy, invDz):
    get_compiled_extension().reverse_electric_adjoint_to_hx_standard(
                AdjHxMid, AdjHxPost, AdjEyPost, AdjEzPost, EyCurl, EzCurl, invDy, invDz
            )


def _reverse_electric_hy_standard(*, AdjHyMid, AdjHyPost, AdjExPost, AdjEzPost, ExCurl, EzCurl, invDx, invDz):
    get_compiled_extension().reverse_electric_adjoint_to_hy_standard(
                AdjHyMid, AdjHyPost, AdjExPost, AdjEzPost, ExCurl, EzCurl, invDx, invDz
            )


def _reverse_electric_hz_standard(*, AdjHzMid, AdjHzPost, AdjExPost, AdjEyPost, ExCurl, EyCurl, invDx, invDy):
    get_compiled_extension().reverse_electric_adjoint_to_hz_standard(
                AdjHzMid, AdjHzPost, AdjExPost, AdjEyPost, ExCurl, EyCurl, invDx, invDy
            )


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
    get_compiled_extension().reverse_electric_adjoint_to_hx_bloch(
                AdjHxMidReal, AdjHxMidImag, AdjHxPostReal, AdjHxPostImag,
                AdjEyPostReal, AdjEyPostImag, AdjEzPostReal, AdjEzPostImag,
                EyCurl, EzCurl, float(phaseCosY), float(phaseSinY), float(phaseCosZ), float(phaseSinZ),
                invDy, invDz,
            )


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
    get_compiled_extension().reverse_electric_adjoint_to_hy_bloch(
                AdjHyMidReal, AdjHyMidImag, AdjHyPostReal, AdjHyPostImag,
                AdjExPostReal, AdjExPostImag, AdjEzPostReal, AdjEzPostImag,
                ExCurl, EzCurl, float(phaseCosX), float(phaseSinX), float(phaseCosZ), float(phaseSinZ),
                invDx, invDz,
            )


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
    get_compiled_extension().reverse_electric_adjoint_to_hz_bloch(
                AdjHzMidReal, AdjHzMidImag, AdjHzPostReal, AdjHzPostImag,
                AdjExPostReal, AdjExPostImag, AdjEyPostReal, AdjEyPostImag,
                ExCurl, EyCurl, float(phaseCosX), float(phaseSinX), float(phaseCosY), float(phaseSinY),
                invDx, invDy,
            )


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
    invDyE,
    invDzE,
    invDyH,
    invDzH,
    yLowBoundaryMode,
    yHighBoundaryMode,
    zLowBoundaryMode,
    zHighBoundaryMode,
):
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
                invDyE,
                invDzE,
                invDyH,
                invDzH,
                int(yLowBoundaryMode),
                int(yHighBoundaryMode),
                int(zLowBoundaryMode),
                int(zHighBoundaryMode),
            )


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
    invDxE,
    invDzE,
    invDxH,
    invDzH,
    xLowBoundaryMode,
    xHighBoundaryMode,
    zLowBoundaryMode,
    zHighBoundaryMode,
):
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
                invDxE,
                invDzE,
                invDxH,
                invDzH,
                int(xLowBoundaryMode),
                int(xHighBoundaryMode),
                int(zLowBoundaryMode),
                int(zHighBoundaryMode),
            )


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
    invDxE,
    invDyE,
    invDxH,
    invDyH,
    xLowBoundaryMode,
    xHighBoundaryMode,
    yLowBoundaryMode,
    yHighBoundaryMode,
):
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
                invDxE,
                invDyE,
                invDxH,
                invDyH,
                int(xLowBoundaryMode),
                int(xHighBoundaryMode),
                int(yLowBoundaryMode),
                int(yHighBoundaryMode),
            )


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
    invDyE,
    invDzE,
    invDyH,
    invDzH,
):
    get_compiled_extension().reverse_magnetic_adjoint_to_ex_bloch(
                AdjExPrevReal, AdjExPrevImag, GradEpsEx, AdjExPostReal, AdjExPostImag,
                AdjHyMidReal, AdjHyMidImag, AdjHzMidReal, AdjHzMidImag,
                ExDecay, ExCurl, EpsEx, HyMidReal, HyMidImag, HzMidReal, HzMidImag,
                HyCurl, HzCurl, float(phaseCosY), float(phaseSinY), float(phaseCosZ), float(phaseSinZ),
                invDyE, invDzE, invDyH, invDzH,
            )


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
    invDxE,
    invDzE,
    invDxH,
    invDzH,
):
    get_compiled_extension().reverse_magnetic_adjoint_to_ey_bloch(
                AdjEyPrevReal, AdjEyPrevImag, GradEpsEy, AdjEyPostReal, AdjEyPostImag,
                AdjHxMidReal, AdjHxMidImag, AdjHzMidReal, AdjHzMidImag,
                EyDecay, EyCurl, EpsEy, HxMidReal, HxMidImag, HzMidReal, HzMidImag,
                HxCurl, HzCurl, float(phaseCosX), float(phaseSinX), float(phaseCosZ), float(phaseSinZ),
                invDxE, invDzE, invDxH, invDzH,
            )


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
    invDxE,
    invDyE,
    invDxH,
    invDyH,
):
    get_compiled_extension().reverse_magnetic_adjoint_to_ez_bloch(
                AdjEzPrevReal, AdjEzPrevImag, GradEpsEz, AdjEzPostReal, AdjEzPostImag,
                AdjHxMidReal, AdjHxMidImag, AdjHyMidReal, AdjHyMidImag,
                EzDecay, EzCurl, EpsEz, HxMidReal, HxMidImag, HyMidReal, HyMidImag,
                HxCurl, HyCurl, float(phaseCosX), float(phaseSinX), float(phaseCosY), float(phaseSinY),
                invDxE, invDyE, invDxH, invDyH,
            )


def _accumulate_diff_adjoint(field_grad, diff_grad, axis, inv_delta, *, forward):
    method = (
                get_compiled_extension().accumulate_forward_diff_adjoint
                if forward
                else get_compiled_extension().accumulate_backward_diff_adjoint
            )
    method(field_grad, diff_grad, int(axis), inv_delta)


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
    get_compiled_extension().reverse_electric_component_ex_cpml(
                AdjExPrev, GradEpsEx, AdjPsiPosPrev, AdjPsiNegPrev, AdjDPos, AdjDNeg,
                AdjExPost, AdjPsiPosPost, AdjPsiNegPost, ExDecay, ExCurl, EpsEx,
                PsiPos, PsiNeg, BPos, CPos, InvKappaPos, BNeg, CNeg, InvKappaNeg,
                HyMid, HzMid, invDy, invDz, int(yLowBoundaryMode), int(yHighBoundaryMode), int(zLowBoundaryMode), int(zHighBoundaryMode),
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
    get_compiled_extension().reverse_electric_component_ey_cpml(
                AdjEyPrev, GradEpsEy, AdjPsiPosPrev, AdjPsiNegPrev, AdjDPos, AdjDNeg,
                AdjEyPost, AdjPsiPosPost, AdjPsiNegPost, EyDecay, EyCurl, EpsEy,
                PsiPos, PsiNeg, BPos, CPos, InvKappaPos, BNeg, CNeg, InvKappaNeg,
                HxMid, HzMid, invDx, invDz, int(xLowBoundaryMode), int(xHighBoundaryMode), int(zLowBoundaryMode), int(zHighBoundaryMode),
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
    get_compiled_extension().reverse_electric_component_ez_cpml(
                AdjEzPrev, GradEpsEz, AdjPsiPosPrev, AdjPsiNegPrev, AdjDPos, AdjDNeg,
                AdjEzPost, AdjPsiPosPost, AdjPsiNegPost, EzDecay, EzCurl, EpsEz,
                PsiPos, PsiNeg, BPos, CPos, InvKappaPos, BNeg, CNeg, InvKappaNeg,
                HxMid, HyMid, invDx, invDy, int(xLowBoundaryMode), int(xHighBoundaryMode), int(yLowBoundaryMode), int(yHighBoundaryMode),
            )


def _reverse_magnetic_cpml_hx(**kwargs):
    get_compiled_extension().reverse_magnetic_component_hx_cpml(
                kwargs["AdjHxPrev"], kwargs["AdjPsiPosPrev"], kwargs["AdjPsiNegPrev"], kwargs["AdjDPos"], kwargs["AdjDNeg"],
                kwargs["AdjHxPost"], kwargs["AdjPsiPosPost"], kwargs["AdjPsiNegPost"], kwargs["HxDecay"], kwargs["HxCurl"],
                kwargs["BPos"], kwargs["CPos"], kwargs["InvKappaPos"], kwargs["BNeg"], kwargs["CNeg"], kwargs["InvKappaNeg"],
            )


def _reverse_magnetic_cpml_hy(**kwargs):
    get_compiled_extension().reverse_magnetic_component_hy_cpml(
                kwargs["AdjHyPrev"], kwargs["AdjPsiPosPrev"], kwargs["AdjPsiNegPrev"], kwargs["AdjDPos"], kwargs["AdjDNeg"],
                kwargs["AdjHyPost"], kwargs["AdjPsiPosPost"], kwargs["AdjPsiNegPost"], kwargs["HyDecay"], kwargs["HyCurl"],
                kwargs["BPos"], kwargs["CPos"], kwargs["InvKappaPos"], kwargs["BNeg"], kwargs["CNeg"], kwargs["InvKappaNeg"],
            )


def _reverse_magnetic_cpml_hz(**kwargs):
    get_compiled_extension().reverse_magnetic_component_hz_cpml(
                kwargs["AdjHzPrev"], kwargs["AdjPsiPosPrev"], kwargs["AdjPsiNegPrev"], kwargs["AdjDPos"], kwargs["AdjDNeg"],
                kwargs["AdjHzPost"], kwargs["AdjPsiPosPost"], kwargs["AdjPsiNegPost"], kwargs["HzDecay"], kwargs["HzCurl"],
                kwargs["BPos"], kwargs["CPos"], kwargs["InvKappaPos"], kwargs["BNeg"], kwargs["CNeg"], kwargs["InvKappaNeg"],
            )


def _reverse_magnetic_hx_decay(*, AdjHxPrev, AdjHxMid, HxDecay):
    get_compiled_extension().reverse_magnetic_adjoint_decay(AdjHxPrev, AdjHxMid, HxDecay)


def _reverse_magnetic_hy_decay(*, AdjHyPrev, AdjHyMid, HyDecay):
    get_compiled_extension().reverse_magnetic_adjoint_decay(AdjHyPrev, AdjHyMid, HyDecay)


def _reverse_magnetic_hz_decay(*, AdjHzPrev, AdjHzMid, HzDecay):
    get_compiled_extension().reverse_magnetic_adjoint_decay(AdjHzPrev, AdjHzMid, HzDecay)


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
    get_compiled_extension().reverse_debye_current(
                AdjElectricPrev,
                AdjPolarizationPrev,
                AdjPolarizationPost,
                AdjCurrentPost,
                DebyeDrive,
                float(decay),
                float(dt),
            )


def _reverse_drude_current(*, AdjElectricPrev, AdjCurrentPrev, AdjCurrentPost, DrudeDrive, decay):
    get_compiled_extension().reverse_drude_current(
                AdjElectricPrev,
                AdjCurrentPrev,
                AdjCurrentPost,
                DrudeDrive,
                float(decay),
            )


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


def _accumulate_tfsf_scalar_sample_adjoint(*, AdjAuxField, AdjFieldPatch, CoeffPatch, sampleIndex, componentScale):
    get_compiled_extension().accumulate_tfsf_scalar_sample_adjoint(
                AdjAuxField,
                AdjFieldPatch,
                CoeffPatch,
                int(sampleIndex),
                float(componentScale),
            )


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
    get_compiled_extension().accumulate_tfsf_line_sample_adjoint(
                AdjAuxField,
                AdjFieldPatch,
                CoeffPatch,
                sample_indices,
                int(sampleAxisCode),
                float(componentScale),
            )


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
    get_compiled_extension().accumulate_tfsf_interpolated_sample_adjoint(
                AdjAuxField,
                AdjFieldPatch,
                CoeffPatch,
                SamplePositions,
                float(origin),
                float(ds),
                float(componentScale),
            )


def _reverse_tfsf_auxiliary_electric(
    *,
    AdjElectricPrev,
    AdjMagneticAfter,
    AdjElectricPost,
    ElectricDecay,
    ElectricCurl,
    sourceIndex,
):
    get_compiled_extension().reverse_tfsf_auxiliary_electric(
                AdjElectricPrev,
                AdjMagneticAfter,
                AdjElectricPost,
                ElectricDecay,
                ElectricCurl,
                int(sourceIndex),
            )


def _reverse_tfsf_auxiliary_magnetic(
    *,
    AdjElectricPrev,
    AdjMagneticPrev,
    AdjMagneticAfter,
    MagneticDecay,
    MagneticCurl,
):
    get_compiled_extension().reverse_tfsf_auxiliary_magnetic(
                AdjElectricPrev,
                AdjMagneticPrev,
                AdjMagneticAfter,
                MagneticDecay,
                MagneticCurl,
            )


def _clamp_field_face(*, field, axis, side):
    get_compiled_extension().clamp_field_face(field, int(axis), int(side))


def _clamp_pec_boundary(*, field, axisA, axisB):
    get_compiled_extension().clamp_pec_boundary(field, int(axisA), int(axisB))


def _mur_abc_face(*, field, axis, boundaryIndex, adjacentIndex, coef, prevBoundary, prevAdjacent):
    _require_cuda_tensors(field, prevBoundary, prevAdjacent)
    get_compiled_extension().mur_abc_face(
        field,
        int(axis),
        int(boundaryIndex),
        int(adjacentIndex),
        float(coef),
        prevBoundary,
        prevAdjacent,
    )


def _project_periodic_boundary(*, field, axis):
    get_compiled_extension().project_periodic_boundary(field, int(axis))


def _project_bloch_boundary(*, fieldReal, fieldImag, axis, phaseCos, phaseSin):
    get_compiled_extension().project_bloch_boundary(
                fieldReal,
                fieldImag,
                int(axis),
                float(phaseCos),
                float(phaseSin),
            )


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
    "applyElectricFieldExCpmlZCorrection3D": _apply_electric_ex_cpml_z_correction,
    "applyElectricFieldEyCpmlZCorrection3D": _apply_electric_ey_cpml_z_correction,
    "updateElectricFieldExBloch3D": _electric_ex_bloch,
    "updateElectricFieldEyBloch3D": _electric_ey_bloch,
    "updateElectricFieldExBlochYStandardZ3D": _electric_ex_bloch_y_standard_z,
    "updateElectricFieldEyBlochXStandardZ3D": _electric_ey_bloch_x_standard_z,
    "updateElectricFieldEzBloch3D": _electric_ez_bloch,
    "addSourcePatch3D": _add_source_patch,
    "addCwPhasedSourcePatch3D": _add_cw_phased_source_patch,
    "addTimeShiftedSourcePatch3D": _add_time_shifted_source_patch,
    "addSourcePatchExPeriodic3D": _add_source_patch_periodic,
    "addSourcePatchEyPeriodic3D": _add_source_patch_periodic,
    "addSourcePatchEzPeriodic3D": _add_source_patch_periodic,
    "addSourcePatchBloch3D": _add_source_patch_bloch,
    "addCwPhasedSourcePatchBloch3D": _add_cw_phased_source_patch_bloch,
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
    "planeFluxReduce": _plane_flux_reduce,
    "updateDebyeCurrent3D": _update_debye_current,
    "updateDrudeCurrent3D": _update_drude_current,
    "updateLorentzCurrent3D": _update_lorentz_current,
    "applyPolarizationCurrent3D": _apply_polarization_current,
    "updateKerrElectricFieldExCurl3D": _update_kerr_ex,
    "updateKerrElectricFieldEyCurl3D": _update_kerr_ey,
    "updateKerrElectricFieldEzCurl3D": _update_kerr_ez,
    "updateNonlinearElectricCoefficients3D": _update_nonlinear_coefficients,
    "updateElectricFieldExFullAniso3D": _electric_ex_full_aniso,
    "updateElectricFieldEyFullAniso3D": _electric_ey_full_aniso,
    "updateElectricFieldEzFullAniso3D": _electric_ez_full_aniso,
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
    "applyMurBoundary3D": _mur_abc_face,
    "projectPeriodicBoundary3D": _project_periodic_boundary,
    "projectBlochBoundary3D": _project_bloch_boundary,
}
