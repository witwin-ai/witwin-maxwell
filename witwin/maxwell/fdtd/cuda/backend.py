from __future__ import annotations

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


def build_info() -> dict[str, Any]:
    return {
        "backend": "torch-cuda",
        "compiled_extension_loaded": _COMPILED_EXTENSION is not None,
        "cuda_available": is_available(),
        "torch_version": torch.__version__,
    }


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
    __slots__ = ("kernel", "kwargs", "may_receive_strided")

    def __init__(self, kernel: Callable[..., None], kwargs: dict, *, may_receive_strided: bool):
        self.kernel = kernel
        self.kwargs = kwargs
        self.may_receive_strided = may_receive_strided

    def launchRaw(self):  # noqa: N802 - kernel launch entry point
        if self.may_receive_strided:
            _call_with_contiguous(self.kernel, self.kwargs)
        else:
            self.kernel(**self.kwargs)
        return None


class NativeFDTDModule:
    def __getattr__(self, name: str):
        kernel = _KERNELS.get(name)
        if kernel is None:
            raise AttributeError(f"Native CUDA FDTD backend does not implement kernel {name!r}.")

        may_receive_strided = name.startswith(
            ("reverse", "seedInject", "accumulate", "applyElectricField")
        )

        def bind(**kwargs):
            return _Launch(kernel, kwargs, may_receive_strided=may_receive_strided)

        bind.__name__ = name
        # Cache on the instance so later lookups bypass __getattr__ entirely.
        object.__setattr__(self, name, bind)
        return bind


_NATIVE_MODULE = NativeFDTDModule()


def get_native_fdtd_module() -> NativeFDTDModule:
    if not is_available():
        raise RuntimeError("Native CUDA FDTD backend requires torch.cuda.is_available() to be True.")
    get_compiled_extension()
    return _NATIVE_MODULE


def _magnetic_hx_standard(*, Hx, Ey, Ez, HxDecay, HxCurl, invDy, invDz):
    _COMPILED_EXTENSION.update_magnetic_hx_standard(Hx, Ey, Ez, HxDecay, HxCurl, invDy, invDz)


def _magnetic_hy_standard(*, Hy, Ex, Ez, HyDecay, HyCurl, invDx, invDz):
    _COMPILED_EXTENSION.update_magnetic_hy_standard(Hy, Ex, Ez, HyDecay, HyCurl, invDx, invDz)


def _magnetic_hz_standard(*, Hz, Ex, Ey, HzDecay, HzCurl, invDx, invDy):
    _COMPILED_EXTENSION.update_magnetic_hz_standard(Hz, Ex, Ey, HzDecay, HzCurl, invDx, invDy)


def _magnetic_hx_standard_bounded(
    *, Hx, Ey, Ez, HxDecay, HxCurl, invDy, invDz,
    localXBegin, localXEnd, globalXOffset, globalXExtent,
):
    _COMPILED_EXTENSION.update_magnetic_hx_standard_bounded(
        Hx, Ey, Ez, HxDecay, HxCurl, invDy, invDz,
        int(localXBegin), int(localXEnd), int(globalXOffset), int(globalXExtent),
    )


def _magnetic_hy_standard_bounded(
    *, Hy, Ex, Ez, HyDecay, HyCurl, invDx, invDz,
    localXBegin, localXEnd, globalXOffset, globalXExtent,
):
    _COMPILED_EXTENSION.update_magnetic_hy_standard_bounded(
        Hy, Ex, Ez, HyDecay, HyCurl, invDx, invDz,
        int(localXBegin), int(localXEnd), int(globalXOffset), int(globalXExtent),
    )


def _magnetic_hz_standard_bounded(
    *, Hz, Ex, Ey, HzDecay, HzCurl, invDx, invDy,
    localXBegin, localXEnd, globalXOffset, globalXExtent,
):
    _COMPILED_EXTENSION.update_magnetic_hz_standard_bounded(
        Hz, Ex, Ey, HzDecay, HzCurl, invDx, invDy,
        int(localXBegin), int(localXEnd), int(globalXOffset), int(globalXExtent),
    )


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
    _COMPILED_EXTENSION.update_magnetic_hx_cpml(
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
    _COMPILED_EXTENSION.update_magnetic_hy_cpml(
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
    _COMPILED_EXTENSION.update_magnetic_hz_cpml(
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
    uniformDecay,
    uniformCurl,
):
    _COMPILED_EXTENSION.update_magnetic_hx_cpml_compressed(
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
                uniformDecay,
                uniformCurl,
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
    uniformDecay,
    uniformCurl,
):
    _COMPILED_EXTENSION.update_magnetic_hy_cpml_compressed(
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
                uniformDecay,
                uniformCurl,
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
    uniformDecay,
    uniformCurl,
):
    _COMPILED_EXTENSION.update_magnetic_hz_cpml_compressed(
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
                uniformDecay,
                uniformCurl,
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
    _COMPILED_EXTENSION.update_electric_ex_standard(
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
    _COMPILED_EXTENSION.update_electric_ey_standard(
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
    _COMPILED_EXTENSION.update_electric_ez_standard(
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




def _electric_ex_standard_bounded(
    *, Ex, Hy, Hz, ExDecay, ExCurl, invDy, invDz,
    yLowBoundaryMode, yHighBoundaryMode, zLowBoundaryMode, zHighBoundaryMode,
    localXBegin, localXEnd, globalXOffset, globalXExtent,
):
    _COMPILED_EXTENSION.update_electric_ex_standard_bounded(
        Ex, Hy, Hz, ExDecay, ExCurl, invDy, invDz,
        int(yLowBoundaryMode), int(yHighBoundaryMode), int(zLowBoundaryMode), int(zHighBoundaryMode),
        int(localXBegin), int(localXEnd), int(globalXOffset), int(globalXExtent),
    )


def _electric_ey_standard_bounded(
    *, Ey, Hx, Hz, EyDecay, EyCurl, invDx, invDz,
    xLowBoundaryMode, xHighBoundaryMode, zLowBoundaryMode, zHighBoundaryMode,
    localXBegin, localXEnd, globalXOffset, globalXExtent,
):
    _COMPILED_EXTENSION.update_electric_ey_standard_bounded(
        Ey, Hx, Hz, EyDecay, EyCurl, invDx, invDz,
        int(xLowBoundaryMode), int(xHighBoundaryMode), int(zLowBoundaryMode), int(zHighBoundaryMode),
        int(localXBegin), int(localXEnd), int(globalXOffset), int(globalXExtent),
    )


def _electric_ez_standard_bounded(
    *, Ez, Hx, Hy, EzDecay, EzCurl, invDx, invDy,
    xLowBoundaryMode, xHighBoundaryMode, yLowBoundaryMode, yHighBoundaryMode,
    localXBegin, localXEnd, globalXOffset, globalXExtent,
):
    _COMPILED_EXTENSION.update_electric_ez_standard_bounded(
        Ez, Hx, Hy, EzDecay, EzCurl, invDx, invDy,
        int(xLowBoundaryMode), int(xHighBoundaryMode), int(yLowBoundaryMode), int(yHighBoundaryMode),
        int(localXBegin), int(localXEnd), int(globalXOffset), int(globalXExtent),
    )
def _electric_ex_modulated(
    *,
    Ex,
    Hy,
    Hz,
    ExDecay,
    ExCurl,
    ModCos,
    ModSin,
    ModOmega,
    ModulationTime,
    invDy,
    invDz,
    yLowBoundaryMode,
    yHighBoundaryMode,
    zLowBoundaryMode,
    zHighBoundaryMode,
):
    _COMPILED_EXTENSION.update_electric_ex_modulated(
                Ex,
                Hy,
                Hz,
                ExDecay,
                ExCurl,
                ModCos,
                ModSin,
                ModOmega,
                ModulationTime,
                invDy,
                invDz,
                int(yLowBoundaryMode),
                int(yHighBoundaryMode),
                int(zLowBoundaryMode),
                int(zHighBoundaryMode),
            )


def _electric_ey_modulated(
    *,
    Ey,
    Hx,
    Hz,
    EyDecay,
    EyCurl,
    ModCos,
    ModSin,
    ModOmega,
    ModulationTime,
    invDx,
    invDz,
    xLowBoundaryMode,
    xHighBoundaryMode,
    zLowBoundaryMode,
    zHighBoundaryMode,
):
    _COMPILED_EXTENSION.update_electric_ey_modulated(
                Ey,
                Hx,
                Hz,
                EyDecay,
                EyCurl,
                ModCos,
                ModSin,
                ModOmega,
                ModulationTime,
                invDx,
                invDz,
                int(xLowBoundaryMode),
                int(xHighBoundaryMode),
                int(zLowBoundaryMode),
                int(zHighBoundaryMode),
            )


def _electric_ez_modulated(
    *,
    Ez,
    Hx,
    Hy,
    EzDecay,
    EzCurl,
    ModCos,
    ModSin,
    ModOmega,
    ModulationTime,
    invDx,
    invDy,
    xLowBoundaryMode,
    xHighBoundaryMode,
    yLowBoundaryMode,
    yHighBoundaryMode,
):
    _COMPILED_EXTENSION.update_electric_ez_modulated(
                Ez,
                Hx,
                Hy,
                EzDecay,
                EzCurl,
                ModCos,
                ModSin,
                ModOmega,
                ModulationTime,
                invDx,
                invDy,
                int(xLowBoundaryMode),
                int(xHighBoundaryMode),
                int(yLowBoundaryMode),
                int(yHighBoundaryMode),
            )


def _electric_ex_cpml_modulated(
    *,
    Ex,
    Hy,
    Hz,
    ExDecay,
    ExCurl,
    ModCos,
    ModSin,
    ModOmega,
    ModulationTime,
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
    _COMPILED_EXTENSION.update_electric_ex_cpml_modulated(
                Ex,
                Hy,
                Hz,
                ExDecay,
                ExCurl,
                ModCos,
                ModSin,
                ModOmega,
                ModulationTime,
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


def _electric_ey_cpml_modulated(
    *,
    Ey,
    Hx,
    Hz,
    EyDecay,
    EyCurl,
    ModCos,
    ModSin,
    ModOmega,
    ModulationTime,
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
    _COMPILED_EXTENSION.update_electric_ey_cpml_modulated(
                Ey,
                Hx,
                Hz,
                EyDecay,
                EyCurl,
                ModCos,
                ModSin,
                ModOmega,
                ModulationTime,
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


def _electric_ez_cpml_modulated(
    *,
    Ez,
    Hx,
    Hy,
    EzDecay,
    EzCurl,
    ModCos,
    ModSin,
    ModOmega,
    ModulationTime,
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
    _COMPILED_EXTENSION.update_electric_ez_cpml_modulated(
                Ez,
                Hx,
                Hy,
                EzDecay,
                EzCurl,
                ModCos,
                ModSin,
                ModOmega,
                ModulationTime,
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
    _COMPILED_EXTENSION.update_electric_ex_cpml(
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
    _COMPILED_EXTENSION.update_electric_ey_cpml(
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
    _COMPILED_EXTENSION.update_electric_ez_cpml(
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
    _COMPILED_EXTENSION.apply_electric_ex_cpml_z_correction(
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
    _COMPILED_EXTENSION.apply_electric_ey_cpml_z_correction(
                Ey, Hx, EyCurl, PsiEyZ, InvKappaEyZ, BEyZ, CEyZ, invDz,
                int(offsetI), int(offsetJ), int(offsetK),
                int(xLowBoundaryMode), int(xHighBoundaryMode),
                int(fullSizeX), int(fullSizeZ),
            )


def _apply_electric_ex_cpml_y_correction(
    *,
    Ex,
    Hz,
    ExCurl,
    PsiExY,
    InvKappaExY,
    BExY,
    CExY,
    invDy,
    offsetI,
    offsetJ,
    offsetK,
    zLowBoundaryMode,
    zHighBoundaryMode,
    fullSizeZ,
    fullSizeY,
):
    _COMPILED_EXTENSION.apply_electric_ex_cpml_y_correction(
                Ex, Hz, ExCurl, PsiExY, InvKappaExY, BExY, CExY, invDy,
                int(offsetI), int(offsetJ), int(offsetK),
                int(zLowBoundaryMode), int(zHighBoundaryMode),
                int(fullSizeZ), int(fullSizeY),
            )


def _apply_electric_ez_cpml_y_correction(
    *,
    Ez,
    Hx,
    EzCurl,
    PsiEzY,
    InvKappaEzY,
    BEzY,
    CEzY,
    invDy,
    offsetI,
    offsetJ,
    offsetK,
    xLowBoundaryMode,
    xHighBoundaryMode,
    fullSizeX,
    fullSizeY,
):
    _COMPILED_EXTENSION.apply_electric_ez_cpml_y_correction(
                Ez, Hx, EzCurl, PsiEzY, InvKappaEzY, BEzY, CEzY, invDy,
                int(offsetI), int(offsetJ), int(offsetK),
                int(xLowBoundaryMode), int(xHighBoundaryMode),
                int(fullSizeX), int(fullSizeY),
            )


def _apply_electric_ey_cpml_x_correction(
    *,
    Ey,
    Hz,
    EyCurl,
    PsiEyX,
    InvKappaEyX,
    BEyX,
    CEyX,
    invDx,
    offsetI,
    offsetJ,
    offsetK,
    zLowBoundaryMode,
    zHighBoundaryMode,
    fullSizeZ,
    fullSizeX,
):
    _COMPILED_EXTENSION.apply_electric_ey_cpml_x_correction(
                Ey, Hz, EyCurl, PsiEyX, InvKappaEyX, BEyX, CEyX, invDx,
                int(offsetI), int(offsetJ), int(offsetK),
                int(zLowBoundaryMode), int(zHighBoundaryMode),
                int(fullSizeZ), int(fullSizeX),
            )


def _apply_electric_ez_cpml_x_correction(
    *,
    Ez,
    Hy,
    EzCurl,
    PsiEzX,
    InvKappaEzX,
    BEzX,
    CEzX,
    invDx,
    offsetI,
    offsetJ,
    offsetK,
    yLowBoundaryMode,
    yHighBoundaryMode,
    fullSizeY,
    fullSizeX,
):
    _COMPILED_EXTENSION.apply_electric_ez_cpml_x_correction(
                Ez, Hy, EzCurl, PsiEzX, InvKappaEzX, BEzX, CEzX, invDx,
                int(offsetI), int(offsetJ), int(offsetK),
                int(yLowBoundaryMode), int(yHighBoundaryMode),
                int(fullSizeY), int(fullSizeX),
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
    uniformDecay,
    uniformCurl,
):
    _COMPILED_EXTENSION.update_electric_ex_cpml_compressed(
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
                uniformDecay,
                uniformCurl,
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
    uniformDecay,
    uniformCurl,
):
    _COMPILED_EXTENSION.update_electric_ey_cpml_compressed(
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
                uniformDecay,
                uniformCurl,
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
    uniformDecay,
    uniformCurl,
):
    _COMPILED_EXTENSION.update_electric_ez_cpml_compressed(
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
                uniformDecay,
                uniformCurl,
            )


def _electric_ex_cpml_modulated_compressed(
    *,
    Ex,
    Hy,
    Hz,
    ExDecay,
    ExCurl,
    ModCos,
    ModSin,
    ModOmega,
    ModulationTime,
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
    _COMPILED_EXTENSION.update_electric_ex_cpml_modulated_compressed(
                Ex,
                Hy,
                Hz,
                ExDecay,
                ExCurl,
                ModCos,
                ModSin,
                ModOmega,
                ModulationTime,
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
            )


def _electric_ey_cpml_modulated_compressed(
    *,
    Ey,
    Hx,
    Hz,
    EyDecay,
    EyCurl,
    ModCos,
    ModSin,
    ModOmega,
    ModulationTime,
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
    _COMPILED_EXTENSION.update_electric_ey_cpml_modulated_compressed(
                Ey,
                Hx,
                Hz,
                EyDecay,
                EyCurl,
                ModCos,
                ModSin,
                ModOmega,
                ModulationTime,
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
            )


def _electric_ez_cpml_modulated_compressed(
    *,
    Ez,
    Hx,
    Hy,
    EzDecay,
    EzCurl,
    ModCos,
    ModSin,
    ModOmega,
    ModulationTime,
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
    _COMPILED_EXTENSION.update_electric_ez_cpml_modulated_compressed(
                Ez,
                Hx,
                Hy,
                EzDecay,
                EzCurl,
                ModCos,
                ModSin,
                ModOmega,
                ModulationTime,
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
    _COMPILED_EXTENSION.update_electric_ex_bloch(
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
    _COMPILED_EXTENSION.update_electric_ey_bloch(
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
    _COMPILED_EXTENSION.update_electric_ex_bloch_y_standard_z(
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
    _COMPILED_EXTENSION.update_electric_ey_bloch_x_standard_z(
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
    _COMPILED_EXTENSION.update_electric_ez_bloch(
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
    _COMPILED_EXTENSION.add_source_patch(
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
    _COMPILED_EXTENSION.add_cw_phased_source_patch(
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
    _COMPILED_EXTENSION.add_time_shifted_source_patch(
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
    getattr(_COMPILED_EXTENSION, extension_method)(
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
    _COMPILED_EXTENSION.add_source_patch_bloch(
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
    extension = _COMPILED_EXTENSION
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
    _COMPILED_EXTENSION.add_scaled_slice_source_patch(
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
    _COMPILED_EXTENSION.add_scaled_line_source_patch(
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
    _COMPILED_EXTENSION.add_interpolated_source_patch(
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
    _COMPILED_EXTENSION.add_batched_reference_source_patches(
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
    _COMPILED_EXTENSION.add_batched_interpolated_source_patches(
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
    _COMPILED_EXTENSION.update_auxiliary_magnetic(
                Magnetic,
                Electric,
                MagneticDecay,
                MagneticCurl,
            )


def _update_auxiliary_electric(*, Electric, Magnetic, ElectricDecay, ElectricCurl, sourceIndex, sourceValue):
    _COMPILED_EXTENSION.update_auxiliary_electric(
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
    _COMPILED_EXTENSION.accumulate_dft_batched(
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
    _COMPILED_EXTENSION.accumulate_point_observers(
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
    _COMPILED_EXTENSION.accumulate_plane_observer(
                field,
                planeRealAccum,
                planeImagAccum,
                int(axisCode),
                int(planeIndex),
                float(weightedCos),
                float(weightedSin),
            )


def _plane_flux_reduce(*, ea, eb, ha, hb, weights, out, outIndex, scale):
    _COMPILED_EXTENSION.plane_flux_reduce(
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
    _COMPILED_EXTENSION.update_debye_current(
                ElectricField,
                Polarization,
                PolarizationCurrent,
                DebyeDrive,
                float(decay),
                float(dt),
            )


def _update_drude_current(*, ElectricField, PolarizationCurrent, DrudeDrive, decay):
    _COMPILED_EXTENSION.update_drude_current(
                ElectricField,
                PolarizationCurrent,
                DrudeDrive,
                float(decay),
            )


def _update_lorentz_current(*, ElectricField, Polarization, PolarizationCurrent, LorentzDrive, decay, restoring, dt):
    _COMPILED_EXTENSION.update_lorentz_current(
                ElectricField,
                Polarization,
                PolarizationCurrent,
                LorentzDrive,
                float(decay),
                float(restoring),
                float(dt),
            )


def _apply_polarization_current(*, ElectricField, PolarizationCurrent, InvPermittivity, dt):
    _COMPILED_EXTENSION.apply_polarization_current(
                ElectricField,
                PolarizationCurrent,
                InvPermittivity,
                float(dt),
            )


def _apply_polarization_current_modulated(
    *, ElectricField, PolarizationCurrent, InvPermittivity, ModCos, ModSin, ModOmega, ModulationTime, dt
):
    _COMPILED_EXTENSION.apply_polarization_current_modulated(
                ElectricField,
                PolarizationCurrent,
                InvPermittivity,
                ModCos,
                ModSin,
                ModOmega,
                ModulationTime,
                float(dt),
            )


def _advance_modulation_time(*, ModulationTime, dt):
    _COMPILED_EXTENSION.advance_modulation_time(ModulationTime, float(dt))


def _update_kerr_ex(*, DynamicCurl, Ex, Ey, Ez, LinearPermittivity, ExDecay, KerrChi3, dt, eps0):
    _COMPILED_EXTENSION.update_kerr_ex_curl(
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
    _COMPILED_EXTENSION.update_kerr_ey_curl(
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
    _COMPILED_EXTENSION.update_kerr_ez_curl(
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
    _COMPILED_EXTENSION.update_nonlinear_coefficients(
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
    _COMPILED_EXTENSION.update_electric_ex_full_aniso(
                Ex, Hx, Hy, Hz, CoeffY, CoeffZ, invDx, invDy, invDz,
                int(periodicX), int(periodicY), int(periodicZ),
            )


def _electric_ey_full_aniso(*, Ey, Hx, Hy, Hz, CoeffX, CoeffZ, invDx, invDy, invDz, periodicX, periodicY, periodicZ):
    _COMPILED_EXTENSION.update_electric_ey_full_aniso(
                Ey, Hx, Hy, Hz, CoeffX, CoeffZ, invDx, invDy, invDz,
                int(periodicX), int(periodicY), int(periodicZ),
            )


def _electric_ez_full_aniso(*, Ez, Hx, Hy, Hz, CoeffX, CoeffY, invDx, invDy, invDz, periodicX, periodicY, periodicZ):
    _COMPILED_EXTENSION.update_electric_ez_full_aniso(
                Ez, Hx, Hy, Hz, CoeffX, CoeffY, invDx, invDy, invDz,
                int(periodicX), int(periodicY), int(periodicZ),
            )


def _electric_ex_full_aniso_cpml(
    *,
    Ex, Hx, Hy, Hz, CoeffY, CoeffZ, invDx, invDy, invDz,
    InvKappaX, BX, CX, InvKappaY, BY, CY, InvKappaZ, BZ, CZ,
    periodicX, periodicY, periodicZ, PsiX, PsiY, PsiZ,
):
    _COMPILED_EXTENSION.update_electric_ex_full_aniso_cpml(
                Ex, Hx, Hy, Hz, CoeffY, CoeffZ, invDx, invDy, invDz,
                InvKappaX, BX, CX, InvKappaY, BY, CY, InvKappaZ, BZ, CZ,
                int(periodicX), int(periodicY), int(periodicZ), PsiX, PsiY, PsiZ,
            )


def _electric_ey_full_aniso_cpml(
    *,
    Ey, Hx, Hy, Hz, CoeffX, CoeffZ, invDx, invDy, invDz,
    InvKappaX, BX, CX, InvKappaY, BY, CY, InvKappaZ, BZ, CZ,
    periodicX, periodicY, periodicZ, PsiX, PsiY, PsiZ,
):
    _COMPILED_EXTENSION.update_electric_ey_full_aniso_cpml(
                Ey, Hx, Hy, Hz, CoeffX, CoeffZ, invDx, invDy, invDz,
                InvKappaX, BX, CX, InvKappaY, BY, CY, InvKappaZ, BZ, CZ,
                int(periodicX), int(periodicY), int(periodicZ), PsiX, PsiY, PsiZ,
            )


def _electric_ez_full_aniso_cpml(
    *,
    Ez, Hx, Hy, Hz, CoeffX, CoeffY, invDx, invDy, invDz,
    InvKappaX, BX, CX, InvKappaY, BY, CY, InvKappaZ, BZ, CZ,
    periodicX, periodicY, periodicZ, PsiX, PsiY, PsiZ,
):
    _COMPILED_EXTENSION.update_electric_ez_full_aniso_cpml(
                Ez, Hx, Hy, Hz, CoeffX, CoeffY, invDx, invDy, invDz,
                InvKappaX, BX, CX, InvKappaY, BY, CY, InvKappaZ, BZ, CZ,
                int(periodicX), int(periodicY), int(periodicZ), PsiX, PsiY, PsiZ,
            )


def _aniso_offdiag_current_ex(*, Ex, Jy, Jz, CoeffY, CoeffZ, periodicX, periodicY, periodicZ):
    _COMPILED_EXTENSION.apply_aniso_offdiag_current_ex(
                Ex, Jy, Jz, CoeffY, CoeffZ,
                int(periodicX), int(periodicY), int(periodicZ),
            )


def _capture_aniso_conduction_current(*, SigmaX, SigmaY, SigmaZ, Ex, Ey, Ez, Jx, Jy, Jz):
    _COMPILED_EXTENSION.capture_aniso_conduction_current(
        SigmaX, SigmaY, SigmaZ, Ex, Ey, Ez, Jx, Jy, Jz
    )


def _aniso_offdiag_current_ey(*, Ey, Jx, Jz, CoeffX, CoeffZ, periodicX, periodicY, periodicZ):
    _COMPILED_EXTENSION.apply_aniso_offdiag_current_ey(
                Ey, Jx, Jz, CoeffX, CoeffZ,
                int(periodicX), int(periodicY), int(periodicZ),
            )


def _aniso_offdiag_current_ez(*, Ez, Jx, Jy, CoeffX, CoeffY, periodicX, periodicY, periodicZ):
    _COMPILED_EXTENSION.apply_aniso_offdiag_current_ez(
                Ez, Jx, Jy, CoeffX, CoeffY,
                int(periodicX), int(periodicY), int(periodicZ),
            )


def _reverse_electric_hx_standard(*, AdjHxMid, AdjHxPost, AdjEyPost, AdjEzPost, EyCurl, EzCurl, invDy, invDz):
    _COMPILED_EXTENSION.reverse_electric_adjoint_to_hx_standard(
                AdjHxMid, AdjHxPost, AdjEyPost, AdjEzPost, EyCurl, EzCurl, invDy, invDz
            )


def _reverse_electric_hy_standard(*, AdjHyMid, AdjHyPost, AdjExPost, AdjEzPost, ExCurl, EzCurl, invDx, invDz):
    _COMPILED_EXTENSION.reverse_electric_adjoint_to_hy_standard(
                AdjHyMid, AdjHyPost, AdjExPost, AdjEzPost, ExCurl, EzCurl, invDx, invDz
            )


def _reverse_electric_hz_standard(*, AdjHzMid, AdjHzPost, AdjExPost, AdjEyPost, ExCurl, EyCurl, invDx, invDy):
    _COMPILED_EXTENSION.reverse_electric_adjoint_to_hz_standard(
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
    _COMPILED_EXTENSION.reverse_electric_adjoint_to_hx_bloch(
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
    _COMPILED_EXTENSION.reverse_electric_adjoint_to_hy_bloch(
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
    _COMPILED_EXTENSION.reverse_electric_adjoint_to_hz_bloch(
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
    _COMPILED_EXTENSION.reverse_magnetic_adjoint_to_ex_standard(
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
    _COMPILED_EXTENSION.reverse_magnetic_adjoint_to_ey_standard(
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
    _COMPILED_EXTENSION.reverse_magnetic_adjoint_to_ez_standard(
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
    _COMPILED_EXTENSION.reverse_magnetic_adjoint_to_ex_bloch(
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
    _COMPILED_EXTENSION.reverse_magnetic_adjoint_to_ey_bloch(
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
    _COMPILED_EXTENSION.reverse_magnetic_adjoint_to_ez_bloch(
                AdjEzPrevReal, AdjEzPrevImag, GradEpsEz, AdjEzPostReal, AdjEzPostImag,
                AdjHxMidReal, AdjHxMidImag, AdjHyMidReal, AdjHyMidImag,
                EzDecay, EzCurl, EpsEz, HxMidReal, HxMidImag, HyMidReal, HyMidImag,
                HxCurl, HyCurl, float(phaseCosX), float(phaseSinX), float(phaseCosY), float(phaseSinY),
                invDxE, invDyE, invDxH, invDyH,
            )


def _accumulate_diff_adjoint(field_grad, diff_grad, axis, inv_delta, *, forward):
    method = (
                _COMPILED_EXTENSION.accumulate_forward_diff_adjoint
                if forward
                else _COMPILED_EXTENSION.accumulate_backward_diff_adjoint
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


def _seed_inject_dense(*, AdjField, GradReal, GradImag, CosPack, SinPack, step):
    _COMPILED_EXTENSION.seed_inject_dense(AdjField, GradReal, GradImag, CosPack, SinPack, int(step))


def _seed_inject_point(*, AdjField, GradReal, GradImag, PointI, PointJ, PointK, CosPack, SinPack, step):
    _COMPILED_EXTENSION.seed_inject_point(
        AdjField, GradReal, GradImag, PointI, PointJ, PointK, CosPack, SinPack, int(step)
    )


def _seed_inject_plane(*, AdjField, GradReal, GradImag, CosPack, SinPack, axis, planeIndex, step):
    _COMPILED_EXTENSION.seed_inject_plane(
        AdjField, GradReal, GradImag, CosPack, SinPack, int(axis), int(planeIndex), int(step)
    )


def _accumulate_in_place(*, dst, src):
    _COMPILED_EXTENSION.accumulate_in_place(dst, src)


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
    _COMPILED_EXTENSION.reverse_electric_component_ex_cpml(
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
    _COMPILED_EXTENSION.reverse_electric_component_ey_cpml(
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
    _COMPILED_EXTENSION.reverse_electric_component_ez_cpml(
                AdjEzPrev, GradEpsEz, AdjPsiPosPrev, AdjPsiNegPrev, AdjDPos, AdjDNeg,
                AdjEzPost, AdjPsiPosPost, AdjPsiNegPost, EzDecay, EzCurl, EpsEz,
                PsiPos, PsiNeg, BPos, CPos, InvKappaPos, BNeg, CNeg, InvKappaNeg,
                HxMid, HyMid, invDx, invDy, int(xLowBoundaryMode), int(xHighBoundaryMode), int(yLowBoundaryMode), int(yHighBoundaryMode),
            )


def _reverse_electric_cpml_conductive_ex(
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
    ExHalf,
    ExPrev,
    EpsEx,
    Dt,
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
    _COMPILED_EXTENSION.reverse_electric_component_ex_cpml_conductive(
                AdjExPrev, GradEpsEx, AdjPsiPosPrev, AdjPsiNegPrev, AdjDPos, AdjDNeg,
                AdjExPost, AdjPsiPosPost, AdjPsiNegPost, ExDecay, ExCurl, ExHalf, ExPrev, EpsEx, float(Dt),
                PsiPos, PsiNeg, BPos, CPos, InvKappaPos, BNeg, CNeg, InvKappaNeg,
                HyMid, HzMid, invDy, invDz, int(yLowBoundaryMode), int(yHighBoundaryMode), int(zLowBoundaryMode), int(zHighBoundaryMode),
            )


def _reverse_electric_cpml_conductive_ey(
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
    EyHalf,
    EyPrev,
    EpsEy,
    Dt,
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
    _COMPILED_EXTENSION.reverse_electric_component_ey_cpml_conductive(
                AdjEyPrev, GradEpsEy, AdjPsiPosPrev, AdjPsiNegPrev, AdjDPos, AdjDNeg,
                AdjEyPost, AdjPsiPosPost, AdjPsiNegPost, EyDecay, EyCurl, EyHalf, EyPrev, EpsEy, float(Dt),
                PsiPos, PsiNeg, BPos, CPos, InvKappaPos, BNeg, CNeg, InvKappaNeg,
                HxMid, HzMid, invDx, invDz, int(xLowBoundaryMode), int(xHighBoundaryMode), int(zLowBoundaryMode), int(zHighBoundaryMode),
            )


def _reverse_electric_cpml_conductive_ez(
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
    EzHalf,
    EzPrev,
    EpsEz,
    Dt,
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
    _COMPILED_EXTENSION.reverse_electric_component_ez_cpml_conductive(
                AdjEzPrev, GradEpsEz, AdjPsiPosPrev, AdjPsiNegPrev, AdjDPos, AdjDNeg,
                AdjEzPost, AdjPsiPosPost, AdjPsiNegPost, EzDecay, EzCurl, EzHalf, EzPrev, EpsEz, float(Dt),
                PsiPos, PsiNeg, BPos, CPos, InvKappaPos, BNeg, CNeg, InvKappaNeg,
                HxMid, HyMid, invDx, invDy, int(xLowBoundaryMode), int(xHighBoundaryMode), int(yLowBoundaryMode), int(yHighBoundaryMode),
            )


def _collocation_transpose(
    *,
    AdjEx,
    AdjEy,
    AdjEz,
    GEx,
    GEy,
    GEz,
    Ex,
    Ey,
    Ez,
):
    _COMPILED_EXTENSION.collocation_transpose(
                AdjEx, AdjEy, AdjEz, GEx, GEy, GEz, Ex, Ey, Ez,
            )


def _collocate_field_square(*, FsqEx, FsqEy, FsqEz, Ex, Ey, Ez):
    _COMPILED_EXTENSION.collocate_field_square(FsqEx, FsqEy, FsqEz, Ex, Ey, Ez)


def _full_aniso_curl_adjoint(
    *,
    AdjCurlX,
    AdjCurlY,
    AdjCurlZ,
    AdjEx,
    AdjEy,
    AdjEz,
    CoeffExY,
    CoeffExZ,
    CoeffEyX,
    CoeffEyZ,
    CoeffEzX,
    CoeffEzY,
):
    _COMPILED_EXTENSION.full_aniso_curl_adjoint(
                AdjCurlX,
                AdjCurlY,
                AdjCurlZ,
                AdjEx,
                AdjEy,
                AdjEz,
                CoeffExY,
                CoeffExZ,
                CoeffEyX,
                CoeffEyZ,
                CoeffEzX,
                CoeffEzY,
            )


def _reverse_electric_cpml_kerr_ex(
    *,
    AdjExPrev,
    GradEpsEx,
    GradChi3Ex,
    GFsqEx,
    AdjPsiPosPrev,
    AdjPsiNegPrev,
    AdjDPos,
    AdjDNeg,
    AdjExPost,
    AdjPsiPosPost,
    AdjPsiNegPost,
    ExDecay,
    EpsEx,
    Chi3Ex,
    FsqEx,
    Dt,
    Eps0,
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
    _COMPILED_EXTENSION.reverse_electric_component_ex_cpml_kerr(
                AdjExPrev, GradEpsEx, GradChi3Ex, GFsqEx, AdjPsiPosPrev, AdjPsiNegPrev, AdjDPos, AdjDNeg,
                AdjExPost, AdjPsiPosPost, AdjPsiNegPost, ExDecay, EpsEx, Chi3Ex, FsqEx, float(Dt), float(Eps0),
                PsiPos, PsiNeg, BPos, CPos, InvKappaPos, BNeg, CNeg, InvKappaNeg,
                HyMid, HzMid, invDy, invDz, int(yLowBoundaryMode), int(yHighBoundaryMode), int(zLowBoundaryMode), int(zHighBoundaryMode),
            )


def _reverse_electric_cpml_kerr_ey(
    *,
    AdjEyPrev,
    GradEpsEy,
    GradChi3Ey,
    GFsqEy,
    AdjPsiPosPrev,
    AdjPsiNegPrev,
    AdjDPos,
    AdjDNeg,
    AdjEyPost,
    AdjPsiPosPost,
    AdjPsiNegPost,
    EyDecay,
    EpsEy,
    Chi3Ey,
    FsqEy,
    Dt,
    Eps0,
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
    _COMPILED_EXTENSION.reverse_electric_component_ey_cpml_kerr(
                AdjEyPrev, GradEpsEy, GradChi3Ey, GFsqEy, AdjPsiPosPrev, AdjPsiNegPrev, AdjDPos, AdjDNeg,
                AdjEyPost, AdjPsiPosPost, AdjPsiNegPost, EyDecay, EpsEy, Chi3Ey, FsqEy, float(Dt), float(Eps0),
                PsiPos, PsiNeg, BPos, CPos, InvKappaPos, BNeg, CNeg, InvKappaNeg,
                HxMid, HzMid, invDx, invDz, int(xLowBoundaryMode), int(xHighBoundaryMode), int(zLowBoundaryMode), int(zHighBoundaryMode),
            )


def _reverse_electric_cpml_kerr_ez(
    *,
    AdjEzPrev,
    GradEpsEz,
    GradChi3Ez,
    GFsqEz,
    AdjPsiPosPrev,
    AdjPsiNegPrev,
    AdjDPos,
    AdjDNeg,
    AdjEzPost,
    AdjPsiPosPost,
    AdjPsiNegPost,
    EzDecay,
    EpsEz,
    Chi3Ez,
    FsqEz,
    Dt,
    Eps0,
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
    _COMPILED_EXTENSION.reverse_electric_component_ez_cpml_kerr(
                AdjEzPrev, GradEpsEz, GradChi3Ez, GFsqEz, AdjPsiPosPrev, AdjPsiNegPrev, AdjDPos, AdjDNeg,
                AdjEzPost, AdjPsiPosPost, AdjPsiNegPost, EzDecay, EpsEz, Chi3Ez, FsqEz, float(Dt), float(Eps0),
                PsiPos, PsiNeg, BPos, CPos, InvKappaPos, BNeg, CNeg, InvKappaNeg,
                HxMid, HyMid, invDx, invDy, int(xLowBoundaryMode), int(xHighBoundaryMode), int(yLowBoundaryMode), int(yHighBoundaryMode),
            )


def _reverse_electric_cpml_nonlinear(**kwargs):
    _COMPILED_EXTENSION.reverse_electric_component_cpml_nonlinear(
        int(kwargs["component"]),
        kwargs["AdjPrev"], kwargs["GradEps"], kwargs["GradChi2"],
        kwargs["GradChi3"], kwargs["GradTpa"], kwargs["GFsq"],
        kwargs["AdjPsiPosPrev"], kwargs["AdjPsiNegPrev"],
        kwargs["AdjDPos"], kwargs["AdjDNeg"], kwargs["AdjPost"],
        kwargs["AdjPsiPosPost"], kwargs["AdjPsiNegPost"], kwargs["EPrev"],
        kwargs["ExternalDecay"], kwargs["Eps"], kwargs["Chi2"],
        kwargs["Chi3"], kwargs["Tpa"], kwargs["SigmaStatic"], kwargs["Fsq"],
        float(kwargs["Dt"]), float(kwargs["Eps0"]), kwargs["PsiPos"],
        kwargs["PsiNeg"], kwargs["BPos"], kwargs["CPos"],
        kwargs["InvKappaPos"], kwargs["BNeg"], kwargs["CNeg"],
        kwargs["InvKappaNeg"], kwargs["HPosMid"], kwargs["HNegMid"],
        kwargs["InvPos"], kwargs["InvNeg"], int(kwargs["LowModeA"]),
        int(kwargs["HighModeA"]), int(kwargs["LowModeB"]), int(kwargs["HighModeB"]),
    )


def _reverse_cpml_correction(**kwargs):
    _COMPILED_EXTENSION.reverse_cpml_correction(
        kwargs["AdjPsiPrev"], kwargs["AdjDerivative"], kwargs["AdjField"],
        kwargs["AdjPsiPost"], kwargs["Curl"], kwargs["B"], kwargs["C"],
        kwargs["InvKappa"], int(kwargs["NormalAxis"]), int(kwargs["TangentAxis"]),
        int(kwargs["TangentLowMode"]), int(kwargs["TangentHighMode"]), float(kwargs["Sign"]),
    )


def _reverse_magnetic_cpml_hx(**kwargs):
    _COMPILED_EXTENSION.reverse_magnetic_component_hx_cpml(
                kwargs["AdjHxPrev"], kwargs["AdjPsiPosPrev"], kwargs["AdjPsiNegPrev"], kwargs["AdjDPos"], kwargs["AdjDNeg"],
                kwargs["AdjHxPost"], kwargs["AdjPsiPosPost"], kwargs["AdjPsiNegPost"], kwargs["HxDecay"], kwargs["HxCurl"],
                kwargs["BPos"], kwargs["CPos"], kwargs["InvKappaPos"], kwargs["BNeg"], kwargs["CNeg"], kwargs["InvKappaNeg"],
            )


def _reverse_magnetic_cpml_hy(**kwargs):
    _COMPILED_EXTENSION.reverse_magnetic_component_hy_cpml(
                kwargs["AdjHyPrev"], kwargs["AdjPsiPosPrev"], kwargs["AdjPsiNegPrev"], kwargs["AdjDPos"], kwargs["AdjDNeg"],
                kwargs["AdjHyPost"], kwargs["AdjPsiPosPost"], kwargs["AdjPsiNegPost"], kwargs["HyDecay"], kwargs["HyCurl"],
                kwargs["BPos"], kwargs["CPos"], kwargs["InvKappaPos"], kwargs["BNeg"], kwargs["CNeg"], kwargs["InvKappaNeg"],
            )


def _reverse_magnetic_cpml_hz(**kwargs):
    _COMPILED_EXTENSION.reverse_magnetic_component_hz_cpml(
                kwargs["AdjHzPrev"], kwargs["AdjPsiPosPrev"], kwargs["AdjPsiNegPrev"], kwargs["AdjDPos"], kwargs["AdjDNeg"],
                kwargs["AdjHzPost"], kwargs["AdjPsiPosPost"], kwargs["AdjPsiNegPost"], kwargs["HzDecay"], kwargs["HzCurl"],
                kwargs["BPos"], kwargs["CPos"], kwargs["InvKappaPos"], kwargs["BNeg"], kwargs["CNeg"], kwargs["InvKappaNeg"],
            )


def _reverse_magnetic_hx_decay(*, AdjHxPrev, AdjHxMid, HxDecay):
    _COMPILED_EXTENSION.reverse_magnetic_adjoint_decay(AdjHxPrev, AdjHxMid, HxDecay)


def _reverse_magnetic_hy_decay(*, AdjHyPrev, AdjHyMid, HyDecay):
    _COMPILED_EXTENSION.reverse_magnetic_adjoint_decay(AdjHyPrev, AdjHyMid, HyDecay)


def _reverse_magnetic_hz_decay(*, AdjHzPrev, AdjHzMid, HzDecay):
    _COMPILED_EXTENSION.reverse_magnetic_adjoint_decay(AdjHzPrev, AdjHzMid, HzDecay)


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
    _COMPILED_EXTENSION.reverse_debye_current(
                AdjElectricPrev,
                AdjPolarizationPrev,
                AdjPolarizationPost,
                AdjCurrentPost,
                DebyeDrive,
                float(decay),
                float(dt),
            )


def _reverse_drude_current(*, AdjElectricPrev, AdjCurrentPrev, AdjCurrentPost, DrudeDrive, decay):
    _COMPILED_EXTENSION.reverse_drude_current(
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
    _COMPILED_EXTENSION.reverse_lorentz_current(
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


def _reverse_dispersive_correction(
    *,
    AdjCurrentCorrected,
    GradEps,
    AdjCurrentPost,
    AdjElectricPost,
    Current,
    Eps,
    dt,
):
    _COMPILED_EXTENSION.reverse_dispersive_correction(
                AdjCurrentCorrected,
                GradEps,
                AdjCurrentPost,
                AdjElectricPost,
                Current,
                Eps,
                float(dt),
            )


def _accumulate_tfsf_scalar_sample_adjoint(*, AdjAuxField, AdjFieldPatch, CoeffPatch, sampleIndex, componentScale):
    _COMPILED_EXTENSION.accumulate_tfsf_scalar_sample_adjoint(
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
    _COMPILED_EXTENSION.accumulate_tfsf_line_sample_adjoint(
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
    _COMPILED_EXTENSION.accumulate_tfsf_interpolated_sample_adjoint(
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
    _COMPILED_EXTENSION.reverse_tfsf_auxiliary_electric(
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
    _COMPILED_EXTENSION.reverse_tfsf_auxiliary_magnetic(
                AdjElectricPrev,
                AdjMagneticPrev,
                AdjMagneticAfter,
                MagneticDecay,
                MagneticCurl,
            )


def _clamp_field_face(*, field, axis, side):
    _COMPILED_EXTENSION.clamp_field_face(field, int(axis), int(side))


def _clamp_pec_boundary(*, field, axisA, axisB):
    _COMPILED_EXTENSION.clamp_pec_boundary(field, int(axisA), int(axisB))


def _mur_abc_face(*, field, axis, boundaryIndex, adjacentIndex, coef, prevBoundary, prevAdjacent):
    _COMPILED_EXTENSION.mur_abc_face(
        field,
        int(axis),
        int(boundaryIndex),
        int(adjacentIndex),
        float(coef),
        prevBoundary,
        prevAdjacent,
    )


def _apply_sibc_surface(
    *, electric, magnetic, axis, electricIndex, magneticIndex, sign, surfaceR, surfaceLOverDt, hPrev
):
    _COMPILED_EXTENSION.apply_sibc_surface(
        electric,
        magnetic,
        int(axis),
        int(electricIndex),
        int(magneticIndex),
        float(sign),
        float(surfaceR),
        float(surfaceLOverDt),
        hPrev,
    )


def _sample_wire_emf(
    *, Ex, Ey, Ez, segmentOffsets, edgeComponents, edgeOffsets, weights, emf
):
    """Internal launch; compiled topology contents are validated once at preparation."""
    _COMPILED_EXTENSION.sample_wire_emf(
        Ex,
        Ey,
        Ez,
        segmentOffsets,
        edgeComponents,
        edgeOffsets,
        weights,
        emf,
    )


def _update_wire_state(
    *,
    emf,
    tail,
    head,
    inductance,
    nodeCapacitance,
    grounded,
    nodeOffsets,
    nodeSegments,
    nodeSigns,
    dt,
    current,
    charge,
):
    """Internal launch; compiled topology contents are validated once at preparation."""
    _COMPILED_EXTENSION.update_wire_state(
        emf,
        tail,
        head,
        inductance,
        nodeCapacitance,
        grounded,
        nodeOffsets,
        nodeSegments,
        nodeSigns,
        float(dt),
        current,
        charge,
    )


def _deposit_wire_current(
    *,
    Ex,
    Ey,
    Ez,
    edgeGroupOffsets,
    targetComponents,
    targetOffsets,
    contributionSegments,
    contributionScales,
    current,
):
    """Internal launch; compiled topology contents are validated once at preparation."""
    _COMPILED_EXTENSION.deposit_wire_current(
        Ex,
        Ey,
        Ez,
        edgeGroupOffsets,
        targetComponents,
        targetOffsets,
        contributionSegments,
        contributionScales,
        current,
    )


def _project_periodic_boundary(*, field, axis):
    _COMPILED_EXTENSION.project_periodic_boundary(field, int(axis))


def _project_bloch_boundary(*, fieldReal, fieldImag, axis, phaseCos, phaseSin):
    _COMPILED_EXTENSION.project_bloch_boundary(
                fieldReal,
                fieldImag,
                int(axis),
                float(phaseCos),
                float(phaseSin),
            )


_KERNELS: dict[str, Callable[..., None]] = {
    "sampleWireEmf3D": _sample_wire_emf,
    "updateWireState1D": _update_wire_state,
    "depositWireCurrent3D": _deposit_wire_current,
    "updateMagneticFieldHxStandard3D": _magnetic_hx_standard,
    "updateMagneticFieldHyStandard3D": _magnetic_hy_standard,
    "updateMagneticFieldHzStandard3D": _magnetic_hz_standard,
    "updateMagneticFieldHxStandardBounded3D": _magnetic_hx_standard_bounded,
    "updateMagneticFieldHyStandardBounded3D": _magnetic_hy_standard_bounded,
    "updateMagneticFieldHzStandardBounded3D": _magnetic_hz_standard_bounded,
    "updateMagneticFieldHx3D": _magnetic_hx_cpml,
    "updateMagneticFieldHy3D": _magnetic_hy_cpml,
    "updateMagneticFieldHz3D": _magnetic_hz_cpml,
    "updateMagneticFieldHxCpmlCompressed3D": _magnetic_hx_cpml_compressed,
    "updateMagneticFieldHyCpmlCompressed3D": _magnetic_hy_cpml_compressed,
    "updateMagneticFieldHzCpmlCompressed3D": _magnetic_hz_cpml_compressed,
    "updateElectricFieldExStandard3D": _electric_ex_standard,
    "updateElectricFieldExStandardBounded3D": _electric_ex_standard_bounded,
    "advanceModulationTime3D": _advance_modulation_time,
    "updateElectricFieldExModulated3D": _electric_ex_modulated,
    "updateElectricFieldEyModulated3D": _electric_ey_modulated,
    "updateElectricFieldEzModulated3D": _electric_ez_modulated,
    "updateElectricFieldExCpmlModulated3D": _electric_ex_cpml_modulated,
    "updateElectricFieldEyCpmlModulated3D": _electric_ey_cpml_modulated,
    "updateElectricFieldEzCpmlModulated3D": _electric_ez_cpml_modulated,
    "updateElectricFieldEyStandard3D": _electric_ey_standard,
    "updateElectricFieldEzStandard3D": _electric_ez_standard,
    "updateElectricFieldEyStandardBounded3D": _electric_ey_standard_bounded,
    "updateElectricFieldEzStandardBounded3D": _electric_ez_standard_bounded,
    "updateElectricFieldExCpml3D": _electric_ex_cpml,
    "updateElectricFieldEyCpml3D": _electric_ey_cpml,
    "updateElectricFieldEzCpml3D": _electric_ez_cpml,
    "updateElectricFieldExCpmlCompressed3D": _electric_ex_cpml_compressed,
    "updateElectricFieldEyCpmlCompressed3D": _electric_ey_cpml_compressed,
    "updateElectricFieldEzCpmlCompressed3D": _electric_ez_cpml_compressed,
    "updateElectricFieldExCpmlModulatedCompressed3D": _electric_ex_cpml_modulated_compressed,
    "updateElectricFieldEyCpmlModulatedCompressed3D": _electric_ey_cpml_modulated_compressed,
    "updateElectricFieldEzCpmlModulatedCompressed3D": _electric_ez_cpml_modulated_compressed,
    "applyElectricFieldExCpmlZCorrection3D": _apply_electric_ex_cpml_z_correction,
    "applyElectricFieldEyCpmlZCorrection3D": _apply_electric_ey_cpml_z_correction,
    "applyElectricFieldExCpmlYCorrection3D": _apply_electric_ex_cpml_y_correction,
    "applyElectricFieldEzCpmlYCorrection3D": _apply_electric_ez_cpml_y_correction,
    "applyElectricFieldEyCpmlXCorrection3D": _apply_electric_ey_cpml_x_correction,
    "applyElectricFieldEzCpmlXCorrection3D": _apply_electric_ez_cpml_x_correction,
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
    "applyPolarizationCurrentModulated3D": _apply_polarization_current_modulated,
    "updateKerrElectricFieldExCurl3D": _update_kerr_ex,
    "updateKerrElectricFieldEyCurl3D": _update_kerr_ey,
    "updateKerrElectricFieldEzCurl3D": _update_kerr_ez,
    "updateNonlinearElectricCoefficients3D": _update_nonlinear_coefficients,
    "updateElectricFieldExFullAniso3D": _electric_ex_full_aniso,
    "updateElectricFieldEyFullAniso3D": _electric_ey_full_aniso,
    "updateElectricFieldEzFullAniso3D": _electric_ez_full_aniso,
    "updateElectricFieldExFullAnisoCpml3D": _electric_ex_full_aniso_cpml,
    "updateElectricFieldEyFullAnisoCpml3D": _electric_ey_full_aniso_cpml,
    "updateElectricFieldEzFullAnisoCpml3D": _electric_ez_full_aniso_cpml,
    "captureAnisoConductionCurrent3D": _capture_aniso_conduction_current,
    "applyAnisoOffdiagCurrentEx3D": _aniso_offdiag_current_ex,
    "applyAnisoOffdiagCurrentEy3D": _aniso_offdiag_current_ey,
    "applyAnisoOffdiagCurrentEz3D": _aniso_offdiag_current_ez,
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
    "seedInjectDense3D": _seed_inject_dense,
    "seedInjectPoint3D": _seed_inject_point,
    "seedInjectPlane3D": _seed_inject_plane,
    "accumulateInPlace3D": _accumulate_in_place,
    "reverseElectricComponentExCpml3D": _reverse_electric_cpml_ex,
    "reverseElectricComponentEyCpml3D": _reverse_electric_cpml_ey,
    "reverseElectricComponentEzCpml3D": _reverse_electric_cpml_ez,
    "reverseElectricComponentExCpmlConductive3D": _reverse_electric_cpml_conductive_ex,
    "reverseElectricComponentEyCpmlConductive3D": _reverse_electric_cpml_conductive_ey,
    "reverseElectricComponentEzCpmlConductive3D": _reverse_electric_cpml_conductive_ez,
    "collocationTranspose3D": _collocation_transpose,
    "collocateFieldSquare3D": _collocate_field_square,
    "fullAnisoCurlAdjoint3D": _full_aniso_curl_adjoint,
    "reverseElectricComponentExCpmlKerr3D": _reverse_electric_cpml_kerr_ex,
    "reverseElectricComponentEyCpmlKerr3D": _reverse_electric_cpml_kerr_ey,
    "reverseElectricComponentEzCpmlKerr3D": _reverse_electric_cpml_kerr_ez,
    "reverseElectricComponentCpmlNonlinear3D": _reverse_electric_cpml_nonlinear,
    "reverseCpmlCorrection3D": _reverse_cpml_correction,
    "reverseMagneticComponentHxCpml3D": _reverse_magnetic_cpml_hx,
    "reverseMagneticComponentHyCpml3D": _reverse_magnetic_cpml_hy,
    "reverseMagneticComponentHzCpml3D": _reverse_magnetic_cpml_hz,
    "reverseDebyeCurrent3D": _reverse_debye_current,
    "reverseDrudeCurrent3D": _reverse_drude_current,
    "reverseLorentzCurrent3D": _reverse_lorentz_current,
    "reverseDispersiveCorrection3D": _reverse_dispersive_correction,
    "accumulateTfsfScalarSampleAdjoint3D": _accumulate_tfsf_scalar_sample_adjoint,
    "accumulateTfsfLineSampleAdjoint3D": _accumulate_tfsf_line_sample_adjoint,
    "accumulateTfsfInterpolatedSampleAdjoint3D": _accumulate_tfsf_interpolated_sample_adjoint,
    "reverseTfsfAuxiliaryElectric1D": _reverse_tfsf_auxiliary_electric,
    "reverseTfsfAuxiliaryMagnetic1D": _reverse_tfsf_auxiliary_magnetic,
    "clampFieldFace3D": _clamp_field_face,
    "clampPecBoundary3D": _clamp_pec_boundary,
    "applyMurBoundary3D": _mur_abc_face,
    "applySibcSurface3D": _apply_sibc_surface,
    "projectPeriodicBoundary3D": _project_periodic_boundary,
    "projectBlochBoundary3D": _project_bloch_boundary,
}
