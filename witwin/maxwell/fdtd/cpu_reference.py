"""Torch CPU reference implementation of the native FDTD kernel subset.

This backend exists for the explicit ``Simulation.fdtd(..., backend="cpu_reference")``
profile. It executes the ordinary Scene -> Simulation -> Result pipeline on a CPU
device for small validation problems: real fields, Cartesian non-periodic
boundaries (CPML/PEC/PMC/none), dense CPML memory, soft electric sources and the
point/plane/full-field running-DFT observers. Every other kernel raises a clear
``NotImplementedError`` at the first launch, so an unsupported feature can never
silently skip work.

The arithmetic mirrors the native CUDA kernels (``fdtd/cuda/kernels``): the H
update is a forward difference, the E update a backward difference whose outer
boundary planes are frozen for none/PML, zeroed for PEC and mirrored for PMC.
Coefficient uniformity hints are ignored because the full coefficient arrays
always exist. The class is deliberately not named ``NativeFDTDModule`` so the
existing boundary clamps take their Torch fallback.
"""
from __future__ import annotations

import torch

from .boundary.common import BOUNDARY_NONE, BOUNDARY_PEC, BOUNDARY_PERIODIC, BOUNDARY_PMC, BOUNDARY_PML


def _view(values, axis):
    shape = [1, 1, 1]
    shape[axis] = -1
    return values.reshape(shape)


def _forward(field, axis, inv):
    upper = field.narrow(axis, 1, field.shape[axis] - 1)
    lower = field.narrow(axis, 0, field.shape[axis] - 1)
    return (upper - lower) * _view(inv, axis)


def _backward(source, shape, axis, low_mode, high_mode, inv):
    if low_mode in (BOUNDARY_PERIODIC,) or high_mode in (BOUNDARY_PERIODIC,):
        raise NotImplementedError("The CPU reference FDTD backend does not support periodic boundaries.")
    result = source.new_zeros(shape)
    size = shape[axis]
    if size > 2:
        interior = (source.narrow(axis, 1, size - 2) - source.narrow(axis, 0, size - 2))
        result.narrow(axis, 1, size - 2).copy_(interior * _view(inv[1:size - 1], axis))
    if low_mode == BOUNDARY_PMC:
        result.narrow(axis, 0, 1).copy_(2.0 * source.narrow(axis, 0, 1) * inv[0])
    if size > 1 and high_mode == BOUNDARY_PMC:
        last = source.shape[axis] - 1
        result.narrow(axis, size - 1, 1).copy_(-2.0 * source.narrow(axis, last, 1) * inv[size - 1])
    return result


_MASK_CACHE = {}


def _boundary_masks(shape, specs, device):
    key = (tuple(shape), tuple(specs), str(device))
    cached = _MASK_CACHE.get(key)
    if cached is not None:
        return cached
    pec = torch.zeros(shape, dtype=torch.bool, device=device)
    inactive = torch.zeros(shape, dtype=torch.bool, device=device)
    for axis, low, high in specs:
        for index, mode in ((0, int(low)), (shape[axis] - 1, int(high))):
            if index < 0:
                continue
            plane = [slice(None)] * 3
            plane[axis] = index
            if mode == BOUNDARY_PEC:
                pec[tuple(plane)] = True
            elif mode in (BOUNDARY_NONE, BOUNDARY_PML):
                inactive[tuple(plane)] = True
    result = (pec, ~(pec | inactive))
    if len(_MASK_CACHE) > 64:
        _MASK_CACHE.clear()
    _MASK_CACHE[key] = result
    return result


def _finish_electric(field, updated, specs):
    pec, active = _boundary_masks(field.shape, specs, field.device)
    field.copy_(torch.where(pec, torch.zeros_like(field), torch.where(active, updated, field)))


# Magnetic updates -----------------------------------------------------------

_MAGNETIC = {
    "x": ("Hx", "Ey", "Ez", 1, 2, "y", "z"),
    "y": ("Hy", "Ez", "Ex", 2, 0, "z", "x"),
    "z": ("Hz", "Ex", "Ey", 0, 1, "x", "y"),
}


def _magnetic_kernel(component, cpml):
    h_name, a_name, b_name, a_axis, b_axis, a_label, b_label = _MAGNETIC[component]
    # curl_component = d(B)/d(a_axis) - d(A)/d(b_axis) with (A, B) = (a_name, b_name)
    # following H_x: dEz/dy - dEy/dz, H_y: dEx/dz - dEz/dx, H_z: dEy/dx - dEx/dy.

    def kernel(**kw):
        h = kw[h_name]
        first = _forward(kw[b_name], a_axis, kw["invD" + a_label])
        second = _forward(kw[a_name], b_axis, kw["invD" + b_label])
        if cpml:
            prefix = h_name
            psi_a = kw[f"Psi{prefix}{a_label.upper()}"]
            psi_b = kw[f"Psi{prefix}{b_label.upper()}"]
            psi_a.copy_(_view(kw[f"By{prefix}{a_label.upper()}"], a_axis) * psi_a
                        + _view(kw[f"Cy{prefix}{a_label.upper()}"], a_axis) * first)
            psi_b.copy_(_view(kw[f"By{prefix}{b_label.upper()}"], b_axis) * psi_b
                        + _view(kw[f"Cy{prefix}{b_label.upper()}"], b_axis) * second)
            curl = (first * _view(kw[f"InvKappa{prefix}{a_label.upper()}"], a_axis) + psi_a
                    - second * _view(kw[f"InvKappa{prefix}{b_label.upper()}"], b_axis) - psi_b)
        else:
            curl = first - second
        h.copy_(h * kw[h_name + "Decay"] - kw[h_name + "Curl"] * curl)

    return kernel


# Electric updates -----------------------------------------------------------

_ELECTRIC = {
    "x": ("Ex", "Hz", "Hy", 1, 2, "y", "z"),
    "y": ("Ey", "Hx", "Hz", 2, 0, "z", "x"),
    "z": ("Ez", "Hy", "Hx", 0, 1, "x", "y"),
}


def _electric_kernel(component, cpml):
    e_name, first_name, second_name, a_axis, b_axis, a_label, b_label = _ELECTRIC[component]
    # E_x: dHz/dy - dHy/dz, E_y: dHx/dz - dHz/dx, E_z: dHy/dx - dHx/dy.

    def kernel(**kw):
        e = kw[e_name]
        low_a, high_a = kw[f"{a_label}LowBoundaryMode"], kw[f"{a_label}HighBoundaryMode"]
        low_b, high_b = kw[f"{b_label}LowBoundaryMode"], kw[f"{b_label}HighBoundaryMode"]
        first = _backward(kw[first_name], e.shape, a_axis, low_a, high_a, kw["invD" + a_label])
        second = _backward(kw[second_name], e.shape, b_axis, low_b, high_b, kw["invD" + b_label])
        specs = ((a_axis, int(low_a), int(high_a)), (b_axis, int(low_b), int(high_b)))
        if cpml:
            psi_a = kw[f"Psi{e_name}{a_label.upper()}"]
            psi_b = kw[f"Psi{e_name}{b_label.upper()}"]
            _, active = _boundary_masks(e.shape, specs, e.device)
            new_a = (_view(kw[f"B{e_name}{a_label.upper()}"], a_axis) * psi_a
                     + _view(kw[f"C{e_name}{a_label.upper()}"], a_axis) * first)
            new_b = (_view(kw[f"B{e_name}{b_label.upper()}"], b_axis) * psi_b
                     + _view(kw[f"C{e_name}{b_label.upper()}"], b_axis) * second)
            psi_a.copy_(torch.where(active, new_a, psi_a))
            psi_b.copy_(torch.where(active, new_b, psi_b))
            curl = (first * _view(kw[f"InvKappa{e_name}{a_label.upper()}"], a_axis) + new_a
                    - second * _view(kw[f"InvKappa{e_name}{b_label.upper()}"], b_axis) - new_b)
        else:
            curl = first - second
        _finish_electric(e, e * kw[e_name + "Decay"] + kw[e_name + "Curl"] * curl, specs)

    return kernel


# Sources --------------------------------------------------------------------

def _patch_view(field, patch, offsets):
    slices, local = [], []
    for axis in range(3):
        start = int(offsets[axis])
        lo, hi = max(start, 0), min(start + patch.shape[axis], field.shape[axis])
        if hi <= lo:
            return None, None
        slices.append(slice(lo, hi))
        local.append(slice(lo - start, hi - start))
    return tuple(slices), tuple(local)


def _scatter(field, patch, offsets, values):
    target, local = _patch_view(field, patch, offsets)
    if target is not None:
        field[target] += values[local].to(field.dtype)


def _add_source_patch(*, field, sourcePatch, offsetI, offsetJ, offsetK, signal):
    _scatter(field, sourcePatch, (offsetI, offsetJ, offsetK), sourcePatch * float(signal))


def _add_cw_phased_source_patch(*, field, sourcePatchCos, sourcePatchSin, offsetI, offsetJ, offsetK,
                                signalCos, signalSin):
    values = sourcePatchCos * float(signalCos) + sourcePatchSin * float(signalSin)
    _scatter(field, sourcePatchCos, (offsetI, offsetJ, offsetK), values)


def _add_time_shifted_source_patch(*, field, sourcePatch, delayPatch, activationDelayPatch, offsetI, offsetJ,
                                   offsetK, timeKind, time, frequency, fwidth, amplitude, phase, delay, causalGate,
                                   pedestal=0.0, pedestalFwidth=0.0):
    from ..waveforms import evaluate_time_shifted_waveform_torch
    source_time = dict(kind_code=int(timeKind), amplitude=amplitude, frequency=frequency, fwidth=fwidth,
                       phase=phase, delay=delay, pedestal=float(pedestal), pedestal_fwidth=float(pedestalFwidth))
    sample_time = float(time) - delayPatch.double()
    values = evaluate_time_shifted_waveform_torch(source_time, sample_time) * sourcePatch.double()
    if int(causalGate) != 0:
        values = torch.where(float(time) >= activationDelayPatch.double(), values, torch.zeros_like(values))
    _scatter(field, sourcePatch, (offsetI, offsetJ, offsetK), values)


# Observers ------------------------------------------------------------------

def _scalar(value):
    return value if torch.is_tensor(value) else float(value)


def _accumulate_point_observers(*, field, pointI, pointJ, pointK, realAccum, imagAccum, weightedCos, weightedSin):
    values = field[pointI.long(), pointJ.long(), pointK.long()].to(realAccum.dtype)
    realAccum.add_(values * _scalar(weightedCos))
    imagAccum.add_(values * _scalar(weightedSin))


def _accumulate_plane_observer(*, field, planeRealAccum, planeImagAccum, axisCode, planeIndex,
                               weightedCos, weightedSin):
    values = field.select(int(axisCode), int(planeIndex)).to(planeRealAccum.dtype)
    planeRealAccum.add_(values * _scalar(weightedCos))
    planeImagAccum.add_(values * _scalar(weightedSin))


def _accumulate_dft_batched(*, Ex, Ey, Ez, ExRealAccum, ExImagAccum, EyRealAccum, EyImagAccum, EzRealAccum,
                            EzImagAccum, weightedCos, weightedSin):
    shape = (weightedCos.shape[0], 1, 1, 1)
    cos, sin = weightedCos.reshape(shape), weightedSin.reshape(shape)
    for field, real, imag in ((Ex, ExRealAccum, ExImagAccum), (Ey, EyRealAccum, EyImagAccum),
                              (Ez, EzRealAccum, EzImagAccum)):
        if real.numel() == 0 or real.shape[1:] != field.shape:
            continue
        real.add_(field.unsqueeze(0).to(real.dtype) * cos)
        imag.add_(field.unsqueeze(0).to(imag.dtype) * sin)


_KERNELS = {
    **{f"updateMagneticFieldH{c}Standard3D": _magnetic_kernel(c, False) for c in "xyz"},
    **{f"updateMagneticFieldH{c}3D": _magnetic_kernel(c, True) for c in "xyz"},
    **{f"updateElectricFieldE{c}Standard3D": _electric_kernel(c, False) for c in "xyz"},
    **{f"updateElectricFieldE{c}Cpml3D": _electric_kernel(c, True) for c in "xyz"},
    "addSourcePatch3D": _add_source_patch,
    "addCwPhasedSourcePatch3D": _add_cw_phased_source_patch,
    "addTimeShiftedSourcePatch3D": _add_time_shifted_source_patch,
    "accumulatePointObservers3D": _accumulate_point_observers,
    "accumulatePlaneObserver3D": _accumulate_plane_observer,
    "accumulateRunningDftYee3DBatched": _accumulate_dft_batched,
}


class _Launch:
    __slots__ = ("kernel", "kwargs")

    def __init__(self, kernel, kwargs):
        self.kernel, self.kwargs = kernel, kwargs

    def launchRaw(self):  # noqa: N802 - mirrors the native launch entry point
        self.kernel(**self.kwargs)


class CpuReferenceFDTDModule:
    """Kernel-name compatible Torch module for the CPU reference profile."""

    kernel_names = tuple(sorted(_KERNELS))

    def __getattr__(self, name):
        kernel = _KERNELS.get(name)
        if kernel is None:
            raise NotImplementedError(
                f"FDTD kernel {name!r} is outside the CPU reference profile; run this feature on the native CUDA backend.")

        def bind(**kwargs):
            return _Launch(kernel, kwargs)

        object.__setattr__(self, name, bind)
        return bind


_MODULE = CpuReferenceFDTDModule()


def get_cpu_reference_module():
    return _MODULE
