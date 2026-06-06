import math
import os

import pytest
import torch

from witwin.maxwell.fdtd.cuda import backend


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native CUDA source tests."),
    pytest.mark.skipif(
        os.environ.get("WITWIN_RUN_CUDA_EXTENSION_BUILD") != "1",
        reason="Set WITWIN_RUN_CUDA_EXTENSION_BUILD=1 to compile and run native CUDA source kernels.",
    ),
]


class _CountingExtension:
    def __init__(self, extension, method_names):
        self._extension = extension
        self.calls = {name: 0 for name in method_names}

    def __getattr__(self, name):
        target = getattr(self._extension, name)
        if name not in self.calls:
            return target

        def counted(*args, **kwargs):
            self.calls[name] += 1
            return target(*args, **kwargs)

        return counted


def _rand(shape, generator):
    return torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator).contiguous()


def _assert_methods(extension, method_names):
    for name in method_names:
        assert hasattr(extension, name), name


def _phase_positive(value, phase_cos, phase_sin):
    return complex(phase_cos * value.real - phase_sin * value.imag, phase_sin * value.real + phase_cos * value.imag)


def _phase_negative(value, phase_cos, phase_sin):
    return complex(phase_cos * value.real + phase_sin * value.imag, phase_cos * value.imag - phase_sin * value.real)


def _periodic_expected(shape, patch, offsets, signal, axes):
    expected = torch.zeros(shape, dtype=torch.float32)
    patch_cpu = patch.detach().cpu()
    for i in range(patch_cpu.shape[0]):
        for j in range(patch_cpu.shape[1]):
            for k in range(patch_cpu.shape[2]):
                coords = [offsets[0] + i, offsets[1] + j, offsets[2] + k]
                if any(coords[axis] < 0 or coords[axis] >= shape[axis] for axis in range(3)):
                    continue
                delta = float(signal) * float(patch_cpu[i, j, k])
                expected[coords[0], coords[1], coords[2]] += delta
                boundary = []
                pair = []
                for axis in axes:
                    at_boundary = coords[axis] == 0 or coords[axis] + 1 >= shape[axis]
                    boundary.append(at_boundary)
                    pair.append(shape[axis] - 1 if coords[axis] == 0 else 0)
                if boundary[0]:
                    dst = list(coords)
                    dst[axes[0]] = pair[0]
                    expected[dst[0], dst[1], dst[2]] += delta
                if boundary[1]:
                    dst = list(coords)
                    dst[axes[1]] = pair[1]
                    expected[dst[0], dst[1], dst[2]] += delta
                if boundary[0] and boundary[1]:
                    dst = list(coords)
                    dst[axes[0]] = pair[0]
                    dst[axes[1]] = pair[1]
                    expected[dst[0], dst[1], dst[2]] += delta
    return expected.to(device="cuda")


def _bloch_expected(shape, patch, offsets, signal, axis_code, phase_a, phase_b):
    axes = ((1, 2), (0, 2), (0, 1))[axis_code]
    expected = torch.zeros(shape, dtype=torch.complex64)
    patch_cpu = patch.detach().cpu()
    for i in range(patch_cpu.shape[0]):
        for j in range(patch_cpu.shape[1]):
            for k in range(patch_cpu.shape[2]):
                coords = [offsets[0] + i, offsets[1] + j, offsets[2] + k]
                if any(coords[axis] < 0 or coords[axis] >= shape[axis] for axis in range(3)):
                    continue
                delta = complex(signal[0] * float(patch_cpu[i, j, k]), signal[1] * float(patch_cpu[i, j, k]))
                expected[coords[0], coords[1], coords[2]] += delta
                boundary = []
                pair = []
                for axis in axes:
                    at_boundary = coords[axis] == 0 or coords[axis] + 1 >= shape[axis]
                    boundary.append(at_boundary)
                    pair.append(shape[axis] - 1 if coords[axis] == 0 else 0)
                if boundary[0]:
                    dst = list(coords)
                    dst[axes[0]] = pair[0]
                    value = (
                        _phase_positive(delta, *phase_a)
                        if coords[axes[0]] == 0
                        else _phase_negative(delta, *phase_a)
                    )
                    expected[dst[0], dst[1], dst[2]] += value
                if boundary[1]:
                    dst = list(coords)
                    dst[axes[1]] = pair[1]
                    value = (
                        _phase_positive(delta, *phase_b)
                        if coords[axes[1]] == 0
                        else _phase_negative(delta, *phase_b)
                    )
                    expected[dst[0], dst[1], dst[2]] += value
                if boundary[0] and boundary[1]:
                    dst = list(coords)
                    dst[axes[0]] = pair[0]
                    dst[axes[1]] = pair[1]
                    value = (
                        _phase_positive(delta, *phase_a)
                        if coords[axes[0]] == 0
                        else _phase_negative(delta, *phase_a)
                    )
                    value = (
                        _phase_positive(value, *phase_b)
                        if coords[axes[1]] == 0
                        else _phase_negative(value, *phase_b)
                    )
                    expected[dst[0], dst[1], dst[2]] += value
    return expected.to(device="cuda")


def test_compiled_cuda_extension_uniform_cw_and_time_shifted_sources_match_torch_dispatcher(monkeypatch):
    method_names = ("add_source_patch", "add_cw_phased_source_patch", "add_time_shifted_source_patch")
    real_extension = backend.get_compiled_extension()
    _assert_methods(real_extension, method_names)
    counted = _CountingExtension(real_extension, method_names)
    monkeypatch.setattr(backend, "_COMPILED_EXTENSION", counted)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(601)
    patch = _rand((2, 3, 2), generator)
    patch_sin = _rand((2, 3, 2), generator)
    delay_patch = torch.rand((2, 3, 2), device="cuda", dtype=torch.float32, generator=generator) * 0.2
    activation_delay = torch.full((2, 3, 2), 0.12, device="cuda", dtype=torch.float32)
    activation_delay[0, 0, 0] = 0.4

    expected = torch.zeros((5, 6, 5), device="cuda", dtype=torch.float32)
    actual = torch.zeros_like(expected)

    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "0")
    backend._add_source_patch(field=expected, sourcePatch=patch, offsetI=1, offsetJ=2, offsetK=1, signal=0.7)
    backend._add_cw_phased_source_patch(
        field=expected,
        sourcePatchCos=patch,
        sourcePatchSin=patch_sin,
        offsetI=0,
        offsetJ=1,
        offsetK=2,
        signalCos=0.25,
        signalSin=-0.5,
    )
    backend._add_time_shifted_source_patch(
        field=expected,
        sourcePatch=patch,
        delayPatch=delay_patch,
        activationDelayPatch=activation_delay,
        offsetI=2,
        offsetJ=0,
        offsetK=1,
        timeKind=1,
        time=0.25,
        frequency=1.2,
        fwidth=0.8,
        amplitude=2.0,
        phase=0.3,
        delay=0.1,
        causalGate=1,
    )

    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "1")
    backend._add_source_patch(field=actual, sourcePatch=patch, offsetI=1, offsetJ=2, offsetK=1, signal=0.7)
    backend._add_cw_phased_source_patch(
        field=actual,
        sourcePatchCos=patch,
        sourcePatchSin=patch_sin,
        offsetI=0,
        offsetJ=1,
        offsetK=2,
        signalCos=0.25,
        signalSin=-0.5,
    )
    backend._add_time_shifted_source_patch(
        field=actual,
        sourcePatch=patch,
        delayPatch=delay_patch,
        activationDelayPatch=activation_delay,
        offsetI=2,
        offsetJ=0,
        offsetK=1,
        timeKind=1,
        time=0.25,
        frequency=1.2,
        fwidth=0.8,
        amplitude=2.0,
        phase=0.3,
        delay=0.1,
        causalGate=1,
    )
    torch.cuda.synchronize()

    assert counted.calls == {name: 1 for name in method_names}
    torch.testing.assert_close(actual, expected, rtol=2.0e-6, atol=2.0e-7)


@pytest.mark.parametrize(
    ("field_name", "method_name", "shape", "patch_shape", "offsets", "axes"),
    [
        ("Ex", "add_source_patch_ex_periodic", (3, 2, 2), (1, 2, 2), (1, 0, 0), (1, 2)),
        ("Ey", "add_source_patch_ey_periodic", (2, 3, 2), (2, 1, 2), (0, 1, 0), (0, 2)),
        ("Ez", "add_source_patch_ez_periodic", (2, 2, 3), (2, 2, 1), (0, 0, 1), (0, 1)),
    ],
)
def test_compiled_cuda_extension_periodic_sources_apply_face_and_corner_duplicates(
    monkeypatch,
    field_name,
    method_name,
    shape,
    patch_shape,
    offsets,
    axes,
):
    real_extension = backend.get_compiled_extension()
    _assert_methods(real_extension, (method_name,))
    counted = _CountingExtension(real_extension, (method_name,))
    monkeypatch.setattr(backend, "_COMPILED_EXTENSION", counted)
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "1")

    patch = torch.arange(1, math.prod(patch_shape) + 1, device="cuda", dtype=torch.float32).reshape(patch_shape)
    actual = torch.zeros(shape, device="cuda", dtype=torch.float32)
    backend._add_source_patch_periodic(
        **{field_name: actual},
        sourcePatch=patch,
        offsetI=offsets[0],
        offsetJ=offsets[1],
        offsetK=offsets[2],
        signal=0.5,
        wrapAxisA=1,
        wrapAxisB=1,
    )
    torch.cuda.synchronize()

    assert counted.calls[method_name] == 1
    torch.testing.assert_close(actual, _periodic_expected(shape, patch, offsets, 0.5, axes), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("axis_code", [0, 1, 2])
def test_compiled_cuda_extension_bloch_sources_apply_phase_to_faces_and_corners(monkeypatch, axis_code):
    method_name = "add_source_patch_bloch"
    real_extension = backend.get_compiled_extension()
    _assert_methods(real_extension, (method_name,))
    counted = _CountingExtension(real_extension, (method_name,))
    monkeypatch.setattr(backend, "_COMPILED_EXTENSION", counted)
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "1")

    shape = (2, 2, 2)
    patch = torch.arange(1, 9, device="cuda", dtype=torch.float32).reshape(shape)
    fields = [torch.zeros(shape, device="cuda", dtype=torch.float32) for _ in range(6)]
    phase_a = (0.8, 0.6)
    phase_b = (0.5, -0.25)
    signal = (0.7, -0.2)

    backend._add_source_patch_bloch(
        ExReal=fields[0],
        ExImag=fields[1],
        EyReal=fields[2],
        EyImag=fields[3],
        EzReal=fields[4],
        EzImag=fields[5],
        sourcePatch=patch,
        offsetI=0,
        offsetJ=0,
        offsetK=0,
        signalReal=signal[0],
        signalImag=signal[1],
        axisCode=axis_code,
        phaseCosA=phase_a[0],
        phaseSinA=phase_a[1],
        phaseCosB=phase_b[0],
        phaseSinB=phase_b[1],
    )
    torch.cuda.synchronize()

    assert counted.calls[method_name] == 1
    expected = _bloch_expected(shape, patch, (0, 0, 0), signal, axis_code, phase_a, phase_b)
    actual_complex = torch.complex(fields[2 * axis_code], fields[2 * axis_code + 1])
    torch.testing.assert_close(actual_complex, expected, rtol=2.0e-6, atol=2.0e-6)
    for index, field in enumerate(fields):
        if index not in (2 * axis_code, 2 * axis_code + 1):
            assert torch.count_nonzero(field).item() == 0


def test_compiled_cuda_extension_tfsf_and_auxiliary_source_kernels_match_torch_dispatcher(monkeypatch):
    method_names = (
        "add_scaled_slice_source_patch",
        "add_scaled_line_source_patch",
        "add_interpolated_source_patch",
        "add_batched_reference_source_patches",
        "add_batched_interpolated_source_patches",
        "update_auxiliary_magnetic",
        "update_auxiliary_electric",
    )
    real_extension = backend.get_compiled_extension()
    _assert_methods(real_extension, method_names)
    counted = _CountingExtension(real_extension, method_names)
    monkeypatch.setattr(backend, "_COMPILED_EXTENSION", counted)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(603)
    incident = torch.linspace(-0.5, 1.25, 6, device="cuda", dtype=torch.float32)

    expected = [torch.zeros((4, 4, 4), device="cuda", dtype=torch.float32) for _ in range(3)]
    actual = [tensor.clone() for tensor in expected]
    patch = _rand((2, 2, 1), generator)
    line_patch = _rand((2, 3, 1), generator)
    interp_patch = _rand((1, 2, 2), generator)
    sample_indices = torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32)
    sample_positions = torch.tensor([[[0.1, 0.7], [1.3, 2.2]]], device="cuda", dtype=torch.float32)

    coeff_data = torch.tensor([0.5, -0.25, 0.75, 1.25, -0.6, 0.4], device="cuda", dtype=torch.float32)
    term_starts = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
    term_shapes = torch.tensor([[1, 2, 2], [2, 1, 1]], device="cuda", dtype=torch.int32)
    term_offsets = torch.tensor([[0, 1, 0], [2, 0, 1]], device="cuda", dtype=torch.int32)
    field_codes = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    sample_axis_codes = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
    sample_index_starts = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    batched_indices = torch.tensor([0, 3, 1, 4], device="cuda", dtype=torch.int32)
    batched_positions = torch.tensor([0.0, 0.4, 1.1, 2.0, 0.3, 1.7], device="cuda", dtype=torch.float32)

    def launch_all(fields):
        backend._add_scaled_slice_source_patch(
            field=fields[0],
            sourcePatch=patch,
            incidentField=incident,
            sampleIndex=2,
            offsetI=0,
            offsetJ=1,
            offsetK=0,
            scale=-0.75,
        )
        backend._add_scaled_line_source_patch(
            field=fields[1],
            coeffPatch=line_patch,
            incidentField=incident,
            sampleIndices=sample_indices,
            sampleAxisCode=1,
            offsetI=1,
            offsetJ=0,
            offsetK=2,
            scale=0.5,
        )
        backend._add_interpolated_source_patch(
            field=fields[2],
            coeffPatch=interp_patch,
            incidentField=incident,
            samplePositions=sample_positions,
            origin=0.0,
            ds=0.5,
            offsetI=2,
            offsetJ=1,
            offsetK=1,
            scale=1.25,
        )
        backend._add_batched_reference_source_patches(
            fieldX=fields[0],
            fieldY=fields[1],
            fieldZ=fields[2],
            coeffData=coeff_data,
            incidentField=incident,
            termStarts=term_starts,
            termShapes=term_shapes,
            termOffsets=term_offsets,
            fieldCodes=field_codes,
            sampleAxisCodes=sample_axis_codes,
            sampleIndexStarts=sample_index_starts,
            sampleIndices=batched_indices,
        )
        backend._add_batched_interpolated_source_patches(
            fieldX=fields[0],
            fieldY=fields[1],
            fieldZ=fields[2],
            coeffData=coeff_data,
            incidentField=incident,
            samplePositions=batched_positions,
            termStarts=term_starts,
            termShapes=term_shapes,
            termOffsets=term_offsets,
            fieldCodes=field_codes,
            origin=0.0,
            ds=0.5,
        )

    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "0")
    launch_all(expected)
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "1")
    launch_all(actual)

    electric = torch.linspace(-0.25, 0.75, 6, device="cuda", dtype=torch.float32)
    magnetic = torch.linspace(0.1, 0.9, 5, device="cuda", dtype=torch.float32)
    magnetic_decay = torch.full((5,), 0.8, device="cuda", dtype=torch.float32)
    magnetic_curl = torch.linspace(0.2, 0.6, 5, device="cuda", dtype=torch.float32)
    electric_decay = torch.full((6,), 0.9, device="cuda", dtype=torch.float32)
    electric_curl = torch.linspace(0.15, 0.65, 6, device="cuda", dtype=torch.float32)
    expected_magnetic = magnetic.clone()
    actual_magnetic = magnetic.clone()
    expected_electric = electric.clone()
    actual_electric = electric.clone()

    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "0")
    backend._update_auxiliary_magnetic(
        Magnetic=expected_magnetic,
        Electric=electric,
        MagneticDecay=magnetic_decay,
        MagneticCurl=magnetic_curl,
    )
    backend._update_auxiliary_electric(
        Electric=expected_electric,
        Magnetic=magnetic,
        ElectricDecay=electric_decay,
        ElectricCurl=electric_curl,
        sourceIndex=3,
        sourceValue=-1.25,
    )
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "1")
    backend._update_auxiliary_magnetic(
        Magnetic=actual_magnetic,
        Electric=electric,
        MagneticDecay=magnetic_decay,
        MagneticCurl=magnetic_curl,
    )
    backend._update_auxiliary_electric(
        Electric=actual_electric,
        Magnetic=magnetic,
        ElectricDecay=electric_decay,
        ElectricCurl=electric_curl,
        sourceIndex=3,
        sourceValue=-1.25,
    )
    torch.cuda.synchronize()

    assert counted.calls == {name: 1 for name in method_names}
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=2.0e-6, atol=2.0e-6)
    torch.testing.assert_close(actual_magnetic, expected_magnetic, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_electric, expected_electric, rtol=2.0e-6, atol=2.0e-7)
