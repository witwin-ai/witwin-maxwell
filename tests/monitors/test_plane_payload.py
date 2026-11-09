"""External Yee fields share native plane geometry and payload assembly."""

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.coords import component_coords
from witwin.maxwell.scene import prepare_scene


FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _compare(actual, expected, *, rtol=0):
    if isinstance(expected, torch.Tensor):
        if rtol == 0 or torch.equal(actual, expected):
            assert torch.equal(actual, expected), "Unsampled plane differs from the native DFT"
        else:
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _compare(actual[key], expected[key], rtol=rtol)
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _compare(a, b, rtol=rtol)
    elif isinstance(expected, np.ndarray):
        assert np.array_equal(actual, expected)
    else:
        assert actual == expected


def _differences(actual, expected):
    if isinstance(expected, torch.Tensor):
        a, b = actual.detach().cpu(), expected.detach().cpu()
        delta = float((a - b).abs().max())
        # Count representable float steps separately in real and imaginary lanes.
        if a.is_complex():
            a, b = torch.view_as_real(a), torch.view_as_real(b)
        integer = torch.int32 if a.dtype == torch.float32 else torch.int64
        ai, bi = a.contiguous().view(integer), b.contiguous().view(integer)
        ai = torch.where(ai < 0, torch.iinfo(integer).min - ai, ai).to(torch.int64)
        bi = torch.where(bi < 0, torch.iinfo(integer).min - bi, bi).to(torch.int64)
        ulp = int((ai - bi).abs().max())
        relative = float(((actual - expected).abs() / expected.abs()).nan_to_num().max())
        return delta, ulp, relative
    if isinstance(expected, dict):
        children = [_differences(actual[k], v) for k, v in expected.items()]
    elif isinstance(expected, (tuple, list)):
        children = [_differences(a, b) for a, b in zip(actual, expected)]
    else:
        return 0., 0, 0.
    return tuple(max(c[i] for c in children) for i in range(3)) if children else (0., 0, 0.)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_plane_payload_matches_native_observers(record_property, capsys):
    frequencies = (2e9, 2.45e9)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.04, 0.04),) * 3),
        grid=mw.GridSpec.uniform(.004),
        boundary=mw.BoundarySpec.pml(num_layers=4), device="cuda",
        sources=[mw.PointDipole(name="drive", position=(.001, -.002, .003),
                               polarization=(1., 2., 3.),
                               source_time=mw.GaussianPulse(frequency=2.45e9, fwidth=2.45e9))],
        monitors=[
            mw.PlaneMonitor("plane", axis="z", position=.009, fields=FIELDS),
            mw.FinitePlaneMonitor("finite", position=(.001, .002, .011), size=(.029, .031, 0.), fields=FIELDS),
            mw.FinitePlaneMonitor("sampled", position=(.001, .002, .011), size=(.029, .031, 0.),
                                  fields=FIELDS, sample_spacing=.006),
            mw.FluxMonitor("flux", axis="y", position=.009),
            mw.ClosedSurfaceMonitor.box("surface", size=(.037, .035, .033)),
            mw.PlaneMonitor("single", axis="x", position=.009, fields=("Ez",)),
        ],
    )
    target_monitors = scene.resolved_monitors()
    # Full-field DFT exposes E only. Native on-grid H planes supply a test-only
    # magnetic volume reference without changing the production DFT path.
    prepared = prepare_scene(scene)
    magnetic_planes = {}
    for component in FIELDS[3:]:
        magnetic_planes[component] = []
        for index, position in enumerate(component_coords(prepared, component)[2]):
            name = f"reference_{component}_{index}"
            scene.add_monitor(mw.PlaneMonitor(name, axis="z", position=float(position), fields=(component,)))
            magnetic_planes[component].append(name)
    result = mw.Simulation.fdtd(
        scene, frequencies=frequencies, full_field_dft=True,
        run_time=mw.TimeConfig(time_steps=128, transient_cycles=0, steady_cycles=1),
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=False),
    ).run()
    fields = dict(result.fields)
    assert set(fields) == {"EX", "EY", "EZ"}
    for component, names in magnetic_planes.items():
        fields[component.upper()] = torch.stack([result._monitors[name]["data"] for name in names], dim=-1)
    payloads = {}
    # Compare the stored raw observer payload before Result's finite aperture crop.
    for monitor in target_monitors:
        actual = mw.postprocess.plane_payload(scene, monitor, fields,
                                             frequencies=result.frequencies, prepared_scene=result.prepared_scene)
        expected = dict(result._monitors[monitor.name])
        expected.pop("samples")
        difference = _differences(actual, expected)
        record_property(monitor.name, str(difference))
        with capsys.disabled():
            print(f"{monitor.name}: max_abs={difference[0]:.12g}, max_ulp={difference[1]}, max_relative={difference[2]:.12g}")
        # Measured max relative error 4.79701e-7 (7 ULPs): round up to 8 eps.
        _compare(actual, expected, rtol=0 if getattr(monitor, "sample_spacing", None) is None
                 else 8 * torch.finfo(torch.float32).eps)
        payloads[monitor.name] = actual
    external = mw.Result.from_fields(scene, frequencies=result.frequencies, kernel="t", fields=fields,
                                     monitors=payloads, prepared_scene=result.prepared_scene)
    for name in ("plane", "flux"):
        expected = dict(result.monitor(name))
        expected.pop("samples")
        _compare(external.monitor(name), expected)
    single = mw.postprocess.plane_payload(scene, scene.monitors[5], result.fields, frequencies=result.frequencies)
    _compare(single, payloads["single"])
    selected_monitor = mw.PlaneMonitor("selected", axis="x", position=.009, fields=("Ez",),
                                       frequencies=(frequencies[1],))
    selected = mw.postprocess.plane_payload(scene, selected_monitor, result.fields, frequencies=frequencies)
    assert torch.equal(selected["data"], single["data"][1])
    scalar = mw.postprocess.plane_payload(scene, scene.monitors[5],
                                         {key.lower(): value[1] for key, value in result.fields.items()},
                                         frequencies=(frequencies[1],))
    _compare(scalar, selected)
    rebuilt = mw.postprocess.plane_payload(scene, scene.monitors[0], fields, frequencies=result.frequencies)
    _compare(rebuilt, payloads["plane"])


def test_plane_payload_rejects_other_monitor_types():
    with pytest.raises(TypeError, match="plane_payload"):
        mw.postprocess.plane_payload(None, mw.PointMonitor("point", position=(0., 0., 0.)), {}, frequencies=(1e9,))
