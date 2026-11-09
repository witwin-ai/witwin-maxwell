"""Native plane-monitor payloads from externally computed Yee fields."""

from ..compiler.monitors import _compile_plane_observers
from ..fdtd.observers import (
    _assemble_plane_monitor_payload,
    _combine_plane_sample_data,
    _normalize_frequency_list,
    _prepare_plane_observer,
)
from ..fdtd.quadrature import _AXIS_CODES
from ..monitors import FinitePlaneMonitor, FluxMonitor, PlaneMonitor, normalize_component
from ..scene import prepare_scene


def plane_payload(scene, monitor, fields, *, frequencies, prepared_scene=None) -> dict:
    """Build the raw FDTD plane-monitor payload from complex Yee-native fields.

    Supports PlaneMonitor, FinitePlaneMonitor (including sampled and resolved
    closed-surface faces), and FluxMonitor with normalize_source=False only.
    Source normalization is not performed. The spectral ``samples`` count is
    omitted because external fields have no time-domain accumulation count.
    When prepared_scene is omitted, prepare_scene(scene) uses the same float32
    material default as Result.prepared_scene for external fields.

    Component keys use normalize_component, accepting Result.fields uppercase
    keys (EX, EY, EZ) as well as Ex/ex and the corresponding magnetic keys.
    Each tensor has its prepared component-grid shape, preceded by a frequency
    axis only for multiple frequencies. ``frequencies`` describes that axis; monitor-specific frequency
    selections use the same ordering as native observers.

    Sampled planes reduce already scaled volume fields, whereas FDTD observers
    scale after reducing the native taps. In the float32 validation scene this
    ordering differs by at most 7 ULPs (relative error 4.79701e-7); sampled
    comparisons use rtol=8*float32_eps (9.5367431640625e-7), atol=0. Unsampled
    planes, flux, and native closed-surface faces are bitwise identical there.
    Result.fields currently supplies electric components only; magnetic
    components must also be supplied for monitors that request them. The
    electric comparison uses the solver's full-field DFT; magnetic volumes in
    that validation come from native on-grid plane observers of the same run,
    so magnetic exactness covers the shared assembly, not an independent DFT.
    """
    if not isinstance(monitor, (PlaneMonitor, FinitePlaneMonitor, FluxMonitor)):
        raise TypeError("plane_payload() expects a PlaneMonitor, FinitePlaneMonitor, or FluxMonitor.")
    if prepared_scene is None:
        prepared_scene = prepare_scene(scene)
    native_fields = {normalize_component(key): value for key, value in fields.items()}
    frequencies = _normalize_frequency_list(frequencies)
    monitor_frequencies = _normalize_frequency_list(monitor.frequencies) or frequencies
    frequency_indices = [frequencies.index(frequency) for frequency in monitor_frequencies]
    component_samples = []
    for observer in _compile_plane_observers(monitor):
        observer["frequencies"] = monitor_frequencies
        field = native_fields[observer["component"]]
        _prepare_plane_observer(prepared_scene, observer, device=field.device, dtype=field.real.dtype)
        plane_sample_data = []
        for sample in observer["plane_samples"]:
            if "sample_weights" in sample:
                points = sample["sample_points"].long()
                values = field[..., points[:, 0], points[:, 1], points[:, 2]]
                weights = sample["sample_weights"]
                values = values.reshape(*field.shape[:-3], *weights.shape)
                values = (values * weights).sum(-1)
            else:
                values = field.select(field.ndim - 3 + _AXIS_CODES[observer["axis"]], sample["plane_index"])
            if len(frequencies) > 1:
                values = values[frequency_indices]
                if len(monitor_frequencies) == 1:
                    values = values[0]
            plane_sample_data.append((sample["weight"], values))
        component_samples.append((observer, _combine_plane_sample_data(plane_sample_data)))
    return _assemble_plane_monitor_payload(prepared_scene, component_samples)
