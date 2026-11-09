from types import SimpleNamespace

import numpy as np
import torch

from witwin.maxwell.fdtd.observers import _field_name, get_observer_results
from witwin.maxwell.fdtd.runtime.spectral import source_reference_metadata


def test_source_normalization_applies_once_per_monitor():
    solver = SimpleNamespace(
        observers_enabled=True,
        complex_fields_enabled=False,
        _normalize_source=True,
        _normalization_source_time={"kind": "gaussian_pulse"},
        # Pulses integrate with dt times each output frequency.
        # A power-of-two dt keeps the asserted values exact.
        dt=0.25,
        _spectral_integration_dt=0.25,
        _spectral_reference_frequency=None,
        _observer_spectral_entries=[
            {
                "sample_count": 1,
                "frequency": 1.0,
                "source_dft_real": 1.0,
                "source_dft_imag": 0.0,
                "window_normalization": 0.5,
            }
        ],
        observers=[
            {
                "name": "shared_ex",
                "monitor_name": "shared",
                "kind": "point",
                "component": "Ex",
                "monitor_fields": ("Ex", "Ey"),
                "frequencies": (1.0,),
                "global_freq_indices": (0,),
                "group_component": "Ex",
                "group_offset": 0,
                "field_index": (0, 0, 0),
                "position": (0.0, 0.0, 0.0),
            },
            {
                "name": "shared_ey",
                "monitor_name": "shared",
                "kind": "point",
                "component": "Ey",
                "monitor_fields": ("Ex", "Ey"),
                "frequencies": (1.0,),
                "global_freq_indices": (0,),
                "group_component": "Ey",
                "group_offset": 0,
                "field_index": (0, 0, 0),
                "position": (0.0, 0.0, 0.0),
            },
        ],
        _point_observer_groups={
            "Ex": {
                "field_name": _field_name("Ex"),
                "global_freq_indices": (0,),
                "freq_local_lookup": {0: 0},
                "real": torch.tensor([[3.0]], dtype=torch.float32),
                "imag": torch.tensor([[0.0]], dtype=torch.float32),
            },
            "Ey": {
                "field_name": _field_name("Ey"),
                "global_freq_indices": (0,),
                "freq_local_lookup": {0: 0},
                "real": torch.tensor([[5.0]], dtype=torch.float32),
                "imag": torch.tensor([[0.0]], dtype=torch.float32),
            },
        },
        _plane_observer_groups={},
    )

    # Fill the entry's source-reference fields through the production builder so new
    # entry fields cannot silently go missing here. A label offset of one makes the
    # reference conversion the identity, keeping this test about normalization alone.
    solver._normalization_reference = dict(solver._normalization_source_time)
    solver._normalization_label_offset = 1.0
    for entry in solver._observer_spectral_entries:
        entry.update(source_reference_metadata(solver, entry["frequency"]))

    result = get_observer_results(solver)["shared"]

    # spectral_scale = integration_dt * frequency = 0.25 for both the raw point
    # values and the source spectrum, so single-pass normalization recovers 3 and 5.
    assert result["Ex"] == np.complex128(3.0)
    assert result["Ey"] == np.complex128(5.0)
    assert result["components"]["Ex"] == np.complex128(3.0)
    assert result["components"]["Ey"] == np.complex128(5.0)
