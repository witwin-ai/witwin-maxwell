"""R4-2 gates: streamed per-frequency mm-wave networks (frequency_groups.py).

A multi-frequency call whose full-set ungraded vertical table fails the z gate
streams the exact single-frequency prepare+solve per frequency instead of one
graded table for every frequency. Each frequency must be torch.equal to its
own single-frequency call; consecutive frequencies on one wall mesh form a
group. Ungraded multi-frequency calls keep the single-table path (bitwise A/B
against the base is in the R4-2 report; here: no streaming metadata).
"""

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.errors import UnsupportedFeatureError
from witwin.maxwell.mom.layered import LayeredGreenTable
from witwin.maxwell.mom.layered.frequency_groups import (FrequencyGroup, GroupedLayeredSurfaceData,
                                                         merged_metadata)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")
H, WIDTH = 254e-6, 254e-6


def _scene():
    stack = mw.mom.LayerStack((mw.mom.Layer("substrate", H, 9.8),), "pec", mw.mom.HalfSpace())
    scene = mw.Scene(background=stack, structures=(
                mw.Structure(mw.geometry.PlanarPEC(
            [(0., -WIDTH / 2), (2e-3, -WIDTH / 2), (2e-3, WIDTH / 2), (0., WIDTH / 2)], normal_axis="z", coordinate=H),
            mw.Material.pec()),
            ))
    scene = scene.updated(ports=(
        *scene.ports,
        *(mw.LumpedPort(name, positive=(x, 0., H), negative=(x, 0., 0.), voltage_path=mw.AxisPath("z")) for name, x in (("p1", 0.), ("p2", 2e-3))),
    ))
    return scene


def _config(**changes):
    layered = mw.mom.LayeredSurfaceConfig(bridge_width_m=WIDTH, max_edge_length_m=H / 2, edge_cell_m=WIDTH / 8,
                                          **changes)
    return mw.mom.MoMConfig(layered=layered)


def test_merged_metadata_concatenates_per_frequency_tuples():
    first = dict(solver="layered_mpie_dense", unknowns=10, factor_dtype=("complex64",), backward_error=(1e-12,),
                 vertical_z_grading=dict(levels=1), mesh=torch.arange(3))
    second = dict(solver="layered_mpie_dense", unknowns=12, factor_dtype=("complex128",), backward_error=(2e-12,),
                  vertical_z_grading=dict(levels=2), mesh=torch.arange(3))
    groups = (FrequencyGroup(0, 1, None, None), FrequencyGroup(1, 2, None, None))
    merged = merged_metadata([first, second], groups)
    assert merged["solver"] == "layered_mpie_dense" and torch.equal(merged["mesh"], torch.arange(3))
    assert "unknowns" not in merged and "vertical_z_grading" not in merged
    assert merged["factor_dtype"] == ("complex64", "complex128") and merged["backward_error"] == (1e-12, 2e-12)
    assert merged["frequency_groups"] == ((0, 1), (1, 2)) and merged["frequency_metadata"] == (first, second)


def test_grouped_surface_data_refuses_single_mesh_access():
    groups = (FrequencyGroup(0, 2, "mesh-a", torch.zeros(2, 3, 1)), FrequencyGroup(2, 3, "mesh-b", torch.ones(1, 5, 1)))
    data = GroupedLayeredSurfaceData(groups, torch.tensor([1., 2., 3.]), ("p1",))
    assert data.frequency_groups == ((0, 2), (2, 3))
    assert data.groups[1].layout == "mesh-b" and torch.equal(data.groups[1].frequencies, torch.tensor([3.]))
    with pytest.raises(UnsupportedFeatureError, match="one wall mesh per frequency group"):
        data.currents
    with pytest.raises(UnsupportedFeatureError, match="one wall mesh per frequency group"):
        data.layout


@requires_cuda
@pytest.mark.cuda
def test_green_table_defers_multi_frequency_grading():
    stack = mw.mom.LayerStack((mw.mom.Layer("substrate", H, 9.8),), "pec", mw.mom.HalfSpace())
    def build(frequencies, stream):
        return LayeredGreenTable.build(stack, (0.0, H), frequencies, 1.01 * math.hypot(4e-3, H), device="cuda",
                                       via_spans=((0.0, H),), z_nodes=12, extract_distance=max(math.sqrt(2) * H, H),
                                       stream_graded=stream)
    assert build((3e9, 30e9), True) is None
    # An admitted ungraded table and a single frequency are never deferred.
    assert build((1e9, 3e9), True).vertical.grading is None
    assert build((30e9,), True).vertical.grading is not None


@requires_cuda
@pytest.mark.cuda
def test_streamed_network_equals_single_frequency_calls(record_property):
    # 30 GHz grades to level 1, 60 and 100 GHz to level 2: two groups, one of two frequencies.
    frequencies = (30e9, 60e9, 100e9)
    joint = mw.Simulation.mom(_scene(), frequencies=frequencies, config=_config()).run()
    data = mw.mom.surface_currents(joint)
    metadata = joint.network.metadata
    assert metadata["frequency_streaming"] == "per_frequency_graded_tables"
    assert data.frequency_groups == metadata["frequency_groups"] == ((0, 1), (1, 3))
    assert [m["vertical_z_grading"]["levels"] for m in metadata["frequency_metadata"]] == [1, 2, 2]
    assert len(metadata["factor_dtype"]) == 3
    record_property("unknowns", [m["unknowns"] for m in metadata["frequency_metadata"]])
    for index, frequency in enumerate(frequencies):
        group = next(g for g, (a, b) in enumerate(data.frequency_groups) if a <= index < b)
        single = mw.Simulation.mom(_scene(), frequencies=(frequency,), config=_config()).run()
        currents = mw.mom.surface_currents(single)
        assert torch.equal(single.network.s[0], joint.network.s[index])
        assert torch.equal(single.network.z0[0], joint.network.z0[index])
        assert torch.equal(currents.currents[0], data.groups[group].currents[index - data.frequency_groups[group][0]])
        assert torch.equal(currents.layout.mesh.vertices, data.groups[group].layout.mesh.vertices)
        del single, currents
    again = mw.Simulation.mom(_scene(), frequencies=frequencies, config=_config()).run()
    assert torch.equal(again.network.s, joint.network.s)


@requires_cuda
@pytest.mark.cuda
def test_streamed_network_refuses_gradients():
    scene = _scene()
    sigma = torch.tensor(5.8e7, dtype=torch.float64, requires_grad=True)
    strip = mw.geometry.PlanarPEC([(0., -WIDTH / 2), (2e-3, -WIDTH / 2), (2e-3, WIDTH / 2), (0., WIDTH / 2)],
                                  normal_axis="z", coordinate=H, physical_thickness=2e-6)
    scene = scene.updated(structures=(mw.Structure(strip, mw.media.LossyMetalMedium(conductivity=sigma)), *scene.structures[1:]))
    # Explicit sheet: under the A2 default "auto" the 2 um copper strip (t/delta >= 0.3) selects
    # surface_admittance, whose gradient refusal (guard 5a2c81e7) would fire before the streamed guard.
    with pytest.raises(UnsupportedFeatureError, match="one prepared geometry for every frequency"):
        mw.Simulation.mom(scene, frequencies=(30e9, 60e9), config=_config(conductor_model="sheet")).prepare()


@requires_cuda
@pytest.mark.cuda
def test_streamed_adaptive_sweep(record_property):
    """R6-3: the sweep samples a streamed band; every sample is its own single-frequency call.

    30-45 GHz keeps the 2 mm line within the sweep's order-8 fit (30-60 GHz, 0.5-1.6 guided
    wavelengths, exhausted 25 samples before 1e-3).
    """
    config = _config(sweep=mw.mom.AdaptiveSweep(30e9, 45e9, 1e-3, 25))
    result = mw.Simulation.mom(_scene(), config=config).run()
    network, data = result.network, mw.mom.surface_currents(result)
    metadata = network.metadata
    frequencies = network.frequencies.tolist()
    record_property("solved_frequencies", frequencies)
    record_property("levels", [m["vertical_z_grading"]["levels"] for m in metadata["frequency_metadata"]])
    assert frequencies == sorted(frequencies) and len(frequencies) == data.sweep.solved_frequencies <= 25
    assert metadata["frequency_streaming"] == "per_frequency_graded_tables"
    assert data.frequency_groups == metadata["frequency_groups"]
    assert data.frequency_groups[0][0] == 0 and data.frequency_groups[-1][1] == len(frequencies)
    assert all(a[1] == b[0] for a, b in zip(data.frequency_groups, data.frequency_groups[1:]))
    assert len(metadata["frequency_metadata"]) == len(metadata["factor_dtype"]) == len(frequencies)
    assert data.sweep.passivity_report.passive
    # Two samples (the first seed and one bisection sample) equal their single-frequency calls.
    seeds = config.layered.sweep.seed_frequencies()
    added = [f for f in frequencies if f not in seeds]
    assert added
    for frequency in (frequencies[0], added[-1]):
        index = frequencies.index(frequency)
        group = next(g for g, (a, b) in enumerate(data.frequency_groups) if a <= index < b)
        single = mw.Simulation.mom(_scene(), frequencies=(frequency,), config=_config()).run()
        assert torch.equal(single.network.s[0], network.s[index])
        assert torch.equal(mw.mom.surface_currents(single).currents[0],
                           data.groups[group].currents[index - data.frequency_groups[group][0]])
        del single
