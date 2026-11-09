"""CPU-only source/monitor admission, fallback and injection-clock contracts."""
from types import SimpleNamespace

import pytest
import torch

from benchmark.core_path.scenes import CORE_SCENES, SRCMON_REFUSED_SCENES, SRCMON_SCENES, build_core_case
from benchmark.scenes import SCENARIOS
from witwin.maxwell.fdtd.runtime.core_path import CorePathRuntime, core_path_blockers


pytestmark = pytest.mark.host_validation


@pytest.fixture(autouse=True)
def forbid_cuda_init(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Source/monitor host checks must not initialize CUDA.")
    monkeypatch.setattr(torch.cuda, "_lazy_init", fail)


@pytest.mark.parametrize("name", SRCMON_SCENES)
def test_benchmark_admission(name):
    case = build_core_case(name)
    # The refused PMC cavity and the closed-transverse CPML gratings stay out of the default gate list.
    assert (name in CORE_SCENES) == (name not in SRCMON_REFUSED_SCENES)
    assert case.frequencies == SCENARIOS[name].frequencies
    assert str(case.scene.device) == "cpu"
    expected = ["PMC boundaries", "non-CUDA execution"] if name == "pmc_cavity" else ["non-CUDA execution"]
    assert core_path_blockers(SimpleNamespace(scene=case.scene)) == expected


@pytest.mark.parametrize("name", ["grating_diffraction", "diffraction_normal_orders"])
def test_closed_transverse_cpml_gratings_are_refused(name):
    case = build_core_case(name)
    assert case.absorber == "cpml"
    # The blocker reads the resolved absorber, which exists only after solver preparation.
    solver = SimpleNamespace(scene=case.scene, active_absorber_type=case.absorber)
    assert "closed-transverse CPML (passivity guard)" in core_path_blockers(solver)


@pytest.mark.parametrize("attribute,label", [("sources", "source"), ("monitors", "monitor")])
def test_unknown_types_stay_refused(attribute, label):
    scene = build_core_case("uniform_current").scene
    setattr(scene, attribute, [object()])
    assert f"unsupported {label} type" in core_path_blockers(SimpleNamespace(scene=scene))


@pytest.mark.parametrize("observer", ["time", "host_spectral"])
def test_observer_tail_prevents_chunks(observer):
    runtime = CorePathRuntime.__new__(CorePathRuntime)
    runtime.sources = SimpleNamespace(steps=64)
    solver = SimpleNamespace(
        scene=SimpleNamespace(), time_observers_enabled=observer == "time", observers_enabled=True,
        _single_observer_engine=object() if observer == "time" else None,
        dft_enabled=True, Ex=torch.empty(0), _plane_observer_groups={"plane": object()},
    )
    result = runtime.run_chunks(
        solver, time_steps=64, start_step=0, end_step=64,
        use_cuda_graph=True, use_gpu_dft=True, cooperative=False,
        cooperative_driver=False, cooperative_observer=None, resume_from=None,
        shutoff=0.0, shutoff_check_interval=100, shutoff_min_step=0,
        enable_plot=False, pbar=None,
    )
    assert result is None
    assert solver._core_step_graph_active is False
    assert solver._core_step_graph_chunks == []


def test_surface_current_rows_preserve_injection_clocks():
    import witwin.maxwell as mw
    from witwin.maxwell.fdtd.excitation.temporal import build_source_term
    from witwin.maxwell.fdtd.runtime.core_tables import compile_source_tables
    from witwin.maxwell.sources import compile_source_time, evaluate_source_time

    source = compile_source_time(mw.CW(frequency=2.0, phase=0.31), default_frequency=2.0)
    fields = {name: torch.zeros(1, 1, 1) for name in ("Ex", "Ey", "Hz")}
    terms = {name: build_source_term(None, field_name=name, offsets=(0, 0, 0),
                                   patch=torch.ones(1, 1, 1), source_time=source)
             for name in fields}
    solver = SimpleNamespace(
        scene=build_core_case("uniform_current").scene, device="cpu", dt=0.07,
        _source_time=source, source_omega=4.0 * torch.pi, tfsf_enabled=False,
        _source_terms=[terms["Ex"]], _electric_source_terms=[terms["Ey"]],
        _magnetic_source_terms=[terms["Hz"]], **fields,
    )
    extension = SimpleNamespace(CoreStepLaunch=lambda opcode, tensors, integers, values:
                                SimpleNamespace(tensors=tensors))
    tables = compile_source_tables(solver, extension, torch.zeros(1, dtype=torch.int64), 4)
    for group, field, half_step in (("ordinary", "Ex", True), ("electric", "Ey", True),
                                    ("magnetic", "Hz", False)):
        times = [n * solver.dt + (0.5 * solver.dt if half_step else 0.0) for n in range(4)]
        expected = torch.tensor([evaluate_source_time(source, t) for t in times], dtype=torch.float32)
        assert torch.equal(tables.rows[f"{group}/0/signals"], expected)
        assert tables.groups[group][0].tensors[1] is fields[field]
    assert not torch.equal(tables.rows["magnetic/0/signals"], tables.rows["electric/0/signals"])
