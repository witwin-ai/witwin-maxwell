"""Shared sparse bulk-loss accounting at a two-trace conductive interface."""
import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.postprocess.fdtd_loss import native_pole_work_conductivity
from witwin.maxwell.fdtd.runtime.representation import spectral_phasor_factor, spectral_scale


@pytest.mark.parametrize("kind", ("debye", "drude", "lorentz"))
def test_native_spectral_material_work_matches_independent_time_recurrence(kind):
    dt, decay, drive, restoring = .01, .7, .2, .3
    theta = 2*math.pi/128
    polarization, current = 0., 0.
    work = []
    for n in range(128*128):
        electric = math.cos(theta*n)
        next_electric = math.cos(theta*(n+1))
        if kind == "debye":
            updated = decay*polarization+drive*electric
            current = (updated-polarization)/dt
            polarization = updated
        else:
            current = decay*current+drive*electric-(restoring*polarization if kind == "lorentz" else 0.)
            if kind == "lorentz":
                polarization += dt*current
        if n >= 128*64:
            work.append(.5*(electric+next_electric)*current)
    gain = native_pole_work_conductivity(kind, decay, torch.tensor([drive], dtype=torch.float64), dt,
        torch.tensor([theta/(2*math.pi*dt)], dtype=torch.float64), restoring)[0, 0]
    assert float(gain)/2 == pytest.approx(float(np.mean(work)), rel=1e-11, abs=1e-13)
    assert gain >= 0


def _simulation(kind="debye", *, graph=False, steps=256, pulse=False, normalize_source=True):
    from tests.materials.surface_impedance.test_copper_foil_material_ade import _local_scene
    scene = _local_scene(kind, one_side=True)
    scene.monitors = (*scene.monitors, mw.PowerLossMonitor("bulk", position=(0., 0., 0.), size=(.04, .04, .04),
        channels=("conduction", "electric_dispersion")),
        mw.PowerLossMonitor("subset", position=(0., 0., 0.), size=(.04, .04, .04),
            frequencies=(3.e9,), channels=("conduction", "electric_dispersion")))
    return mw.Simulation.fdtd(scene, frequencies=(2.5e9, 3.e9), cuda_graph=graph,
        run_time=mw.TimeConfig(time_steps=steps), full_field_dft=True,
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=normalize_source),
        excitations=mw.PortExcitation("feed", source_time=mw.GaussianPulse(3.e9, fwidth=1.e9) if pulse else mw.CW(3.e9)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind", ("debye", "drude", "lorentz"))
def test_shared_loss_preserves_sparse_integrals_and_graph_resume(kind, tmp_path):
    simulation = _simulation(kind, graph=True)
    result = simulation.run()
    data = result.power_loss("bulk")
    for channel in ("conduction", "electric_dispersion"):
        independent = sum((density*data.component_volumes[component][None]).sum(-1)
                          for component, density in data.volume_density[channel].items())
        torch.testing.assert_close(independent, data.channel_power[channel])
        assert bool(torch.all(data.channel_power[channel] > 0))
    checkpoint = _simulation(kind, graph=True).prepare().run_until(91)
    path = tmp_path/"state.pt"
    checkpoint.save(path)
    resumed = _simulation(kind, graph=True).prepare().run(resume_from=mw.FDTDResumeCheckpoint.load(path))
    torch.testing.assert_close(resumed.power_loss("bulk").total, data.total, rtol=2e-5, atol=1e-20)
    saved = tmp_path/"result.pt"
    result.save(saved)
    loaded = mw.Result.load(saved, scene=result.scene, map_location="cuda:0")
    torch.testing.assert_close(loaded.power_loss("bulk").total, data.total, rtol=2e-5, atol=1e-20)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_two_trace_bulk_loss_uses_known_side_amplitudes_and_physical_half_volumes():
    from witwin.maxwell.compiler.power_loss import compile_power_loss_monitor
    from witwin.maxwell.fdtd.copper_foil import copper_loss_report
    result = _simulation(steps=32).run()
    for value in result._fields.values():
        value.fill_(2.)
    for entry in result.solver._copper_foils:
        entry["spectral_norm"].fill_(2.)
        entry["spectral_real"][:, 0].fill_(2.)
        entry["spectral_real"][:, 1].fill_(7.)
        entry["spectral_imag"].zero_()
        assert entry["left_width"] == pytest.approx(.0025)
        assert entry["right_width"] == pytest.approx(.0025)
        torch.testing.assert_close(entry["area"], torch.full_like(entry["area"], .005**2))
    result._metadata["conductor_losses"] = copper_loss_report(result.solver)
    actual = result.power_loss("bulk")
    monitor = next(m for m in result.scene.monitors if m.name == "bulk")
    compiled = compile_power_loss_monitor(result.prepared_scene, monitor)
    frequencies = torch.tensor(result.frequencies, device=compiled.device, dtype=torch.float64)
    expected = actual.channel_power["conduction"].new_zeros(len(frequencies))
    for component in ("Ex", "Ey", "Ez"):
        expected += (.5*compiled.conductivity[component]*4*compiled.component_volumes[component]).sum()
    for entry in result.solver._copper_foils:
        component = entry["electric_name"]
        native_sigma = getattr(result.solver, "sigma_e_"+component)[entry["index"]]
        expected -= (.5*native_sigma.double()*4*.005**3).sum()
        expected += len(entry["indices"])*.5*.1*.0025*.005**2*49
        mask = compiled.component_masks[component]
        indices = entry["indices"]
        flat = (indices[:, 0]*mask.shape[1]+indices[:, 1])*mask.shape[2]+indices[:, 2]
        positions = torch.searchsorted(torch.nonzero(mask.flatten()).flatten(), flat)
        angle = torch.pi*frequencies*result.solver.dt
        omega = 2*torch.tan(angle)/result.solver.dt
        physical_gain = 8.8541878128e-12*1.5*1e-10*omega.square()/(1+(1e-10*omega).square())*torch.cos(angle).square()
        expected_density = .5*physical_gain[:, None]*49*.5
        torch.testing.assert_close(actual.volume_density["electric_dispersion"][component][:, positions],
            expected_density.expand(-1, len(positions)), rtol=2e-6, atol=1e-12)
    torch.testing.assert_close(actual.channel_power["conduction"], expected, rtol=2e-6, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_split_bulk_loss_records_survive_repeated_native_graph_batch():
    simulation = _simulation(graph=False, steps=128)
    reference = simulation.run().power_loss("bulk")
    batch = mw.Simulation.batch([simulation, simulation], outputs=mw.BatchOutput(fields=("Ex", "Ey", "Ez")),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=2, graph="required", graph_steps=7))
    with batch.prepare() as prepared:
        first = prepared.run().results()
        second = prepared.run().results()
        for result in (*first, *second):
            torch.testing.assert_close(result.power_loss("bulk").total, reference.total, rtol=2e-4, atol=1e-20)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("normalize", (False, True))
def test_pulse_bulk_loss_frequency_selection_uses_same_source_normalization(normalize):
    result = _simulation(graph=True, steps=512, pulse=True, normalize_source=normalize).run()
    full = result.power_loss("bulk")
    subset = result.power_loss("subset")
    torch.testing.assert_close(subset.total, full.total[1:])
    for channel in ("conduction", "electric_dispersion"):
        torch.testing.assert_close(subset.channel_power[channel], full.channel_power[channel][1:])
    assert bool(torch.all(full.total > 0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_split_spectral_bulk_loss_rejects_an_unregistered_side_gradient():
    result = _simulation(steps=64).run()
    for value in result._fields.values():
        value.requires_grad_(True)
    with pytest.raises(NotImplementedError, match="differentiable side spectra"):
        result.power_loss("bulk")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("pulse", (False, True))
def test_complex_bloch_bulk_loss_combines_replicas_before_sparse_replacement(pulse):
    from tests.gradients.test_fdtd_copper_foil_adjoint import BlochFoilDesign
    from tests.materials.surface_impedance.test_copper_foil_material_ade import _local_material
    design = BlochFoilDesign()
    design.requires_grad_(False)
    scene = design.to_scene()
    if pulse:
        source = scene.sources[0]
        scene = scene.clone(sources=(mw.UniformCurrentSource(name=source.name,
            size=source.size, center=source.center, polarization=source.polarization,
            source_time=mw.GaussianPulse(3.e9, fwidth=2.4e9)),))
    scene.structures = (*scene.structures, mw.Structure(name="substrate", geometry=mw.Box(
        position=(0., 0., 0.), size=(.06, .06, .06)), material=_local_material("debye")))
    scene.monitors = (*scene.monitors, mw.PowerLossMonitor("bulk", position=(0., 0., 0.), size=(.06, .06, .06),
        channels=("conduction", "electric_dispersion")))
    result = mw.Simulation.fdtd(scene, frequencies=(2.5e9, 3.e9), run_time=mw.TimeConfig(time_steps=128),
        full_field_dft=True, cuda_graph=True, spectral_sampler=mw.SpectralSampler(window="none")).run()
    assert result.solver.complex_fields_enabled
    assert any(bool(entry["spectral_real"].abs().sum() > 0) for entry in result.solver._copper_foils if entry["imaginary"])
    traces = result.conductor_losses["foil"]["bulk_electric_traces"]
    assert sorted(trace["component"] for trace in traces) == ["Ey", "Ez"]
    for trace in traces:
        entry = next(entry for entry in result.solver._copper_foils if entry["electric_name"] == trace["component"])
        partner = next(entry for entry in result.solver._copper_foils if entry["electric_name"] == trace["component"]+"_imag")
        # Same per-reference-period (pulse) or CW phasor scaling as the library.
        scale = spectral_scale(entry["spectral_norm"].clamp_min(1)[:, None, None],
            phasor_factor=spectral_phasor_factor(result.solver),
            integration_dt=result.solver._spectral_integration_dt,
            reference_frequency=result.solver._spectral_reference_frequency,
            frequency=entry["spectral_frequencies"][:, None, None])
        combined = (torch.complex(entry["spectral_real"], entry["spectral_imag"])
            + 1j*torch.complex(partner["spectral_real"], partner["spectral_imag"])) * scale
        torch.testing.assert_close(trace["electric_fields"], combined.cpu())
    data = result.power_loss("bulk")
    for channel in ("conduction", "electric_dispersion"):
        assert bool(torch.all(data.channel_power[channel] > 0))
        torch.testing.assert_close(data.channel_power[channel], sum((density*data.component_volumes[component][None]).sum(-1)
            for component, density in data.volume_density[channel].items()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_source_patch_overlapping_foil_fails_before_injecting_unbalanced_power():
    from tests.gradients.test_fdtd_copper_foil_adjoint import BlochFoilDesign
    design = BlochFoilDesign()
    design.requires_grad_(False)
    scene = design.to_scene()
    scene.sources = (mw.PointDipole(name="overlap", position=(-.01, 0., 0.), polarization="Ez", width=.003,
                                   source_time=mw.CW(3.e9)),)
    with pytest.raises(NotImplementedError, match="joint source/trace elimination"):
        mw.Simulation.fdtd(scene, frequency=3.e9).prepare()
