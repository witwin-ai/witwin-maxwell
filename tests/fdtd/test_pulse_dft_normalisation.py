"""Executed-record pulse spectra and consistent source-normalized ports."""
from dataclasses import replace

import pytest
import torch

import witwin.maxwell as mw
from scripts.dev.fdtd.dft_horizon_normalisation_repro import build_scene

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_mixed_sources_reject_absolute_spectra_but_keep_ratios(tmp_path):
    from witwin.maxwell.postprocess.scattering_parameters import compute_s_parameters
    from witwin.maxwell.result import _RatioOnlyPortData

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.02, .02),) * 3), grid=mw.GridSpec.uniform(.005),
        boundary=mw.BoundarySpec.none(), device="cuda",
        ports=(mw.LumpedPort("feed", positive=(0., 0., .005), negative=(0., 0., -.005),
            voltage_path=mw.AxisPath("z"), current_surface=mw.Box(
                position=(0., 0., -.0025), size=(.015, .015, 0.))),),
        sources=(mw.PointDipole((.01, 0., 0.), polarization="Ez",
            source_time=mw.GaussianPulse(3e9, fwidth=1.5e9)),),
        monitors=(mw.PointMonitor("point", (.005, .005, 0.), fields=("Ez",)),
                  mw.FieldTimeMonitor("trace", components=("Ez",)),
                  mw.FluxMonitor("flux", axis="x", position=.005)),
    )
    kwargs = dict(frequency=3e9, run_time=mw.TimeConfig(time_steps=96),
        shutoff=0., full_field_dft=True, cuda_graph=False,
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=False))
    result = mw.Simulation.fdtd(scene,
        excitations=mw.PortExcitation("feed", source_time=mw.CW(3e9)), **kwargs).run()
    assert result.stats()["time_plan"]["mixed_sources"]
    assert torch.isfinite(result.monitor("trace")["components"]["Ez"]).all()
    message = ("Absolute spectra of a run mixing ongoing (CW / undamped / periodic / DC) "
        "and transient sources have no horizon-independent scale; use normalize_source=True, "
        "ratios such as S/Z, or a run with only ongoing or only transient sources.")
    port = result.port("feed")
    assert isinstance(port, _RatioOnlyPortData)
    for name in ("z_in", "reflection_coefficient", "return_loss_db", "vswr"):
        assert torch.equal(getattr(port, name), getattr(result._ports["feed"], name))
    for read in (lambda: result.field("Ez"), lambda: result.fields,
                 lambda: result.monitor("point"), lambda: result.raw_monitor("flux"),
                 lambda: result.monitors, lambda: result.raw_output,
                 lambda: port.voltage, lambda: port.current, lambda: port.available_power,
                 lambda: port.incident_power, lambda: result.ports["feed"].a,
                 lambda: result.embedded_networks, lambda: result.embedded_network("feed"),
                 lambda: result._snapshot_payload(fields={})):
        with pytest.raises(ValueError) as error:
            read()
        assert str(error.value) == message
    torch.testing.assert_close(port.reflection_coefficient,
        (port.z_in - port.z0) / (port.z_in + port.z0))
    assert torch.isfinite(port.return_loss_db).all()
    ratios = compute_s_parameters(result, "flux", transmitted_monitor="flux", incident_power=1.)
    assert torch.isfinite(ratios["S11"]).all() and torch.isfinite(ratios["S21"]).all()
    with pytest.raises(ValueError) as error:
        ratios["P_incident"]
    assert str(error.value) == message
    sweep = mw.Simulation.fdtd(scene,
        excitations=mw.PortSweep(source_time=mw.CW(3e9)), **kwargs).run()
    assert torch.isfinite(sweep.network.s).all()
    assert torch.isfinite(sweep.network.to_z()).all()
    with pytest.raises(ValueError) as error:
        sweep.port("feed").voltage
    assert str(error.value) == message
    assert torch.equal(sweep.port("feed").reflection_coefficient,
                       sweep._ports["feed"].reflection_coefficient)
    path = tmp_path / "mixed.pt"
    sweep.save(path)
    restored = mw.Result.load(path, scene=scene, map_location="cuda")
    assert isinstance(restored.port("feed"), _RatioOnlyPortData)
    assert torch.equal(restored.network.s, sweep.network.s)
    assert torch.equal(restored.port("feed").z_in, sweep.port("feed").z_in)
    for name in ("voltage", "current", "available_power", "a", "b",
                 "incident_power", "reflected_power", "accepted_power", "delivered_power"):
        for view in (*result.ports.values(), *sweep.ports.values(), *restored.ports.values()):
            with pytest.raises(ValueError) as error:
                getattr(view, name)
            assert str(error.value) == message


def test_pec_dipole_spectrum_is_independent_of_horizon_and_shutoff(record_property):
    powers = {}
    for cycles in (40, 200):
        for shutoff in (0., 1e-5):
            scene, frequencies, excitations = build_scene("dipole", 24, 3)
            scene.monitors.clear()
            result = mw.Simulation.fdtd(
                scene, frequencies=frequencies, excitations=excitations,
                run_time=mw.TimeConfig.auto(steady_cycles=cycles, transient_cycles=8),
                shutoff=shutoff, full_field_dft=False,
            ).run()
            assert result.stats()["time_steps"] == (6976 if cycles == 40 else 27376)
            powers[cycles, shutoff] = result.port("feed").incident_power
    for (cycles, shutoff), power in powers.items():
        record_property(f"P_inc_{cycles}_{shutoff}", power.detach().cpu().tolist())
    record_property("horizon_relative_error", float(((powers[40, 0.] - powers[200, 0.]).abs() / powers[40, 0.]).max()))
    torch.testing.assert_close(powers[40, 0.], powers[200, 0.], rtol=.005, atol=0.)
    assert torch.equal(powers[40, 1e-5], powers[200, 1e-5])
    torch.testing.assert_close(powers[40, 0.], powers[40, 1e-5], rtol=.005, atol=0.)


def test_source_normalization_scales_ports_fields_powers_and_port_gradients():
    for waveform in (mw.GaussianPulse(3e9, fwidth=1.5e9, amplitude=1.3, phase=.43),
                     mw.CW(3e9, amplitude=1.3, phase=.43)):
        runs = []
        for normalized in (False, True):
            amplitude = torch.tensor(.8, device="cuda", requires_grad=True)
            scene = mw.Scene(
                domain=mw.Domain(bounds=((-0.02, .02),) * 3), grid=mw.GridSpec.uniform(.005),
                boundary=mw.BoundarySpec.none(), device="cuda",
                ports=(mw.LumpedPort("feed", positive=(0., 0., .005), negative=(0., 0., -.005),
                    voltage_path=mw.AxisPath("z"), current_surface=mw.Box(
                        position=(0., 0., -.0025), size=(.015, .015, 0.))),),
                monitors=(mw.PointMonitor("point", (.01, .005, 0.), fields=("Ez",)),),
            )
            scene.add_thin_wire(mw.ThinWire(
                "wire", points=((.01, -.005, 0.), (.01, .005, 0.)),
                radius=.0003, conductor=mw.WireConductor.pec(), snap="continuous",
            ))
            scene.add_monitor(mw.WireMonitor("wire_state", "wire", frequencies=(3e9,)))
            prepared = mw.Simulation.fdtd(
                scene, frequency=3e9, run_time=mw.TimeConfig(time_steps=96), shutoff=0.,
                excitations=mw.PortExcitation("feed", amplitude=amplitude, source_time=waveform),
                spectral_sampler=mw.SpectralSampler(window="none", normalize_source=normalized),
            ).prepare()
            result = prepared.run()
            port = result.port("feed")
            # Linearity gives dP/dA = 2P/A, independently of DFT conventions.
            power = port.available_power.sum() + port.accepted_power.sum()
            objective = power / power.detach()
            objective.backward()
            torch.testing.assert_close(amplitude.grad, 2 / amplitude.detach(), rtol=.002, atol=0.)
            runs.append((result, port))
        dt = prepared.solver.dt
        reference = replace(waveform, amplitude=1., phase=0.)
        times = (torch.arange(96, device="cuda", dtype=torch.float64) + .5) * dt
        signal = torch.tensor([reference.evaluate(n * dt + .5 * dt) for n in range(96)],
                              device="cuda", dtype=torch.float64)
        scale = 2 / 96 if isinstance(waveform, mw.CW) else dt * 3e9
        spectrum = ((signal * torch.exp(2j * torch.pi * 3e9 * times)).sum() * scale).to(torch.complex64)
        raw, normalized = runs[0][1], runs[1][1]
        for name in ("voltage", "current", "a", "b"):
            torch.testing.assert_close(getattr(normalized, name), getattr(raw, name) / spectrum,
                                       rtol=1e-5, atol=0.)
        for name in ("incident_power", "reflected_power", "accepted_power", "delivered_power", "available_power"):
            torch.testing.assert_close(getattr(normalized, name), getattr(raw, name) / spectrum.abs().square(),
                                       rtol=1e-5, atol=0.)
        torch.testing.assert_close(normalized.reflection_coefficient, raw.reflection_coefficient, rtol=1e-5, atol=0.)
        torch.testing.assert_close(torch.as_tensor(runs[1][0].monitor("point")["Ez"]),
                                   torch.as_tensor(runs[0][0].monitor("point")["Ez"]) / spectrum,
                                   rtol=1e-5, atol=0.)
        for name in ("current", "charge"):
            torch.testing.assert_close(getattr(runs[1][0].monitor("wire_state"), name),
                                       getattr(runs[0][0].monitor("wire_state"), name) / spectrum,
                                       rtol=1e-5, atol=0.)


def test_reference_frequency_scales_amplitudes_and_round_trips(tmp_path):
    results = []
    for reference_frequency in (None, 4.8e9):
        scene, frequencies, excitations = build_scene("dipole", 24, 3)
        result = mw.Simulation.fdtd(
            scene, frequencies=frequencies, excitations=excitations,
            reference_frequency=reference_frequency, full_field_dft=True,
            run_time=mw.TimeConfig(time_steps=96), shutoff=0.,
        ).run()
        resolved = "per-frequency" if reference_frequency is None else reference_frequency
        assert result._metadata["reference_frequency"] == resolved
        assert result.stats()["reference_frequency"] == resolved
        assert result.port("feed").metadata["reference_frequency"] == resolved
        results.append(result)
    default, explicit = results
    factors = 4.8e9 / default.port("feed").frequencies
    for name in ("voltage", "current", "a", "b"):
        torch.testing.assert_close(getattr(explicit.port("feed"), name),
                                   factors * getattr(default.port("feed"), name), rtol=1e-6, atol=0.)
    # The largest factor is 2.5: scaling a rounded subnormal then rounding
    # again can differ from scaling the raw sum by up to two subnormal ULPs.
    subnormal_ulp = torch.finfo(torch.float32).tiny * torch.finfo(torch.float32).eps
    for name in default.fields:
        torch.testing.assert_close(explicit.fields[name],
            factors.to(default.fields[name].real.dtype)[:, None, None, None] * default.fields[name],
            rtol=1e-6, atol=2 * subnormal_ulp)
    point = torch.as_tensor(default.monitor("point")["Ez"])
    torch.testing.assert_close(torch.as_tensor(explicit.monitor("point")["Ez"]),
        factors.to(point.real) * point, rtol=1e-6, atol=0.)
    torch.testing.assert_close(explicit.port("feed").incident_power,
        factors.square() * default.port("feed").incident_power, rtol=1e-6, atol=0.)
    path = tmp_path / "reference_frequency.pt"
    explicit.save(path)
    restored = mw.Result.load(path, scene=explicit.scene, map_location="cuda")
    assert restored._metadata["reference_frequency"] == 4.8e9
    assert restored.stats()["reference_frequency"] == 4.8e9


def test_mixed_source_guard_survives_a_disk_checkpoint_resume(tmp_path):
    from witwin.maxwell.result import _RatioOnlyPortData

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.02, .02),) * 3), grid=mw.GridSpec.uniform(.005),
        boundary=mw.BoundarySpec.none(), device="cuda",
        ports=(mw.LumpedPort("feed", positive=(0., 0., .005), negative=(0., 0., -.005),
            voltage_path=mw.AxisPath("z"), current_surface=mw.Box(
                position=(0., 0., -.0025), size=(.015, .015, 0.))),),
        sources=(mw.PointDipole((.01, 0., 0.), polarization="Ez",
            source_time=mw.GaussianPulse(3e9, fwidth=1.5e9)),),
        monitors=(mw.PointMonitor("point", (.005, .005, 0.), fields=("Ez",)),),
    )

    def simulation():
        return mw.Simulation.fdtd(scene, excitations=mw.PortExcitation("feed", source_time=mw.CW(3e9)),
            frequency=3e9, run_time=mw.TimeConfig(time_steps=96), shutoff=0., cuda_graph=False,
            spectral_sampler=mw.SpectralSampler(window="none", normalize_source=False))

    path = tmp_path / "mixed.pt"
    simulation().prepare().run_until(48).save(path)
    restored = mw.FDTDResumeCheckpoint.load(path, map_location="cpu")
    # The derived classification is recomputed by preparation, not read from the checkpoint.
    result = simulation().prepare().run(resume_from=restored)
    assert result.stats()["time_plan"]["mixed_sources"]
    port = result.port("feed")
    assert isinstance(port, _RatioOnlyPortData)
    assert torch.isfinite(port.z_in).all()
    for read in (lambda: result.monitor("point"), lambda: port.voltage):
        with pytest.raises(ValueError, match="no horizon-independent scale"):
            read()
