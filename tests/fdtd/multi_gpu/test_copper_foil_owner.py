"""Two-GPU split traces reproduce the single-domain finite foil stencil."""
import pytest
import torch
import numpy as np

import witwin.maxwell as mw


def _simulation(*, normal=0, coordinate=0., overlap=True, parallel=False, kind=None, pml=False,
                nonuniform=False, steps=96):
    size = [.02]*3
    position = [0.]*3
    size[normal], position[normal] = 0., coordinate
    structures = [mw.Structure(name="foil", geometry=mw.Box(
        position=torch.tensor(position, dtype=torch.float64), size=torch.tensor(size, dtype=torch.float64)),
        material=mw.CopperFoil(thickness=2e-6, frequency_range=(.5e9, 2e9)))]
    if kind is not None:
        poles = dict(debye_poles=(mw.DebyePole(delta_eps=1.5, tau=1e-10),)) if kind == "electric" else dict(
            mu_lorentz_poles=(mw.LorentzPole(delta_eps=.5, resonance_frequency=2.5e9, gamma=.3e9),))
        structures.append(mw.Structure(name="substrate", geometry=mw.Box(position=(0.,0.,0.), size=(.04,)*3),
            material=mw.Material(eps_r=2., sigma_e=.01, **poles)))
    grid = (mw.GridSpec.custom(np.array([-.02,-.015,-.01,-.005,0.,.004,.01,.016,.02]),
            np.linspace(-.02,.02,9), np.linspace(-.02,.02,9)) if nonuniform else mw.GridSpec.uniform(.005))
    # The ownership control needs a supported remote drive on every normal.
    # Its former y=0 source lies directly on a y-normal split electric trace.
    source_position = (-.01, -.01 if normal == 1 else 0., -.01)
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.02,.02),)*3), grid=grid,
        boundary=mw.BoundarySpec.pml(num_layers=3) if pml else mw.BoundarySpec.none(),
        device="cuda:0", structures=structures,
        sources=[mw.PointDipole(name="drive", position=source_position, profile="ideal",
            polarization="Ex" if normal == 2 else "Ez", source_time=mw.CW(frequency=1e9))])
    config = mw.FDTDParallelConfig(devices=("cuda:0","cuda:1"), transport="cuda_p2p", overlap=overlap,
                                  gather_fields=True) if parallel else None
    return mw.Simulation.fdtd(scene, frequency=1e9, parallel=config, cuda_graph=False,
        full_field_dft=True, run_time=mw.TimeConfig(time_steps=steps, dt=5e-12),
        spectral_sampler=mw.SpectralSampler(window="none"))


@pytest.mark.parametrize("normal", (0, 1, 2))
@pytest.mark.parametrize("overlap", (False, True))
def test_foil_all_normals_partition_plane_and_crossing_match_single_domain(cuda_p2p_devices, normal, overlap):
    expected = _simulation(normal=normal).run()
    actual = _simulation(normal=normal, overlap=overlap, parallel=True).run()
    _assert_matches(actual, expected, normal)


def _assert_matches(actual, expected, normal):
    solver = actual.solver
    count = sum(len(entry["indices"]) for entry in expected.solver._copper_foils)
    assert solver._distributed_copper.owned_edge_count == count
    assert bool(solver._distributed_copper.shadow_halo_bytes_per_step) == (normal != 0)
    for name in ("Ex","Ey","Ez","Hx","Hy","Hz"):
        fields = tuple(getattr(shard.solver, name) for shard in solver.shards)
        gathered = solver._gather_component(name, fields)
        reference = getattr(expected.solver, name)
        relative = torch.linalg.vector_norm(gathered-reference)/torch.linalg.vector_norm(reference).clamp_min(1e-30)
        assert float(relative) < 3e-5, (name, float(relative))
    for name in expected.fields:
        torch.testing.assert_close(actual.tensor(name), expected.tensor(name), rtol=3e-4, atol=1e-6)
    report, reference = actual.conductor_losses["foil"], expected.conductor_losses["foil"]
    assert report["dissipated_energy"] == pytest.approx(reference["dissipated_energy"], rel=3e-4)
    torch.testing.assert_close(report["surface_power"], reference["surface_power"], rtol=3e-4, atol=1e-12)
    assert report["active_tangential_edge_count"] == reference["active_tangential_edge_count"]


@pytest.mark.parametrize("normal,coordinate,kind,pml,nonuniform", [
    (0,-.005,"electric",True,False), (0,.005,"magnetic",True,False),
    (1,0.,"electric",False,False), (2,0.,"magnetic",False,False),
    (1,0.,"magnetic",True,False), (2,0.,"electric",True,False),
    (1,0.,None,False,True), (2,0.,"electric",False,True)])
def test_owned_foil_off_split_cpml_and_material_poles_match_single_domain(
        cuda_p2p_devices, normal, coordinate, kind, pml, nonuniform):
    options = dict(normal=normal, coordinate=coordinate, kind=kind, pml=pml, nonuniform=nonuniform)
    _assert_matches(_simulation(parallel=True, **options).run(), _simulation(**options).run(), normal)


def test_sparse_shadow_halo_normal_curl_is_load_bearing(cuda_p2p_devices, monkeypatch):
    from witwin.maxwell.fdtd.distributed import copper
    options = dict(normal=2, steps=1)
    reference = _simulation(**options).prepare()
    actual = _simulation(parallel=True, **options).prepare()
    for entry in reference.solver._copper_foils:
        entry["right_e"].fill_(.1)
    for shard in actual.solver.shards:
        for entry in shard.solver._copper_foils:
            entry["right_e"].fill_(.1)
    monkeypatch.setattr(copper, "correct_copper_magnetic_support", lambda *args: None)
    expected, faulty = reference.run(), actual.run()
    field = faulty.solver._gather_component("Hz", tuple(shard.solver.Hz for shard in faulty.solver.shards))
    assert float(torch.linalg.vector_norm(field-expected.solver.Hz)) > 1e-6


def test_distributed_foil_result_loss_metadata_roundtrip(cuda_p2p_devices, tmp_path):
    result = _simulation(normal=2, parallel=True).run()
    result.save(tmp_path / "foil.pt")
    loaded = mw.Result.load(tmp_path / "foil.pt", scene=result.scene)
    torch.testing.assert_close(loaded.conductor_losses["foil"]["surface_power"],
                               result.conductor_losses["foil"]["surface_power"])


def _mna_simulation(*, parallel=False, overlap=False):
    base = _simulation(normal=2, kind="electric")
    ports = tuple(mw.LumpedPort(name=name, positive=(x,0.,.005), negative=(x,0.,-.005),
        voltage_path=mw.AxisPath("z"), current_surface=mw.Box(position=(x,0.,-.0025),size=(.015,.015,0.)))
        for name,x in (("left",-.01),("right",.01)))
    circuit = mw.Circuit("drive_network")
    left, right = circuit.node("left"), circuit.node("right")
    circuit.add(mw.CurrentSource("drive", left, circuit.ground, 0., waveform=mw.SineWaveform(0., .001, 1e9)))
    circuit.add(mw.Resistor("link",left,right,75.))
    circuit.add(mw.Resistor("load",right,circuit.ground,50.))
    circuit.add(mw.Capacitor("cap",left,circuit.ground,1e-12))
    circuit.bind_port("left",positive=left,negative=circuit.ground)
    circuit.bind_port("right",positive=right,negative=circuit.ground)
    scene = base.scene.clone(sources=(), ports=ports, circuits=(circuit,))
    config = mw.FDTDParallelConfig(devices=("cuda:0","cuda:1"),transport="cuda_p2p",overlap=overlap,
                                  gather_fields=True) if parallel else None
    return mw.Simulation.fdtd(scene, frequency=1e9, parallel=config, cuda_graph=False, full_field_dft=True,
        run_time=mw.TimeConfig(time_steps=96,dt=5e-12),spectral_sampler=mw.SpectralSampler(window="none"))


@pytest.mark.parametrize("overlap", (False,True))
def test_distributed_foil_lossy_ade_and_remote_mna_owner_match_single_domain(cuda_p2p_devices, overlap):
    actual, expected = _mna_simulation(parallel=True,overlap=overlap).run(), _mna_simulation().run()
    _assert_matches(actual, expected, 2)
    for name in ("left","right"):
        for component in ("voltage","current","a","b"):
            torch.testing.assert_close(getattr(actual.port(name),component),getattr(expected.port(name),component),
                                       rtol=3e-4,atol=1e-8)
    torch.testing.assert_close(actual.circuit("drive_network").node_voltages,
                               expected.circuit("drive_network").node_voltages,rtol=3e-4,atol=1e-8)


def test_distributed_foil_lossy_ade_and_remote_network_owner_match_single_domain(cuda_p2p_devices):
    from tests.fdtd.multi_gpu.test_network_owner import _two_port_network
    base = _simulation(normal=2, kind="electric")
    ports = tuple(mw.LumpedPort(name=name, positive=(x,0.,.005), negative=(x,0.,-.005),
        voltage_path=mw.AxisPath("z"), current_surface=mw.Box(position=(x,0.,-.0025),size=(.015,.015,0.)))
        for name,x in (("left_port",-.01),("right_port",.01)))
    scene = base.scene.clone(ports=ports, networks=(_two_port_network(),))
    common = dict(frequency=1e9, cuda_graph=False, full_field_dft=True,
        run_time=mw.TimeConfig(time_steps=96,dt=5e-12),spectral_sampler=mw.SpectralSampler(window="none"))
    actual = mw.Simulation.fdtd(scene,parallel=mw.FDTDParallelConfig(devices=("cuda:0","cuda:1"),
        transport="cuda_p2p",gather_fields=True),**common).run()
    expected = mw.Simulation.fdtd(scene,**common).run()
    _assert_matches(actual, expected, 2)
    for name in ("left_port","right_port"):
        for component in ("voltage","current","a","b"):
            torch.testing.assert_close(getattr(actual.port(name),component),getattr(expected.port(name),component),
                                       rtol=3e-4,atol=1e-8)
    for component in ("voltage","current"):
        torch.testing.assert_close(getattr(actual.embedded_networks["split_net"],component),
            getattr(expected.embedded_networks["split_net"],component),rtol=3e-4,atol=1e-8)


def test_distributed_split_trace_bulk_loss_matches_global_reference(cuda_p2p_devices, tmp_path):
    base = _simulation(normal=2,kind="electric")
    scene = base.scene.clone(monitors=(mw.PowerLossMonitor("bulk",position=(0.,0.,0.),size=(.04,)*3,
        channels=("conduction","electric_dispersion")),))
    common = dict(frequency=1e9,cuda_graph=False,full_field_dft=True,
        run_time=mw.TimeConfig(time_steps=96,dt=5e-12),spectral_sampler=mw.SpectralSampler(window="none"))
    actual = mw.Simulation.fdtd(scene,parallel=mw.FDTDParallelConfig(devices=("cuda:0","cuda:1"),
        transport="cuda_p2p",gather_fields=True),**common).run()
    expected = mw.Simulation.fdtd(scene,**common).run()
    for channel in ("conduction","electric_dispersion"):
        torch.testing.assert_close(actual.power_loss("bulk").channel_power[channel],
            expected.power_loss("bulk").channel_power[channel],rtol=3e-4,atol=1e-20)
    actual.save(tmp_path/"power.pt")
    loaded = mw.Result.load(tmp_path/"power.pt",scene=scene,map_location="cuda:0")
    torch.testing.assert_close(loaded.power_loss("bulk").total,actual.power_loss("bulk").total)


def test_distributed_port_cannot_deposit_on_copper_split_trace(cuda_p2p_devices):
    base = _mna_simulation(parallel=True)
    foil = base.scene.structures[0]
    scene = base.scene.clone(structures=(mw.Structure(name="foil",material=foil.material,geometry=mw.Box(
        position=torch.tensor([-.01,0.,0.],dtype=torch.float64),
        size=torch.tensor([0.,.02,.02],dtype=torch.float64))),*base.scene.structures[1:]))
    simulation = mw.Simulation.fdtd(scene,frequency=1e9,parallel=base.config.parallel,
        run_time=mw.TimeConfig(time_steps=96,dt=5e-12))
    with pytest.raises(NotImplementedError,match="overlaps.*CopperFoil"):
        simulation.prepare()


def test_distributed_foil_pec_via_contact_across_partition_matches_global(cuda_p2p_devices):
    base = _simulation(normal=2,kind="electric")
    scene = base.scene.clone(structures=(*base.scene.structures,mw.Structure(name="via",material=mw.Material.pec(),
        geometry=mw.Box(position=(0.,0.,.005),size=(.01,.01,.01)))))
    common = dict(frequency=1e9,cuda_graph=False,full_field_dft=True,
        run_time=mw.TimeConfig(time_steps=96,dt=5e-12),spectral_sampler=mw.SpectralSampler(window="none"))
    actual = mw.Simulation.fdtd(scene,parallel=mw.FDTDParallelConfig(devices=("cuda:0","cuda:1"),
        transport="cuda_p2p",gather_fields=True),**common).run()
    expected = mw.Simulation.fdtd(scene,**common).run()
    _assert_matches(actual,expected,2)
    assert actual.conductor_losses["foil"]["pec_contact_edge_count"] > 0
    assert actual.conductor_losses["foil"]["pec_contact_edge_count"] == expected.conductor_losses["foil"]["pec_contact_edge_count"]


def test_distributed_foil_rejects_unqualified_thin_wire_combination(cuda_p2p_devices):
    base = _simulation(normal=2)
    scene = base.scene.clone(thin_wires=(mw.ThinWire(name="wire",points=((-.01,0.,.01),(.01,0.,.01)),
        radius=.0002,conductor=mw.WireConductor.pec(),snap="strict"),))
    simulation = mw.Simulation.fdtd(scene,frequency=1e9,
        parallel=mw.FDTDParallelConfig(devices=("cuda:0","cuda:1"),transport="cuda_p2p"))
    with pytest.raises(NotImplementedError,match="CopperFoil and ThinWire"):
        simulation.prepare()
