import pytest
import torch

import witwin.maxwell as mw


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Lumped-port PEC gap validation requires CUDA.",
)


def _scene(*, embedded: bool) -> mw.Scene:
    port = mw.LumpedPort(
        name="feed",
        positive=(0.0, 0.0, 0.001),
        negative=(0.0, 0.0, -0.001),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(
            position=(0.0, 0.0, 0.0),
            size=(0.002, 0.006, 0.0),
        ),
        reference_impedance=50.0,
    )
    scene = mw.Scene(
        domain=mw.Domain(
            bounds=((-0.016, 0.016), (-0.016, 0.016), (-0.015, 0.015))
        ),
        grid=mw.GridSpec.uniform(0.002),
        boundary=mw.BoundarySpec.pml(num_layers=4),
        ports=(port,),
        device="cuda",
    )
    centers = (0.0,) if embedded else (-0.002, 0.002)
    for index, center in enumerate(centers):
        scene.add_structure(
            mw.Structure(
                name=f"terminal_{index}",
                geometry=mw.Box(
                    position=(0.0, 0.0, center),
                    size=(0.012, 0.008, 0.002),
                ),
                material=mw.Material.pec(),
            )
        )
    return scene


def test_explicit_port_gap_opens_only_the_pec_suppressed_terminal_edge():
    prepared = mw.Simulation.fdtd(
        _scene(embedded=False),
        frequency=5.0e9,
        excitations=mw.PortExcitation("feed"),
        run_time=mw.TimeConfig(time_steps=1),
    ).prepare()
    solver = prepared.solver
    runtime = solver._port_runtimes[0]
    selector = tuple(runtime.geometry.voltage_indices[:, axis] for axis in range(3))

    torch.testing.assert_close(
        solver.cez_curl[selector],
        solver.dt / solver.eps_Ez[selector],
    )
    torch.testing.assert_close(
        solver.cez_decay[selector],
        torch.ones_like(solver.cez_decay[selector]),
    )


def test_port_gap_rejects_an_edge_embedded_in_one_pec_conductor():
    simulation = mw.Simulation.fdtd(
        _scene(embedded=True),
        frequency=5.0e9,
        excitations=mw.PortExcitation("feed"),
    )

    with pytest.raises(ValueError, match="same PEC conductor|embedded inside"):
        simulation.prepare()


def test_open_probe_rejects_a_fully_pec_suppressed_gap():
    with pytest.raises(ValueError, match="Port feed is an open probe over a PEC-covered gap.*add a termination"):
        mw.Simulation.fdtd(_scene(embedded=True), frequency=5.0e9).prepare()


def test_native_port_roles():
    for excited, role in ((False, "probe"), (True, "coupled")):
        scene = _scene(embedded=False)
        scene.structures = ()
        result = mw.Simulation.fdtd(
            scene, frequency=5.0e9,
            excitations=mw.PortExcitation("feed") if excited else (),
            run_time=mw.TimeConfig(time_steps=20),
            spectral_sampler=mw.SpectralSampler(window="none"),
        ).run()
        assert result.port("feed").metadata["port_role"] == role
