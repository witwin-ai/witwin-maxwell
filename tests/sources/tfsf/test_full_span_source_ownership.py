"""Physical null-scene controls for complete periodic TFSF source faces."""
import pytest
import torch

import witwin.maxwell as mw


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FDTD required")
@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("direction", (-1, 1))
@pytest.mark.parametrize("pulse", (False, True), ids=("cw", "pulse"))
def test_complete_periodic_faces_confine_incident_wave(axis, direction, pulse):
    electric = (axis+1)%3
    propagation = [0., 0., 0.]
    propagation[axis] = direction
    polarization = [0., 0., 0.]
    polarization[electric] = 1.
    bounds = [(-.12, .12)]*3
    bounds[axis] = (-.6, .6)
    faces = {name: "pml" if index == axis else "bloch" for index, name in enumerate("xyz")}
    scene = mw.Scene(domain=mw.Domain(bounds=bounds), grid=mw.GridSpec.uniform(.03),
        boundary=mw.BoundarySpec.faces(default="pml", num_layers=12, strength=1.,
            bloch_wavevector=(0., 0., 0.), **faces), device="cuda")
    waveform = mw.GaussianPulse(1e9, fwidth=.5e9) if pulse else mw.CW(1e9)
    scene.add_source(mw.PlaneWave(direction=propagation, polarization=polarization,
        source_time=waveform, injection=mw.TFSF.slab(axis="xyz"[axis], bounds=(-.24, .24))))
    for name, normal in (("inside", .09), ("outside_low", -.42), ("outside_high", .42)):
        position = [0., 0., 0.]
        position[axis] = normal
        scene.add_monitor(mw.PointMonitor(name, position=position,
            fields=("E"+"xyz"[electric],)))
    result = mw.Simulation.fdtd(scene, frequencies=(1e9,),
        run_time=mw.TimeConfig.auto(transient_cycles=0 if pulse else 40,
            steady_cycles=32 if pulse else 20),
        spectral_sampler=mw.SpectralSampler(window="none" if pulse else "hanning")).run()
    component = "E"+"xyz"[electric]
    inside = torch.as_tensor(result.monitor("inside")[component]).abs().max()
    outside = max(float(torch.as_tensor(result.monitor(name)[component]).abs().max())
        for name in ("outside_low", "outside_high"))
    assert float(inside) > 1e-6
    # The CW uses the exact native discrete plane wave. A pulse uses its fixed
    # carrier dispersion match and therefore retains a finite bandwidth error.
    assert outside/float(inside) < (1e-2 if pulse else 1e-5)
