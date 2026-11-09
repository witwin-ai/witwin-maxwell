"""Independent native metric-annulus controls with an explicit private ring drive.

The public-metric option uses Scene coordinate compilation and its stable CFL.
The historical private path samples arrays after preparation for comparison.
Source amplitudes are arbitrary ring-mode drives, not a claimed physical
plane-wave mapping. Two drives measure the complete Hankel S matrix.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from scipy.special import hankel1,hankel2
import torch
import witwin.maxwell as mw
from witwin.maxwell.fdtd.copper_foil import initialize_copper_foils
from scripts.dev.fdtd.copper_cylinder_reference import shell_scattering
from scripts.dev.fdtd.pcb_acceptance import package_identity


def coherent_reference_heat(scattering, incoming):
    """Normalized absorbed work for the complete measured complex input."""
    incoming=np.asarray(incoming)
    amplitudes=incoming[:,0]+1j*incoming[:,1]
    absorption=np.eye(2)-scattering.conj().T@scattering
    return float(np.vdot(amplitudes,absorption@amplitudes).real/np.vdot(amplitudes,amplitudes).real)


@contextmanager
def _private_ring_drive(wavevector, *, public_metric=False):
    # A chart-periodic arbitrary drive suffices for two-port identification:
    # both incident amplitudes are measured independently, not assumed from
    # source normalization. This does not register a physical mapped source.
    from witwin.maxwell.fdtd.excitation import injection
    from witwin.maxwell.fdtd.boundary import runtime
    from witwin.maxwell.compiler import coordinates
    original=injection._validate_soft_surface_source_boundary
    original_wavevector=runtime.resolve_bloch_wavevector
    original_coordinate_validation=coordinates.validate_coordinate_scene
    injection._validate_soft_surface_source_boundary=lambda *args: None
    runtime.resolve_bloch_wavevector=lambda solver: wavevector
    if public_metric:
        coordinates.validate_coordinate_scene=lambda scene,**kwargs: original_coordinate_validation(scene.clone(sources=[]),**kwargs)
    try:
        yield
    finally:
        injection._validate_soft_surface_source_boundary=original
        runtime.resolve_bloch_wavevector=original_wavevector
        coordinates.validate_coordinate_scene=original_coordinate_validation


def _run(polarization,order,direction,spacing,foil,*,radius,frequency,public_metric=False,save_path=None):
    sigma=5.8e7
    delta=math.sqrt(2/(2*math.pi*frequency*mw.constants.MU_0*sigma))
    k=2*math.pi*frequency/mw.constants.C_0
    angle=math.asin(order/(radius*k))
    carrier=(math.sin(angle),direction*math.cos(angle),0.)
    electric=(0.,0.,1.) if polarization=="tm" else (direction*math.cos(angle),-math.sin(angle),0.)
    kinds=dict(x="bloch" if order else "periodic",y="pml",z="bloch" if order else "periodic")
    radial_samples=np.arange(-radius/2+spacing/2,radius/2,spacing)
    radial_samples=radial_samples[(abs(radial_samples)>=radius/12)&(abs(radial_samples)<=radius*7/30)]
    scene=mw.Scene(domain=mw.Domain(bounds=((-2*spacing,2*spacing),(-radius/2,radius/2),(-2*spacing,2*spacing))),
        grid=mw.GridSpec.uniform(spacing),boundary=mw.BoundarySpec.faces(default="none",num_layers=12,
            bloch_wavevector="auto" if order else (0.,0.,0.),**kinds),device="cuda",
        coordinates=mw.BendCoordinates(curvature=-1/radius) if public_metric else None,
        structures=(mw.Structure(name="film",geometry=mw.Box(position=(0.,0.,0.),size=(16*spacing,0.,16*spacing)),
            material=mw.CopperFoil(thickness=delta,conductivity=sigma,frequency_range=(frequency/2,2*frequency))),) if foil else (),
        monitors=tuple(mw.PointMonitor(f"magnetic_profile_{index}",(spacing/2,float(u),0.),fields=("Hz",))
                       for index,u in enumerate(radial_samples)) if polarization=="te" else (),
        sources=(mw.PlaneWave(name="ring_mode_drive",direction=carrier,polarization=electric,injection_axis="y",
            source_time=mw.CW(frequency,amplitude=80.)),))
    with _private_ring_drive((order/radius,0.,0.),public_metric=public_metric):
        prepared=mw.Simulation.fdtd(scene,frequency=frequency,cuda_graph=True,full_field_dft=True,
            run_time=mw.TimeConfig(transient_cycles=24,steady_cycles=20,dt=None if public_metric else .2*spacing/mw.constants.C_0),
            spectral_sampler=mw.SpectralSampler(window="hanning")).prepare()
    solver=prepared.solver
    for component,name in enumerate(("Ex","Ey","Ez","Hx","Hy","Hz")) if not public_metric else ():
        magnetic=name[0]=="H"
        own="xyz".index(name[1].lower())
        normal_half=(own==1) != magnetic
        y=solver.scene.y_half if normal_half else solver.scene.y
        h=1+y.double()/radius
        factor=1/h if own==0 else h
        parameter=("mu_" if magnetic else "eps_")+name
        native=getattr(solver,parameter)
        native.mul_(factor.to(native.dtype)[None,:,None])
    if not public_metric:
        solver._build_update_coefficients()
        initialize_copper_foils(solver)
    result=prepared.run()
    component="Ez" if polarization=="tm" else "Hz"
    if polarization=="tm":
        field=result.chart_fields.tensor('E_z') if public_metric else result.tensor(component)
        if field.ndim==4:field=field[0]
        profile=field[field.shape[0]//2,:,field.shape[2]//2].cpu().numpy()
        u=np.asarray(solver.scene.y_nodes64)
    else:
        profile=np.array([complex(result.raw_monitor(f"magnetic_profile_{index}")["components"]["H_z" if public_metric else "Hz"].reshape(-1)[0])
                          for index in range(len(radial_samples))])
        u=radial_samples
    coefficients=[];residuals=[]
    for lo,hi in ((-radius*7/30,-radius/12),(radius/12,radius*7/30)):
        select=(u>=lo)&(u<=hi)
        r=radius+u[select]
        basis=np.stack((hankel1(order,k*r),hankel2(order,k*r)),-1)
        amplitudes=np.linalg.lstsq(basis,profile[select],rcond=None)[0]
        coefficients.append(amplitudes)
        residuals.append(float(np.linalg.norm(basis@amplitudes-profile[select])/np.linalg.norm(profile[select])))
    inner,outer=coefficients
    incoming=np.array((inner[0],outer[1]));outgoing=np.array((inner[1],outer[0]))
    if save_path is not None:
        result.save(save_path)
    constant=mw.constants.MU_0 if polarization=='tm' else mw.constants.EPSILON_0
    incoming_power=float((abs(incoming)**2).sum()*((4*spacing)**2)/(math.pi*(2*math.pi*frequency)*constant*radius))
    heat=float(result.conductor_losses['film']['surface_power'][0]) if foil else 0.
    return incoming,outgoing,dict(direction=direction,hankel_fit_relative_residuals=residuals,
        incoming_hankel_power_w=incoming_power,copper_heat_w=heat,
        direct_heat_absorption=heat/incoming_power,
        snapshot=None if save_path is None else str(save_path),
        graph_active=bool(solver._cuda_graph_active),incoming=np.stack((incoming.real,incoming.imag),-1).tolist(),
        outgoing=np.stack((outgoing.real,outgoing.imag),-1).tolist())


def source_binding():
    root=Path(__file__).resolve().parents[3]
    marker=root/'SOURCE_BASE_COMMIT'
    if not marker.exists() or not Path(mw.__file__).resolve().is_relative_to(root):
        raise RuntimeError('Annular physical controls require an immutable source archive and native package.')
    paths=(Path(__file__).resolve(),Path(shell_scattering.__code__.co_filename).resolve())
    return dict(base_commit=marker.read_text().strip(),native=package_identity(mw,'witwin-maxwell')['source_sha256'],
        files={str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in paths})


def campaign(output,*,spacing=.005,orders=(0,),polarizations=("tm","te"),radius=.3,frequency=1e9,public_metric=False,save_results=False):
    before=source_binding()
    report=dict(source_binding_before=before,radius_m=radius,frequency_hz=frequency,spacing_m=spacing,
        metric_path='public coordinate compiler and two-trace initializer' if public_metric else 'private sampled arrays',
        reference="Both physical cylindrical-shell radii, vacuum Hankel two-port; no fitted geometry or material",
        acceptance=dict(vacuum_S_absolute_error_max=.003,reflection_absolute_error_max=.003,
            transmission_relative_error_max=.03,absorption_relative_error_max=.03),rows=[])
    for order in orders:
        for polarization in polarizations:
            start=time.perf_counter()
            row=dict(order=order,polarization=polarization)
            measured={}
            for foil in (False,True):
                runs=[_run(polarization,order,direction,spacing,foil,radius=radius,frequency=frequency,public_metric=public_metric,
                    save_path=output.with_name(output.stem+f'-{polarization}{order}-'+('film' if foil else 'vacuum')+f'-{direction}.pt') if save_results else None)
                    for direction in (1,-1)]
                incoming=np.stack([run[0] for run in runs],-1)
                outgoing=np.stack([run[1] for run in runs],-1)
                scattering=np.linalg.solve(incoming.T,outgoing.T).T
                measured[foil]=scattering
                row["film" if foil else "vacuum"]=dict(S_real=scattering.real.tolist(),S_imag=scattering.imag.tolist(),runs=[run[2] for run in runs])
            delta=math.sqrt(2/(2*math.pi*frequency*mw.constants.MU_0*5.8e7))
            exact=shell_scattering(radius,delta,frequency,5.8e7,order=order,polarization=polarization)
            actual=measured[True]
            exact_absorption=1-(abs(exact)**2).sum(0)
            actual_absorption=1-(abs(actual)**2).sum(0)
            heat_absorption=[];heat_expected=[]
            for run in row['film']['runs']:
                expected=coherent_reference_heat(exact,run['incoming'])
                heat_expected.append(expected);heat_absorption.append(run['direct_heat_absorption'])
            row.update(vacuum_S_absolute_error=float(abs(measured[False]-np.array(((0,1),(1,0)))).max()),
                reflection_absolute_error=float(abs(np.diag(actual-exact)).max()),
                transmission_relative_error=float(max(abs(actual[0,1]-exact[0,1])/abs(exact[0,1]),abs(actual[1,0]-exact[1,0])/abs(exact[1,0]))),
                expected_absorption=exact_absorption.tolist(),actual_absorption=actual_absorption.tolist(),
                absorption_relative_error=float(max(abs(actual_absorption-exact_absorption)/exact_absorption)),
                reference_S_real=exact.real.tolist(),reference_S_imag=exact.imag.tolist(),elapsed_seconds=time.perf_counter()-start)
            row.update(direct_heat_absorption=heat_absorption,direct_heat_reference_absorption=heat_expected,
                direct_heat_reference_contract='a^H(I-S^H S)a / (a^H a), including measured coherent incidence at both ports; Hankel Wronskian power, no vacuum correction',
                direct_heat_relative_error=float(max(abs(np.asarray(heat_absorption)-heat_expected)/np.asarray(heat_expected))))
            row['passed']=(row['vacuum_S_absolute_error']<.003 and row['reflection_absolute_error']<.003
                           and row['transmission_relative_error']<.03 and row['absorption_relative_error']<.03)
            after=source_binding()
            if before!=after:raise RuntimeError('Annular source, controller or reference helper changed during execution; retained fields are diagnostic only.')
            report['source_binding_after']=after
            report["rows"].append(row);output.write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps(row),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--spacing",type=float,default=.005)
    parser.add_argument("--radius",type=float,default=.3)
    parser.add_argument("--frequency",type=float,default=1e9)
    parser.add_argument("--orders",type=int,nargs="+",default=(0,))
    parser.add_argument("--polarizations",nargs="+",default=("tm","te"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--public-metric",action='store_true')
    parser.add_argument("--save-results",action='store_true')
    args=parser.parse_args()
    result=campaign(args.output,spacing=args.spacing,orders=args.orders,polarizations=args.polarizations,radius=args.radius,frequency=args.frequency,public_metric=args.public_metric,save_results=args.save_results)
    if not all(row['passed'] for row in result['rows']):
        raise SystemExit('Native annulus physical acceptance failed; raw results retained at '+str(args.output))
