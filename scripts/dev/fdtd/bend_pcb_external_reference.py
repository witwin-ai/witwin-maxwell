"""Independent cylindrical-grid bent-PCB reference with explicit model limits.

The independent engine evolves physical cylindrical fields and transforms its
own recorded surface to radiation. No native chart material tensor or near to
far field kernel is imported. Its fitted single-trace conducting-sheet model
is recorded as a distinct approximation, not a two-trace copper heat oracle.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import numpy as np
from scripts.dev.fdtd.matched_current_reference import compile_known_current_edges,install_known_current_sources,known_current_spectrum
from scripts.dev.fdtd.reference_material_dump import save_native_material_arrays

DRIVER_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def chart_to_cylinder(points, curvature):
    points=np.asarray(points,dtype=float)
    if not np.isfinite(curvature) or curvature<=0:
        raise ValueError('This independent cylindrical fixture requires positive curvature.')
    return np.stack((1/curvature-points[...,1],curvature*points[...,0]-math.pi/2,points[...,2]),-1)


def cylindrical_world(points,curvature):
    points=np.asarray(points,dtype=float)
    return np.stack((points[...,0]*np.cos(points[...,1]),
        points[...,0]*np.sin(points[...,1])+1/curvature,points[...,2]),-1)


def fixture(spacing=.01,curvature=3.,pml_width=.08,grid_construction="padded_linspace",canonical_grid_faces=False):
    """Declare physical dimensions independently of either solver compiler."""
    if spacing<=0 or pml_width<=0:
        raise ValueError('Grid and physical PML width must be positive.')
    bounds=((-0.12,.12),(-.1,.1),(-.12,.12))
    if grid_construction=='padded_linspace':
        axes=[np.linspace(lo-pml_width,hi+pml_width,round((hi-lo+2*pml_width)/spacing)+1) for lo,hi in bounds]
    elif grid_construction=='domain_then_pml':
        axes=[]
        layers=round(pml_width/spacing)
        for lo,hi in bounds:
            nodes=np.linspace(lo,hi,math.ceil((hi-lo)/spacing)+1,dtype=np.float64)
            axes.append(np.r_[nodes[0]-(nodes[1]-nodes[0])*np.arange(layers,0,-1),nodes,nodes[-1]+(nodes[-1]-nodes[-2])*np.arange(1,layers+1)])
    else:raise ValueError('Unknown declared grid construction.')
    if 1-curvature*axes[1][-1]<=0:
        raise ValueError('The full physical cylindrical domain must have positive radius.')
    grids=(1/curvature-axes[1][::-1],curvature*axes[0]-math.pi/2,axes[2])
    face_roundoff=[]
    def box(position,size):
        lower=np.asarray(position)-np.asarray(size)/2;upper=np.asarray(position)+np.asarray(size)/2
        ends=chart_to_cylinder(np.stack((lower,upper)),curvature)
        lower=np.minimum(*ends);upper=np.maximum(*ends)
        if canonical_grid_faces:
            for endpoint in (lower,upper):
                for dim,axis in enumerate(grids):
                    index=int(np.argmin(abs(axis-endpoint[dim])))
                    nearest=float(axis[index]);original=float(endpoint[dim])
                    tolerance=8*np.spacing(max(abs(original),abs(nearest),spacing))
                    if abs(original-nearest)<=tolerance:
                        if original!=nearest:face_roundoff.append(dict(axis=dim,original=original,canonical=nearest,delta=nearest-original,tolerance=tolerance))
                        endpoint[dim]=nearest
        return lower,upper
    return dict(spacing=spacing,curvature=curvature,radius=1/curvature,grid=grids,face_roundoff=face_roundoff,
        pml_layers=round(pml_width/spacing),ground=box((0.,0.,0.),(.14,0.,.14)),
        patch=box((0.,.02,0.),(.08,0.,.08)),substrate=box((0.,.01,0.),(.14,.02,.14)),
        aperture=box((.01,.01,0.),(.02,.02,.02)),
        surface=box((0.,.01,0.),(.18,.1,.18)))


def _serial(value):
    value=np.asarray(value)
    return dict(real=value.real.tolist(),imag=value.imag.tolist())


def _source_identity(path):
    result=subprocess.run(['git','rev-parse','HEAD'],cwd=path,capture_output=True,text=True,check=True)
    status=subprocess.run(['git','status','--short'],cwd=path,capture_output=True,text=True,check=True)
    records={str(item.relative_to(path)):hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.rglob('*')) if item.is_file() and item.suffix in ('.cpp','.h','.py','.pyx')}
    return dict(head=result.stdout.strip(),status=status.stdout,source_sha256=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest())


def reference_binding(reference_source):
    """Bind immutable controller overlays and the actual loaded backend binaries."""
    import sys
    root=Path(__file__).resolve().parents[3]
    marker=root/'SOURCE_BASE_COMMIT'
    if not marker.exists():
        raise RuntimeError('Independent physical controls require a frozen source archive with SOURCE_BASE_COMMIT.')
    paths={Path(__file__).resolve(),Path(compile_known_current_edges.__code__.co_filename).resolve(),Path(save_native_material_arrays.__code__.co_filename).resolve()}
    # Bind lazy extension modules before any geometry object can import them.
    package_roots={Path(sys.modules[name].__file__).resolve().parent for name in ('openEMS','CSXCAD')}
    for package_root in package_roots:
        paths.update(path.resolve() for path in package_root.rglob('*') if path.is_file() and (path.suffix=='.py' or '.so' in path.name))
    maps=Path('/proc/self/maps').read_text().splitlines()
    backend_library_directories=set()
    for line in maps:
        path=line.split()[-1]
        if path.startswith('/') and '/pcb-reference/' in path and '.so' in path:
            backend_library_directories.add(Path(path).parent)
    for directory in backend_library_directories:
        paths.update(path.resolve() for path in directory.glob('*.so*') if path.is_file())
    return dict(base_commit=marker.read_text().strip(),reference_source=_source_identity(reference_source),
        files={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)})


def run(args):
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS
    from openEMS.utilities import DFT_time2freq
    import openEMS as reference_package
    binding_before=reference_binding(args.reference_source)
    start=time.perf_counter();model=fixture(args.spacing,args.curvature,grid_construction=args.grid_construction,canonical_grid_faces=args.canonical_grid_faces)
    frequency=1e9;duration=args.duration
    # A deliberately conservative independent CFL, including the smallest
    # physical angular edge at the inner padded radius.
    radial,angular,axial=model['grid']
    physical_steps=(np.diff(radial).min(),radial[0]*np.diff(angular).min(),np.diff(axial).min())
    dt=.4/(299792458.*np.sqrt(sum(1/value**2 for value in physical_steps)))
    external=openEMS(CoordSystem=1,NrTS=math.ceil(duration/dt),EndCriteria=0.,TimeStep=dt)
    if args.source=='matched-current':
        external.SetCustomExcite('exp(-0.5*((t-4.8e-9)/6e-10)^2)*sin(2*pi*1e9*(t-4.8e-9))',frequency,3e9)
    else:external.SetGaussExcite(frequency,.7e9)
    external.SetBoundaryCond([f"PML_{model['pml_layers']}"]*6)
    csx=ContinuousStructure(CoordSystem=1);external.SetCSX(csx)
    grid=csx.GetGrid();grid.SetDeltaUnit(1.)
    for axis,values in enumerate(model['grid']):grid.AddLine(axis,values)
    for name in (() if args.vacuum else ('ground','patch')):
        if args.metal=='pec':metal=csx.AddMetal(name)
        else:metal=csx.AddConductingSheet(name,conductivity=5.8e7,thickness=35e-6)
        metal.AddBox(*model[name],priority=10)
    if not args.vacuum:
        dielectric=csx.AddMaterial('substrate',epsilon=2.2,kappa=.001)
        dielectric.AddBox(*model['substrate'],priority=0)
    lower,upper=model['aperture']
    port_start=lower.copy();port_stop=upper.copy()
    port_start[0],port_stop[0]=upper[0],lower[0]
    port=None;edges=()
    if args.source=='matched-current':
        edges=compile_known_current_edges(model['grid'],axis=0,path=(upper[0],lower[0]),
            aperture_bounds={1:(lower[1],upper[1]),2:(lower[2],upper[2])},dt=dt,
            epsilon=(1. if args.vacuum else 2.2)*8.8541878128e-12,conductivity=0. if args.vacuum else .001,cylindrical_radial=True)
        install_known_current_sources(csx,edges,amplitude=args.current_amplitude)
    else:port=external.AddLumpedPort(1,50.,port_start,port_stop,'x',excite=1.,priority=20)
    # Independently integrate the physical finite-area voltage. Its radial
    # transverse area r*dphi*dz has constant r over each averaging plane, so r
    # cancels from normalized weights. The clipped angular/axial dual lengths
    # remain essential at aperture endpoints.
    transverse=[]
    for axis in (1,2):
        nodes=model['grid'][axis]
        boundaries=np.r_[nodes[0],(nodes[1:]+nodes[:-1])/2,nodes[-1]]
        weights=np.maximum(0,np.minimum(boundaries[1:],upper[axis])-np.maximum(boundaries[:-1],lower[axis]))
        indices=np.flatnonzero(weights>0)
        transverse.append([(nodes[i],weights[i]/weights.sum()) for i in indices])
    probes=[]
    for a,(phi,wa) in enumerate(transverse[0]):
        for b,(z,wb) in enumerate(transverse[1]):
            name=f'physical_aperture_voltage_{a}_{b}'
            probe=csx.AddProbe(name,p_type=0,weight=1. if port is None else -1.)
            probe.AddBox([upper[0],phi,z],[lower[0],phi,z])
            probes.append((name,wa*wb))
    # The independent engine's custom signal builder resets f_max to f0.
    # Plain-Nyquist FD recording would alias positive/negative carrier bands.
    surface=external.CreateNF2FFBox('physical_huygens',*model['surface'],frequency=[frequency],over_sampling=8)
    args.output.mkdir(parents=True,exist_ok=True);raw=args.output/'reference-raw'
    run_input=dict(schema='independent-bend-reference-input/1',spacing=args.spacing,curvature=args.curvature,
        source=args.source,metal=args.metal,vacuum=args.vacuum,dt=dt,duration=duration,
        current_amplitude=args.current_amplitude if port is None else None,field_dft_oversampling=8,grid_construction=args.grid_construction,canonical_grid_faces=args.canonical_grid_faces,face_roundoff=model['face_roundoff'],
        grid_sha256=hashlib.sha256(b''.join(values.tobytes() for values in model['grid'])).hexdigest(),
        current_edges_sha256=hashlib.sha256(repr(edges).encode()).hexdigest(),
        source_helper_sha256=hashlib.sha256(Path(compile_known_current_edges.__code__.co_filename).read_bytes()).hexdigest())
    manifest=args.output/'run-input.json'
    if args.reuse_raw:
        if not manifest.exists() or json.loads(manifest.read_text())!={'inputs':run_input,'forward_complete':True}:
            raise ValueError('Retained independent raw data must have a complete matching source/material/grid/spectral input manifest.')
    elif raw.exists() and any(raw.iterdir()):
        raise FileExistsError('Use a fresh independent output directory; retained raw fields require explicit matching reuse.')
    csx.Write2XML(str(args.output/'physical-scene.xml'))
    if not args.reuse_raw:
        manifest.write_text(json.dumps(dict(inputs=run_input,forward_complete=False),indent=2)+'\n')
        raw.mkdir(exist_ok=True)
        external.Run(str(raw),cleanup=False,verbose=0,numThreads=args.threads,
            debug_material=True,debug_operator=args.material_only,setup_only=args.material_only)
        manifest.write_text(json.dumps(dict(inputs=run_input,forward_complete=not args.material_only),indent=2)+'\n')
    material_dump=next(raw.glob('material_dump*.vts'))
    material_report=save_native_material_arrays(material_dump,args.output/'native-materials.npz')
    binding_after=reference_binding(args.reference_source)
    if binding_before!=binding_after:raise RuntimeError('Independent controller/helper/backend changed during execution; retained raw data is diagnostic only.')
    if args.material_only:
        (args.output/'material-setup.json').write_text(json.dumps(dict(inputs=run_input,native_materials=material_report,source_binding_before=binding_before,source_binding_after=binding_after),indent=2)+'\n')
        print(json.dumps(material_report),flush=True)
        return material_report
    if port is not None:port.CalcPort(str(raw),[frequency],ref_impedance=50.)
    voltage=0j
    for name,weight in probes:
        data=np.loadtxt(raw/name,comments='%')
        voltage+=weight*DFT_time2freq(data[:,0],data[:,1],[frequency],signal_type='pulse')[0]
    if port is None:
        source=np.loadtxt(raw/'et')
        current_native=complex(known_current_spectrum(source[:,0],source[:,1],[frequency],dt=dt,amplitude=args.current_amplitude)[0])
    else:current_native=np.conj(complex(port.if_tot[0]))
    voltage_native=np.conj(voltage);impedance=voltage_native/current_native
    theta=np.linspace(0,np.pi,args.theta_points);phi=np.linspace(0,2*np.pi,2*args.theta_points-1)
    radiation=surface.CalcNF2FF(str(raw),[frequency],np.rad2deg(theta),np.rad2deg(phi),radius=12.,
        center=[model['radius'],-math.pi/2,0.],read_cached=False,verbose=0)
    accepted=float(.5*np.real(voltage_native*np.conj(current_native)))
    report=dict(schema='independent-cylindrical-bent-pcb-reference/1',spacing_m=args.spacing,
        curvature_m_inverse=args.curvature,frequency_hz=frequency,duration_s=duration,
        dt_s=dt,steps=math.ceil(duration/dt),elapsed_s=time.perf_counter()-start,
        independent_coordinates='physical cylindrical (r,phi,z), r=1/kappa-u, phi=kappa*s-pi/2; physical world center=(0,1/kappa,0)',
        independent_source=('known impressed radial current, clipped physical aperture, independently derived constitutive soft-E coefficients, sine Gaussian pulse; no local resistor or fitted amplitude' if port is None else 'legacy physical radial50ohm distributed lumped aperture; Gaussian pulse; distinct finite-aperture source/load operator'),
        metal_model='vacuum' if args.vacuum else args.metal,metal_scope=('vacuum current-only control' if args.vacuum else 'PEC comparison only, not finite copper' if args.metal=='pec' else 'independent fitted single-trace conducting sheet; not a two-sided copper heat oracle'),
        source_identity=_source_identity(args.reference_source),module_file=reference_package.__file__,
        execution_driver_sha256=DRIVER_SHA256,analysis_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        theta=theta.tolist(),phi=phi.tolist(),physical_aperture_voltage=_serial(np.conj(voltage)),
        default_port_voltage=None if port is None else _serial(np.conj(port.uf_tot[0])),current=_serial(current_native),impedance=_serial(impedance),
        accepted_spectral_w_s2=accepted,radiated_spectral_w_s2=float(radiation.Prad[0]),
        directivity=float(radiation.Dmax[0]),E_theta=_serial(np.conj(radiation.E_theta[0])),E_phi=_serial(np.conj(radiation.E_phi[0])),
        phase_center_world_m=[0.,0.,0.],reference_phase_conversion='reference exp(+iwt) spectra are conjugated to native exp(-iwt); no fitted phase',
        reference_phase_center_cylindrical=[model['radius'],-math.pi/2,0.],
        independent_field_dft_oversampling=8,
        raw_input_manifest=run_input,
        native_materials=material_report,source_binding_before=binding_before,source_binding_after=binding_after,
        qualification='raw independent physical control; spatial/PML/aperture convergence and native comparison pending')
    if port is None:
        report.update(known_current_peak_a=args.current_amplitude,source_clock='et f[n] represents current at (n+.5)*dt; authoritative dt and +dt/2 midpoint phase',
            voltage_polarity='positive integral Er from r=R to R-gap, identical to positive integral Eu from u=0 to u=gap',
            source_edge_count=len(edges),source_normalized_radial_flux=[sum(edge.current_density_per_amp*edge.dual_area for edge in edges if edge.index[0]==index) for index in sorted({edge.index[0] for edge in edges})])
    (args.output/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('metal_model','spacing_m','elapsed_s','impedance','directivity','accepted_spectral_w_s2','radiated_spectral_w_s2','qualification')}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--reference-source',type=Path,required=True)
    parser.add_argument('--spacing',type=float,default=.01);parser.add_argument('--curvature',type=float,default=3.)
    parser.add_argument('--metal',choices=('sheet','pec'),default='sheet');parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--source',choices=('matched-current','distributed-port'),default='matched-current')
    parser.add_argument('--current-amplitude',type=float,default=1e-3)
    parser.add_argument('--vacuum',action='store_true')
    parser.add_argument('--material-only',action='store_true')
    parser.add_argument('--canonical-grid-faces',action='store_true')
    parser.add_argument('--grid-construction',choices=('padded_linspace','domain_then_pml'),default='padded_linspace')
    parser.add_argument('--duration',type=float,default=96e-9);parser.add_argument('--theta-points',type=int,default=25)
    parser.add_argument('--reuse-raw',action='store_true');args=parser.parse_args()
    args.output=args.output.resolve();args.reference_source=args.reference_source.resolve();run(args)
