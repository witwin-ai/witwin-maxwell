"""Actual CUDA FDTD -> EMFC files -> decoded Yee fields and exterior reconstructions."""
import argparse
import json
from pathlib import Path
import torch
import witwin.maxwell as mw
import emec
import emfc

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('/tmp/maxwell-emfc-validation'))
parser.add_argument('--steps', type=int, default=2400)
parser.add_argument('--compare', type=Path)
args = parser.parse_args()
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)

def errors(ref, got, masks=None):
    answer = []
    for c, (a, b) in enumerate(zip(ref, got)):
        if masks is not None:
            a, b = a[masks[c]], b[masks[c]]
        a, b = a.to(torch.complex128), b.to(torch.complex128)
        answer.append(float((a-b).norm()/a.norm()) if a.norm() else float((a-b).norm()))
    return answer


def outside_masks(region, kind):
    # Independent node support interpretation: all adjacent cells must be inside.
    masks = []
    for c in range(3):
        inside = region
        for axis in range(3):
            if (axis != c) == (kind == 'electric'):
                shape = list(inside.shape)
                shape[axis] += 1
                extended = torch.zeros(shape, dtype=torch.bool, device=inside.device)
                dst, lo, hi = [slice(None)]*3, [slice(None)]*3, [slice(None)]*3
                dst[axis], lo[axis], hi[axis] = slice(1,-1), slice(None,-1), slice(1,None)
                extended[tuple(dst)] = inside[tuple(lo)] & inside[tuple(hi)]
                inside = extended
        masks.append(~inside)
    return masks


def run(name, eps, mu):
    h, f, n, pml = .015, 1e9, 24, 10
    scene = mw.Scene(domain=mw.Domain(bounds=((-n*h/2,n*h/2),)*3),
        grid=mw.GridSpec.uniform(h), boundary=mw.BoundarySpec.pml(num_layers=pml), device='cuda')
    if eps != 1 or mu != 1:
        scene.add_structure(mw.Structure(geometry=mw.Box(position=(.03,0,0), size=(.06,.06,.06)),
                                        material=mw.Material(eps_r=eps, mu_r=mu)))
    scene.add_source(mw.PointDipole(position=(-.045,0,0), width=.015,
        polarization=(.3,.5,1.), source_time=mw.GaussianPulse(frequency=f, fwidth=.5e9)))
    result = mw.Simulation.fdtd(scene, frequency=f, run_time=mw.TimeConfig(time_steps=args.steps),
        spectral_sampler=mw.SpectralSampler(window='none', normalize_source=True),
        full_field_dft=True).run()
    arrays = []
    for c, a in enumerate((result.E.x,result.E.y,result.E.z)):
        arrays.append(a[tuple(slice(pml,pml+n+(axis!=c)) for axis in range(3))].contiguous())
    nodes = tuple(torch.as_tensor(getattr(result.prepared_scene, axis+'_nodes64')[pml:pml+n+1], device='cuda') / h
                  for axis in ('x','y','z'))
    field = emfc.StaggeredData(tuple(arrays), nodes, reference_spacing_m=h)
    original = field.to_emec()
    kappa2 = emec.kappa_squared(f, h, dt=result.solver.dt)
    row = dict(case=name, frequency_hz=f, dt=result.solver.dt, steps=args.steps, cells=n,
               kappa2=kappa2, raw_bytes=sum(a.numel()*a.element_size() for a in arrays))
    if args.compare:
        previous = emfc.load_staggered(args.compare / f'{name}-exact.emfc')
        row['duration_change_errors'] = errors(previous.components, arrays)
    for label, tolerance in [('exact',None),('lossy',.01)]:
        path = OUT / f'{name}-{label}.emfc'
        emfc.save_staggered(path, field, tolerance=tolerance)
        decoded = emfc.load_staggered(path)
        e = errors(arrays,decoded.components)
        assert max(e) <= (tolerance or 0)
        assert all(torch.equal(a,b) for a,b in zip(nodes,decoded.nodes))
        assert decoded.reference_spacing_m == h and decoded.quantity == 'field'
        row[label] = dict(errors=e,bytes=path.stat().st_size)
    current = emec.volume_current_staggered(original, kappa2)
    path = OUT / f'{name}-volume.emfc'
    emfc.save_staggered(path, emfc.StaggeredData.from_emec(current, reference_spacing_m=h))
    decoded_current = emfc.load_staggered(path)
    assert decoded_current.quantity == 'volume_current'
    assert max(errors(current.components,decoded_current.components)) == 0
    operator = emec.StaggeredOperator(original.grid,kappa2,device='cuda')
    rebuilt = operator.solve(decoded_current.to_emec())
    row['volume'] = dict(current_file_errors=errors(current.components,decoded_current.components),
                         field_errors=errors(arrays,rebuilt.components),bytes=path.stat().st_size)
    region = torch.zeros((n,n,n),device='cuda',dtype=torch.bool)
    region[5:19,5:19,5:19] = True
    masks = outside_masks(region,'electric')
    pure = operator.solve(emec.commutator_density(original,region,side='exterior'))
    row['pure_surface_errors'] = errors(arrays,pure.components,masks)
    path = OUT / f'{name}-surface-no-weak.emfc'
    emfc.save_surface(path, field, region, kappa2=kappa2, tolerance=.01, side='exterior')
    plain = emfc.load_surface(path)
    plain_errors = errors(arrays, plain.reconstruct().components, masks)
    assert max(plain_errors) <= .01
    row['surface_no_weak'] = dict(errors=plain_errors, correction_mode=plain.correction_mode,
                                  bytes=path.stat().st_size)
    path = OUT / f'{name}-surface.emfc'
    emfc.save_surface(path, field, region, kappa2=kappa2, tolerance=.01,
                      preserve_weak_fields=True, weak_tolerance=.01, side='exterior')
    surface = emfc.load_surface(path)
    reconstructed = surface.reconstruct()
    e = errors(arrays,reconstructed.components,masks)
    assert max(e) <= .01
    band_errors = []
    for c in range(3):
        ref = arrays[c][masks[c]].to(torch.complex128)
        got = reconstructed.components[c][masks[c]].to(torch.complex128)
        peak = ref.abs().max()
        bands = []
        for low in (-100,-80,-60,-40):
            select = (ref.abs() >= peak*10**(low/20)) & (ref.abs() < peak*10**((low+20)/20))
            bands.append(float((got[select]-ref[select]).norm()/ref[select].norm()) if select.any() else None)
        assert all(v is None or v <= .01 for v in bands)
        band_errors.append(bands)
    row['surface'] = dict(errors=e,weak_band_errors=band_errors,
        correction_mode=surface.correction_mode,bytes=path.stat().st_size)
    print(json.dumps(row),flush=True)
    return row

report = dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),emfc_source=emfc.__file__,
              emec_source=emec.__file__,cases=[])
for case in [('vacuum',1.,1.),('dielectric',3.,1.),('magnetic',2.,1.5)]:
    report['cases'].append(run(*case))
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
