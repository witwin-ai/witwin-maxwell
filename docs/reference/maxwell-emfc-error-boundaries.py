import json
from pathlib import Path
import torch,emec,emfc
import witwin.maxwell as mw

torch.set_num_threads(1)
rows=[]
for label,width,kind,pml in [('sine_pml20',.5e9,'pulse',20),('sine_pml40',.5e9,'pulse',40),('sine_domain48',.5e9,'pulse',10)]:
    h=.015;n=24;f=1e9
    domain_n=48 if label=='sine_domain48' else 24
    offset=pml+(domain_n-n)//2
    scene=mw.Scene(domain=mw.Domain(bounds=((-domain_n*h/2,domain_n*h/2),)*3),grid=mw.GridSpec.uniform(h),boundary=mw.BoundarySpec.pml(num_layers=pml),device='cuda')
    time=mw.GaussianPulse(frequency=f,fwidth=width,phase=1.5707963267948966) if kind=='pulse' else mw.CW(frequency=f)
    scene.add_source(mw.PointDipole(position=(-.045,0,0),width=.015,polarization=(.3,.5,1),source_time=time))
    result=mw.Simulation.fdtd(scene,frequency=f,run_time=mw.TimeConfig(time_steps=4800),spectral_sampler=mw.SpectralSampler(window='hanning' if kind=='cw' else 'none',normalize_source=True),full_field_dft=True).run()
    arrays=tuple(a[tuple(slice(offset,offset+n+(axis!=c)) for axis in range(3))].contiguous() for c,a in enumerate((result.E.x,result.E.y,result.E.z)))
    nodes=tuple(torch.as_tensor(getattr(result.prepared_scene,axis+'_nodes64')[offset:offset+n+1],device='cuda')/h for axis in ('x','y','z'))
    data=emfc.StaggeredData(arrays,nodes,reference_spacing_m=h)
    emfc.save_staggered('/tmp/maxwell-emfc-cause/'+label+'.emfc',data)
    field=data.to_emec();k=emec.kappa_squared(f,h,dt=result.solver.dt)
    region=torch.zeros(field.grid.cells,dtype=torch.bool,device='cuda');region[5:19,5:19,5:19]=True
    op=emec.StaggeredOperator(field.grid,k,device='cuda')
    vol=op.solve(emec.volume_current_staggered(field,k));sur=op.solve(emec.commutator_density(field,region,side='exterior'))
    row=dict(label=label,dt=result.solver.dt,start_steps=result.solver.dft_start_steps,source_free=emec.source_free_residual(field,k,region),volume=[],surface=[],end_energy=sum(float(getattr(result.solver,c).square().sum()) for c in ('Ex','Ey','Ez')))
    for c in range(3):
        m=~emec.region_indicator(region,field.grid,'electric',c)
        a=field.components[c][m].to(torch.complex128)
        for name,out in [('volume',vol),('surface',sur)]:row[name].append(float((a-out.components[c][m]).norm()/a.norm()))
    rows.append(row);print(json.dumps(row),flush=True)
Path('/tmp/maxwell-emfc-cause/boundaries.json').write_text(json.dumps(rows,indent=2)+'\n')
