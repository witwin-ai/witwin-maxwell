import json
from pathlib import Path
import torch,emec,emfc
import witwin.maxwell as mw

torch.set_num_threads(1)
rows=[]
for label,width,kind,pml in [('cosine_pulse',.5e9,'pulse',10),('sine_pulse',.5e9,'pulse',10)]:
    h=.015;n=24;f=1e9
    scene=mw.Scene(domain=mw.Domain(bounds=((-n*h/2,n*h/2),)*3),grid=mw.GridSpec.uniform(h),boundary=mw.BoundarySpec.pml(num_layers=pml),device='cuda')
    time=mw.GaussianPulse(frequency=f,fwidth=width,phase=1.5707963267948966 if label=='sine_pulse' else 0.) if kind=='pulse' else mw.CW(frequency=f)
    scene.add_source(mw.PointDipole(position=(-.045,0,0),width=.015,polarization=(.3,.5,1),source_time=time))
    result=mw.Simulation.fdtd(scene,frequency=f,run_time=mw.TimeConfig(time_steps=4800),spectral_sampler=mw.SpectralSampler(window='hanning' if kind=='cw' else 'none',normalize_source=True),full_field_dft=True).run()
    arrays=tuple(a[tuple(slice(pml,pml+n+(axis!=c)) for axis in range(3))].contiguous() for c,a in enumerate((result.E.x,result.E.y,result.E.z)))
    nodes=tuple(torch.as_tensor(getattr(result.prepared_scene,axis+'_nodes64')[pml:pml+n+1],device='cuda')/h for axis in ('x','y','z'))
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
    end=tuple(getattr(result.solver,cname)[tuple(slice(pml,pml+n+(axis!=c)) for axis in range(3))].to(torch.complex128).contiguous() for c,cname in enumerate(('Ex','Ey','Ez')))
    emfc.save_staggered('/tmp/maxwell-emfc-cause/'+label+'-end.emfc',emfc.StaggeredData(tuple(a.to(torch.complex64) for a in end),nodes,reference_spacing_m=h))
    double=emec.StaggeredField(tuple(a.to(torch.complex128) for a in arrays),field.grid)
    current=emec.volume_current_staggered(double,k)
    lapend=emec.laplacian_staggered(emec.StaggeredField(end,field.grid))
    masks=[]
    for c in range(3):
        mask=~emec.region_indicator(region,field.grid,'electric',c)
        interior=torch.zeros_like(mask);interior[2:-2,2:-2,2:-2]=True;masks.append(mask&interior)
    numerator=sum((e[m].conj()*v[m]).sum() for e,v,m in zip(end,current.components,masks))
    denominator=sum(e[m].abs().square().sum() for e,m in zip(end,masks))
    alpha=numerator/denominator
    row['endpoint_fit_alpha']=[float(alpha.real),float(alpha.imag)]
    row['endpoint_fit_relative_residual']=[float((v[m]-alpha*e[m]).norm()/v[m].norm()) for v,e,m in zip(current.components,end,masks)]
    row['endpoint_laplacian_relative']=[float(l[m].norm()/(k*e[m]).norm()) for l,e,m in zip(lapend.components,end,masks)]
    corrected=emec.StaggeredField(tuple(a-alpha/k*e for a,e in zip(double.components,end)),field.grid)
    predicted=op.solve(emec.commutator_density(corrected,region,side='exterior'))
    row['corrected_pure_surface']=[]
    for c in range(3):
        m=~emec.region_indicator(region,field.grid,'electric',c)
        row['corrected_pure_surface'].append(float((corrected.components[c][m]-predicted.components[c][m]).norm()/corrected.components[c][m].norm()))
    rows.append(row);print(json.dumps(row),flush=True)
Path('/tmp/maxwell-emfc-cause/endpoint.json').write_text(json.dumps(rows,indent=2)+'\n')
