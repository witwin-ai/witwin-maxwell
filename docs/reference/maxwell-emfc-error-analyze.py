import json
from pathlib import Path
import torch
import emec
import emfc

torch.set_num_threads(1)
root=Path('/tmp/maxwell-emfc-validation/long')
report=json.loads((root/'report.json').read_text())
rows=[]
def norms(ref,got,masks):
    return [float((a[m]-b[m]).norm()/a[m].norm()) for a,b,m in zip(ref,got,masks)]
def pack(parts,f,q='field'):
    return emec.StaggeredField(tuple(parts),f.grid,f.kind,q)
for case in report['cases']:
    data=emfc.load_staggered(root/(case['case']+'-exact.emfc'))
    f=pack([a.to(torch.complex128) for a in data.components],data.to_emec())
    k=case['kappa2']
    region=torch.zeros(f.grid.cells,dtype=torch.bool,device='cuda');region[5:19,5:19,5:19]=True
    ext=[~emec.region_indicator(region,f.grid,f.kind,c) for c in range(3)]
    op=emec.StaggeredOperator(f.grid,k,device='cuda')
    v=emec.volume_current_staggered(f,k)
    row=dict(case=case['case'],min_denominator=[op.min_denominator(c) for c in range(3)])
    lap=emec.laplacian_staggered(f)
    fit=[]
    for guard in (1,2,3):
        masks=[]
        for c in range(3):
            m=ext[c].clone();inner=torch.zeros_like(m);inner[guard:-guard,guard:-guard,guard:-guard]=True
            masks.append(m&inner)
        num=sum((a[m].conj()*l[m]).sum() for a,l,m in zip(f.components,lap.components,masks))
        den=sum(a[m].abs().square().sum() for a,m in zip(f.components,masks))
        kh=-num/den
        res=norms([k*a for a in f.components],[-l for l in lap.components],masks)
        fit.append(dict(guard=guard,fitted_kappa2=[float(kh.real),float(kh.imag)],relative_residual=res))
    row['local_operator']=fit
    for label,mask,side in [('exterior',region,'exterior'),('annulus',None,'interior')]:
        if mask is None:
            mask=torch.zeros_like(region);mask[2:22,2:22,2:22]=True;mask[5:19,5:19,5:19]=False
        kept=[emec.region_indicator(mask,f.grid,f.kind,c) for c in range(3)]
        if side=='exterior':kept=[~m for m in kept]
        target=pack([a*m for a,m in zip(f.components,kept)],f)
        density=emec.commutator_density(f,mask,side=side)
        predicted=op.solve(density)
        kept_v=pack([a*m for a,m in zip(v.components,kept)],f,'volume_current')
        source_contribution=op.solve(kept_v)
        full=op.solve(emec.volume_current_staggered(target,k))
        sumfield=[a+b for a,b in zip(predicted.components,source_contribution.components)]
        row[label]=dict(pure_error=norms(target.components,predicted.components,kept),
            with_kept_current_error=norms(target.components,sumfield,kept),
            direct_target_current_error=norms(target.components,full.components,kept),
            identity_error=norms(target.components,[a-b for a,b in zip(sumfield,full.components)],kept))
        row[label]['identity_error']=[float((a[m]-b[m]).norm()/r[m].norm()) for a,b,r,m in zip(sumfield,full.components,target.components,kept)]
        row[label]['omitted_current_field_relative_norm']=[float(a[m].norm()/r[m].norm()) for a,r,m in zip(source_contribution.components,target.components,kept)]
    for label,scale in [('continuum',emec.kappa_squared(1e9,.015)),('fit',fit[1]['fitted_kappa2'][0])]:
        oper=emec.StaggeredOperator(f.grid,scale,device='cuda')
        pred=oper.solve(emec.commutator_density(f,region,side='exterior'))
        row[label]=dict(kappa2=scale,pure_surface_errors=norms(f.components,pred.components,ext))
    plain=emfc.load_surface(root/(case['case']+'-surface-no-weak.emfc'))
    reconstructed=plain.reconstruct()
    weak=[]
    for c in range(3):
        a=f.components[c][ext[c]];b=reconstructed.components[c][ext[c]].to(torch.complex128)
        peak=a.abs().max();bands=[]
        for low in (-100,-80,-60,-40):
            m=(a.abs()>=peak*10**(low/20))&(a.abs()<peak*10**((low+20)/20))
            bands.append(dict(low_db=low,count=int(m.sum()),error=float((a[m]-b[m]).norm()/a[m].norm()) if m.any() else None))
        weak.append(bands)
    row['additive_weak_bands']=weak
    rows.append(row)
    print(json.dumps(row),flush=True)
Path('/tmp/maxwell-emfc-cause/analysis.json').write_text(json.dumps(rows,indent=2)+'\n')
