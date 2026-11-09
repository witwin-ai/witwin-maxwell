import json
from pathlib import Path
import torch,emec,emfc
f=emfc.load_staggered('/tmp/maxwell-emfc-cause/sine_pml40.emfc').to_emec()
k=.09877495267586907
region=torch.zeros(f.grid.cells,dtype=torch.bool,device='cuda');region[5:19,5:19,5:19]=True
rho=emec.commutator_density(f,region,side='exterior')
rows=[]
for pml,sigma in ((8,2),(16,2),(24,2),(32,2),(24,4)):
    op=emec.StaggeredOperator(f.grid,k,pml=pml,sigma_max=sigma,device='cuda')
    try:
        out=op.solve(rho)
    except RuntimeError as error:
        rows.append(dict(pml=pml,sigma=sigma,error=str(error)))
        continue
    errors=[]
    for c in range(3):
        m=~emec.region_indicator(region,f.grid,'electric',c)
        a=f.components[c][m].to(torch.complex128)
        errors.append(float((a-out.components[c][m]).norm()/a.norm()))
    rows.append(dict(pml=pml,sigma=sigma,errors=errors))
print(json.dumps(rows))
Path('/tmp/maxwell-emfc-cause/remaining.json').write_text(json.dumps(rows,indent=2)+'\n')
