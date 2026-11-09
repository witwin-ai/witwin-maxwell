import json
from pathlib import Path
import torch,emec,emfc
rows=[]
for name in ('vacuum','dielectric','magnetic'):
    f=emfc.load_staggered('/tmp/maxwell-emfc-validation/long/'+name+'-exact.emfc').to_emec()
    f=emec.StaggeredField(tuple(a.to(torch.complex128) for a in f.components),f.grid)
    k=.09877495267586907
    cc=emec.curl_curl_staggered(f)
    longitudinal=emec.longitudinal_current_staggered(f)
    region=torch.zeros(f.grid.cells,dtype=torch.bool,device='cuda');region[5:19,5:19,5:19]=True
    row=dict(case=name,components=[])
    for c in range(3):
        mask=~emec.region_indicator(region,f.grid,'electric',c)
        inner=torch.zeros_like(mask);inner[2:-2,2:-2,2:-2]=True;mask &= inner
        e=f.components[c][mask];l=longitudinal.components[c][mask];t=(k*f.components[c]-cc.components[c])[mask]
        row['components'].append(dict(longitudinal_norm=float(l.norm()/(k*e).norm()),transverse_norm=float(t.norm()/(k*e).norm()),volume_norm=float((l+t).norm()/(k*e).norm())))
    rows.append(row)
print(json.dumps(rows))
Path('/tmp/maxwell-emfc-cause/divergence.json').write_text(json.dumps(rows,indent=2)+'\n')
