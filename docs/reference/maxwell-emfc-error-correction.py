import json
from pathlib import Path
import torch,emfc
rows=[]
for name in ('vacuum','dielectric','magnetic'):
    data=emfc.load_staggered('/tmp/maxwell-emfc-validation/long/'+name+'-exact.emfc')
    region=torch.zeros(data.cells,dtype=torch.bool,device='cuda');region[5:19,5:19,5:19]=True
    path=Path('/tmp/maxwell-emfc-cause/'+name+'-tighter-correction.emfc')
    emfc.save_surface(path,data,region,kappa2=.09877495267586907,tolerance=.001,preserve_weak_fields=True,weak_tolerance=.01)
    surface=emfc.load_surface(path)
    row=dict(case=name,correction_mode=surface.correction_mode,bytes=path.stat().st_size,certificate=surface.certificate)
    rows.append(row);print(json.dumps(row),flush=True)
Path('/tmp/maxwell-emfc-cause/correction.json').write_text(json.dumps(rows,indent=2)+'\n')
