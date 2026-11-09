"""Measure exterior reconstruction sensitivity from existing validation files."""
import argparse
import json
from pathlib import Path
import torch
import emec
import emfc
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
root = parser.parse_args().directory
report = json.loads((root/'report.json').read_text())
rows = []
for case in report['cases']:
    f = emfc.load_staggered(root / (case['case']+'-exact.emfc')).to_emec()
    region = torch.zeros(f.grid.cells, dtype=torch.bool, device='cuda')
    region[5:19,5:19,5:19] = True
    source = emfc.load_staggered(root / (case['case']+'-volume.emfc')).to_emec()
    row = dict(case=case['case'],source_free_residual=emec.source_free_residual(f,case['kappa2'],region),pml_sweep=[])
    for pml in (8,16,24):
        op = emec.StaggeredOperator(f.grid,case['kappa2'],pml=pml,device='cuda')
        volume = op.solve(source)
        surface = op.solve(emec.commutator_density(f,region,side='exterior'))
        values = dict(pml=pml,volume_exterior=[],surface_exterior=[])
        for c in range(3):
            mask = ~emec.region_indicator(region,f.grid,'electric',c)
            ref = f.components[c][mask].to(torch.complex128)
            for name, field in [('volume_exterior',volume),('surface_exterior',surface)]:
                values[name].append(float((field.components[c][mask]-ref).norm()/ref.norm()))
        row['pml_sweep'].append(values)
    rows.append(row)
(root/'diagnostics.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows))
