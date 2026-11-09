"""Audit actual independent Yee coefficients and hard PEC edge ownership."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.dev.fdtd.reference_material_dump import read_native_material_dump


def compare_operator(native, reference, *, pml_layers):
    """Compare exact constrained edges after excluding the declared PML strips.

    Native component arrays retain their own stagger. The reference attaches
    component-specific operator samples to a common plotting grid, so only its
    final own-axis slot is discarded. No nearest-neighbor field alignment or
    amplitude normalization is performed.
    """
    if isinstance(pml_layers, bool) or int(pml_layers) != pml_layers or pml_layers < 0:
        raise ValueError('PML layer count must be a nonnegative integer.')
    rows=[]
    for component,(axis,prefix) in enumerate(zip('xyz',('cex','cey','cez'))):
        decay=np.asarray(native[prefix+'_decay']);curl=np.asarray(native[prefix+'_curl'])
        if decay.shape != curl.shape or decay.ndim != 3:
            raise ValueError('Native decay and curl must have the same three-dimensional Yee shape.')
        shape=list(reference['vv'].shape[1:]);shape[component]-=1
        if tuple(shape) != decay.shape or reference['vi'].shape != reference['vv'].shape:
            raise ValueError('Independent common-grid operator shape does not match the native Yee stagger.')
        trimmed=tuple(slice(0,n) for n in decay.shape)
        vv=reference['vv'][component][trimmed];vi=reference['vi'][component][trimmed]
        native_pec=(decay==0)&(curl==0);reference_pec=(vv==0)&(vi==0)
        interior=np.zeros_like(native_pec)
        if pml_layers and min(decay.shape) <= 2*pml_layers:
            raise ValueError('PML exclusion removes the entire operator interior.')
        region=tuple(slice(pml_layers,-pml_layers) if pml_layers else slice(None) for _ in range(3))
        interior[region]=True;mismatch=(native_pec!=reference_pec)&interior
        rows.append(dict(component='E'+axis,native_constrained_edges=int((native_pec&interior).sum()),
            reference_constrained_edges=int((reference_pec&interior).sum()),
            different_edges=int(mismatch.sum()),different_indices=np.argwhere(mismatch).tolist()))
    return dict(components=rows,identical_pec_support=all(row['different_edges']==0 for row in rows),
        scope='Exact hard-PEC edge support only; no continuum, dielectric or antenna accuracy claim.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--pml-layers',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    with np.load(args.native) as native:
        result=compare_operator(native,read_native_material_dump(args.reference)['point_data'],pml_layers=args.pml_layers)
    result['inputs']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.native,args.reference)}
    result['analysis_driver_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(identical_pec_support=result['identical_pec_support'],different_edges=sum(row['different_edges'] for row in result['components']))))


if __name__=='__main__':main()
