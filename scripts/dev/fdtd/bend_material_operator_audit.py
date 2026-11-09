"""Compare independent actual cylindrical material and PEC operators on Yee edges."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import witwin.maxwell as mw
from witwin.maxwell.scene import PreparedScene
from scripts.dev.fdtd.pcb_acceptance import package_identity
from scripts.dev.fdtd.reference_material_dump import read_native_material_dump
from scripts.dev.fdtd.bend_pcb_external_reference import fixture
from scripts.dev.fdtd.matched_current_reference import compile_known_current_edges


def reference_to_chart_edges(values,component):
    """Trim the unused native operator edge and permute Er,Ephi,Ez to Es,Eu,Ez."""
    external_component=(1,0,2)[component]
    selectors=[slice(None)]*3
    selectors[external_component]=slice(None,-1)
    return values[external_component][tuple(selectors)][::-1,:,:].transpose(1,0,2).copy()


def native_scene(spacing):
    structures=[]
    for name,u,size in (('ground',0.,.14),('patch',.02,.08)):
        vertices=[(a*size/2,b*size/2) for a,b in ((-1,-1),(1,-1),(1,1),(-1,1))]
        structures.append(mw.Structure(name=name,geometry=mw.PlanarPEC(vertices,normal_axis='y',coordinate=u),material=mw.Material.pec()))
    structures.append(mw.Structure(name='substrate',geometry=mw.Box(
        position=torch.tensor((0.,.01,0.),dtype=torch.float64),size=torch.tensor((.14,.02,.14),dtype=torch.float64)),
        material=mw.Material(eps_r=2.2,sigma_e=.001)))
    return mw.Scene(domain=mw.Domain(bounds=((-0.12,.12),(-.1,.1),(-.12,.12))),grid=mw.GridSpec.uniform(spacing),
        boundary=mw.BoundarySpec.pml(num_layers=round(.08/spacing)),coordinates=mw.BendCoordinates(curvature=3.),structures=structures,device='cpu')


def interior_mask(scene,component):
    shape=[len(getattr(scene,axis+('_half64' if d==component else '_nodes64')))for d,axis in enumerate('xyz')]
    mask=np.ones(shape,bool)
    coordinates=[]
    for d,(axis,(lo,hi))in enumerate(zip('xyz',((-.12,.12),(-.1,.1),(-.12,.12)))):
        values=np.asarray(getattr(scene,axis+('_half64'if component==d else '_nodes64')))
        coordinates.append(values)
        axis_shape=[1,1,1];axis_shape[d]=len(values)
        mask &= ((values>lo+1e-10)&(values<hi-1e-10)).reshape(axis_shape)
    return mask,coordinates


def run(material_dump,operator_dump,output,*,spacing=.01,input_manifest=None):
    root=Path(__file__).resolve().parents[3]
    if not (root/'SOURCE_BASE_COMMIT').exists() or not Path(mw.__file__).resolve().is_relative_to(root):
        raise RuntimeError('Actual operator audits require an immutable source archive and its native package.')
    paths=(Path(__file__).resolve(),Path(read_native_material_dump.__code__.co_filename).resolve(),
        Path(fixture.__code__.co_filename).resolve(),Path(compile_known_current_edges.__code__.co_filename).resolve())
    if input_manifest is not None:paths=(*paths,input_manifest)
    def binding():
        return dict(base_commit=(root/'SOURCE_BASE_COMMIT').read_text().strip(),native=package_identity(mw,'witwin-maxwell')['source_sha256'],
            files={str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in (*paths,material_dump,operator_dump)})
    before=binding()
    materials=read_native_material_dump(material_dump)['point_data']
    operator=read_native_material_dump(operator_dump)['point_data']
    pec_reference=(operator['vv']==0)&(operator['vi']==0)
    report={'schema':'independent-cylindrical-native-operator-audit/1','spacing_m':spacing,'pec':{},'material':{},'source_binding_before':before}
    if input_manifest is not None:
        inputs=json.loads(input_manifest.read_text())['inputs']
        if inputs['source']!='matched-current':raise ValueError('Source operator audit requires the declared known-current source.')
        model=fixture(inputs['spacing'],inputs['curvature'],grid_construction=inputs['grid_construction'],canonical_grid_faces=inputs['canonical_grid_faces'])
        lower,upper=model['aperture'];relative_epsilon=1. if inputs['vacuum'] else 2.2
        conductivity=0. if inputs['vacuum'] else .001
        edges=compile_known_current_edges(model['grid'],axis=0,path=(upper[0],lower[0]),
            aperture_bounds={1:(lower[1],upper[1]),2:(lower[2],upper[2])},dt=inputs['dt'],
            epsilon=relative_epsilon*8.8541878128e-12,conductivity=conductivity,cylindrical_radial=True)
        expected=np.zeros_like(operator['exc_volt'],dtype=np.float64)
        for edge in edges:
            expected[(edge.axis,*edge.index)]=edge.soft_e_per_amp*inputs['current_amplitude']*edge.edge_length
        support=expected!=0;actual=operator['exc_volt'];actual_support=actual!=0
        report['known_current_operator']=dict(declared_edge_count=int(support.sum()),actual_edge_count=int(actual_support.sum()),
            missing_edges=int((support&~actual_support).sum()),extra_edges=int((~support&actual_support).sum()),
            maximum_relative_edge_voltage_error=float(np.max(abs((actual-expected)[support]/expected[support]))),
            maximum_feed_relative_epsilon_error=float(np.max(abs(materials['epsilon'][support]-relative_epsilon))),
            maximum_feed_conductivity_error=float(np.max(abs(materials['kappa'][support]-conductivity))),
            contract='Independent soft-E per ampere times declared peak current and physical edge length equals the actual stored-voltage excitation coefficient; no fitted scale.')
    base=native_scene(spacing)
    for label,spec in (('point',None),('quarter',mw.SubpixelSpec(samples=(2,2,2),averaging='arithmetic')),
                       ('normal64',mw.SubpixelSpec(samples=(1,64,1),averaging='polarized'))):
        scene=PreparedScene(base if spec is None else base.clone(subpixel_samples=spec))
        model=scene.compile_materials();coefficient_report={}
        for component,axis in enumerate('xyz'):
            name='E'+axis;mask,coordinates=interior_mask(scene,component)
            native_pec=model['pec_sheet_edges'][name].numpy().astype(bool)
            external_pec=reference_to_chart_edges(pec_reference,component)
            if label=='point':
                native_only=np.argwhere(native_pec&~external_pec&mask)
                reference_only=np.argwhere(~native_pec&external_pec&mask)
                report['pec'][name]=dict(native_count=int((native_pec&mask).sum()),reference_count=int((external_pec&mask).sum()),
                    native_only=[tuple(float(coordinates[d][i[d]])for d in range(3))for i in native_only],
                    reference_only=[tuple(float(coordinates[d][i[d]])for d in range(3))for i in reference_only])
            mask &= ~(native_pec|external_pec)
            u=coordinates[1];h=1-3*u;metric=(1/h if component==0 else h)[None,:,None]
            for key,reference_key,tolerance in (('eps','epsilon',2e-6),('sigma_e','kappa',2e-9)):
                actual=model['edge_components'][key][name].numpy()
                reference=reference_to_chart_edges(materials[reference_key],component)*metric
                if actual.shape!=reference.shape:raise ValueError('Independent/native component shapes differ.')
                differences=(actual-reference)[mask]
                coefficient_report[key+'_'+name]=dict(count=len(differences),different=int((abs(differences)>tolerance).sum()),
                    maximum_absolute=float(abs(differences).max()),rms=float(np.sqrt(np.mean(differences**2))),tolerance=tolerance)
        report['material'][label]=coefficient_report
    after=binding()
    if before!=after:raise RuntimeError('Native source, controller or actual operator input changed during audit.')
    report['source_binding_after']=after
    report['pec_edge_masks_equal']=all(not value['native_only'] and not value['reference_only']for value in report['pec'].values())
    report['scope']='Actual constrained-edge equality and constitutive-array comparison; no time-domain spatial/reference accuracy claim.'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--material-dump',type=Path,required=True);parser.add_argument('--operator-dump',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--spacing',type=float,default=.01)
    parser.add_argument('--input-manifest',type=Path)
    args=parser.parse_args();torch.set_num_threads(2)
    run(args.material_dump,args.operator_dump,args.output,spacing=args.spacing,input_manifest=args.input_manifest)
