"""Validation-only uniform-plane limit of the joint native copper response.

Uses the real native field solver and a constant transverse mortar, independent
of triangle approximation, to isolate constitutive/average/jump normalization.
This diagnostic attaches private prepared state; it is not a public Scene API.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import witwin.maxwell as mw
from witwin.maxwell.compiler.curved_copper import compile_copper_yee_response
from witwin.maxwell.compiler.power_loss import _component_geometry
from witwin.maxwell.fdtd.curved_copper import JointCopperMortar
from tests.materials.sheet.sheet_measurement import sheet_scattering_coefficients
from tests.materials.surface_impedance.test_copper_foil import _film_transfer
from scripts.dev.fdtd.curved_copper_prototype import prototype_hooks


def measure(*, offset=0., spacing=.005):
    frequency,sigma=1e9,5.8e7
    delta=math.sqrt(2/(2*math.pi*frequency*mw.constants.MU_0*sigma))
    material=mw.CopperFoil(thickness=delta,conductivity=sigma,frequency_range=(.5e9,2e9))
    scene=mw.Scene(domain=mw.Domain(bounds=((-0.3,.3),(-.01,.01),(-.01,.01))),
        grid=mw.GridSpec.uniform(spacing),boundary=mw.BoundarySpec.faces(default="pml",num_layers=12,y="pmc",z="pec"),
        device="cuda",sources=(mw.PlaneWave(direction=(1.,0.,0.),polarization=(0.,0.,1.),
            source_time=mw.CW(frequency,amplitude=80.)),))
    simulation=mw.Simulation.fdtd(scene,frequency=frequency,full_field_dft=True,
        run_time=mw.TimeConfig(transient_cycles=24,steady_cycles=20),spectral_sampler=mw.SpectralSampler(window="hanning"))
    prepared=simulation.prepare()
    solver=prepared.solver
    nodes=tuple(torch.as_tensor(getattr(solver.scene,axis+"_nodes64"),device="cuda",dtype=torch.float64) for axis in "xyz")
    shapes_e=tuple(getattr(solver,name).shape for name in ("Ex","Ey","Ez"))
    shapes_h=tuple(getattr(solver,name).shape for name in ("Hx","Hy","Hz"))
    i=int(torch.searchsorted(nodes[0],nodes[0].new_tensor(0.),right=True))-1
    x=float(nodes[0][i])+offset*spacing
    frac=(x-float(nodes[0][i]))/spacing
    area=.02**2
    yz=solver.scene.dy_dual64[:,None]*solver.scene.dz_primal64[None,:]/math.sqrt(area)
    y,z=torch.meshgrid(torch.arange(shapes_e[2][1],device="cuda"),torch.arange(shapes_e[2][2],device="cuda"),indexing="ij")
    eoffset=math.prod(shapes_e[0])+math.prod(shapes_e[1])
    columns=torch.cat((((i*shapes_e[2][1]+y)*shapes_e[2][2]+z+eoffset).flatten(),
                        (((i+1)*shapes_e[2][1]+y)*shapes_e[2][2]+z+eoffset).flatten()))
    values=torch.as_tensor(yz,device="cuda").flatten()
    values=torch.cat((values*(1-frac),values*frac))
    keep=values!=0
    esupport,inverse=torch.unique(columns[keep],sorted=True,return_inverse=True)
    ge=torch.sparse_coo_tensor(torch.stack((torch.zeros_like(inverse),inverse)),values[keep],(1,len(esupport))).coalesce()
    hoffset=math.prod(shapes_h[0])
    hsupport=((i*shapes_h[1][1]+y)*shapes_h[1][2]+z+hoffset).flatten()
    gh=torch.as_tensor(yz,device="cuda").reshape(1,-1).to_sparse().coalesce()
    integrated=dict(shapes=shapes_e,magnetic_shapes=shapes_h,support=esupport,matrix=ge,
        magnetic_support=hsupport,magnetic_matrix=gh)
    response=compile_copper_yee_response(integrated,nodes)
    groups_e,groups_h={},{}
    def groups(support,names,magnetic):
        found={};masses=[];conductions=[];offset=0;keep=[]
        for component,name in enumerate(names):
            field=getattr(solver,name)
            positions=torch.nonzero((support>=offset)&(support<offset+field.numel())).flatten()
            linear=support[positions]-offset
            indices=torch.stack((linear//(field.shape[1]*field.shape[2]),linear//field.shape[2]%field.shape[1],linear%field.shape[2]),-1)
            coeff=getattr(solver,"c"+name.lower()+"_curl").flatten()[linear]
            valid=coeff!=0
            positions,linear,indices=positions[valid],linear[valid],indices[valid]
            keep.append(positions)
            offset+=field.numel()
            if not len(linear):continue
            if magnetic:
                widths=tuple(getattr(solver.scene,axis+("_dual64" if dimension==component else "_primal64"))
                             for dimension,axis in enumerate(("dx","dy","dz")))
            else:_,widths=_component_geometry(solver.scene,name)
            volume=values.new_ones(len(linear))
            for axis in range(3):volume*=torch.as_tensor(widths[axis],device="cuda")[indices[:,axis]]
            masses.append(getattr(solver,("mu_" if magnetic else "eps_")+name).flatten()[linear].double()*volume)
            conductions.append(getattr(solver,("sigma_m_" if magnetic else "sigma_e_")+name).flatten()[linear].double()*volume)
            found[name]=dict(indices=indices,linear_indices=linear,positions=positions,
                buffer=field.new_empty(len(linear)),double_buffer=values.new_empty(len(linear)))
        retained=torch.cat(keep)
        for group in found.values():group["positions"]=torch.searchsorted(retained,group["positions"])
        return found,torch.cat(masses),torch.cat(conductions),retained
    groups_e,me,se,free=groups(response["electric_support"],("Ex","Ey","Ez"),False)
    groups_h,mh,sh,_=groups(response["magnetic_support"],("Hx","Hy","Hz"),True)
    ge=response["electric_trace"].to_dense()[:,free]
    gh=response["magnetic_trace"].to_dense()
    ct=response["curl_transpose"].to_dense()[free].to_sparse().coalesce()
    runtime=JointCopperMortar(me,mh,se,sh,ct,ge,gh,torch.ones((1,1),device="cuda",dtype=torch.float64),material,solver.dt)
    solver._curved_copper=[dict(name="plane",runtime=runtime,electric_groups=groups_e,magnetic_groups=groups_h,
        electric=torch.empty_like(me),magnetic=torch.empty_like(mh))]
    with prototype_hooks():
        result=prepared.run()
    field=result.tensor("Ez");field=field[0] if field.ndim==4 else field
    profile=field[:,field.shape[1]//2,field.shape[2]//2].cpu().numpy()
    t,r=sheet_scattering_coefficients(profile,x=solver.scene.x_nodes64-x,dt=solver.dt,frequency=frequency,fit_gap=.025,fit_extent=.15)
    er,et,zi,zt=_film_transfer(delta)
    actual_a=1-abs(r)**2-zi/zt*abs(t)**2
    expected_a=1-abs(er)**2-zi/zt*abs(et)**2
    return dict(offset_cells=offset,spacing_m=spacing,reflection=[r.real,r.imag],transmission=[t.real,t.imag],
        expected_reflection=[er.real,er.imag],expected_transmission=[et.real,et.imag],
        reflection_absolute_error=abs(r-er),transmission_relative_error=abs(t-et)/abs(et),
        absorption=actual_a,expected_absorption=expected_a,absorption_relative_error=abs(actual_a-expected_a)/expected_a)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--offsets",type=float,nargs="+",default=(0.,.3,.7))
    parser.add_argument("--spacing",type=float,default=.005)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=dict(acceptance=dict(reflection_absolute_error_max=.002,transmission_relative_error_max=.03,
        absorption_relative_error_max=.03),rows=[])
    for offset in args.offsets:
        row=measure(offset=offset,spacing=args.spacing)
        report["rows"].append(row);args.output.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(row),flush=True)
