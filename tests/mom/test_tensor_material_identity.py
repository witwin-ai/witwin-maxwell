"""Native susceptibility identities bind values beyond formatted representations."""
import json
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.harmonic_material import material_signature


def test_tensor_material_signature_exact_values_and_portable_content():
    q=torch.full((3,3,3),.25,dtype=torch.float64)
    changed=q.clone();changed[0,1,2]+=2**-40
    a=mw.Material(nonlinearity=mw.TensorNonlinearSusceptibility(chi2=q))
    b=mw.Material(nonlinearity=mw.TensorNonlinearSusceptibility(chi2=changed))
    assert repr(a.nonlinearity)==repr(b.nonlinearity)
    assert material_signature(a)!=material_signature(b)
    copied=mw.Material(nonlinearity=mw.TensorNonlinearSusceptibility(chi2=q.clone().requires_grad_()))
    assert material_signature(a)==material_signature(copied)
    before=material_signature(a)
    q[2,1,0]+=2**-40
    assert material_signature(a)!=before
    assert json.loads(json.dumps(before))
    rotation=torch.eye(3,dtype=torch.float64)
    r=mw.Material(nonlinearity=mw.TensorNonlinearSusceptibility(chi2=q,rotation=rotation))
    old=material_signature(r);rotation[0,1]=2**-40
    assert material_signature(r)!=old


def test_scalar_material_signature_retains_existing_declaration_format():
    from dataclasses import fields
    material=mw.Material(eps_r=2.,nonlinearity=mw.NonlinearSusceptibility(chi2=.2))
    expected=(type(material).__name__,tuple((f.name,repr(getattr(material,f.name)))
        for f in fields(material) if f.name not in ('material_id','version')))
    assert material_signature(material)==expected
