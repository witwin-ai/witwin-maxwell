"""Separate native electric/magnetic traces and physical surface-current mass."""
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.cut_surface import compile_integrated_traces
from witwin.maxwell.compiler.curved_copper import compile_copper_yee_response
from tests.materials.surface_impedance.test_cut_surface import _mesh

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="True mortar integration requires CUDA")


@pytest.mark.parametrize("bent",(False,True))
def test_two_native_trace_spaces_reproduce_independent_polynomials_and_surface_work(bent):
    geometry=(mw.SurfaceMesh.bent_rectangle(width=.02,length=.016,radius=.03,segments=12,device="cuda")
              if bent else _mesh())
    vertices,faces=geometry.to_mesh()
    nodes=(torch.linspace(-.04,.04,17,device="cuda",dtype=torch.float64),)*3
    result=compile_integrated_traces(vertices,faces,nodes,include_magnetic=True)
    electric_fields=[]
    magnetic_fields=[]
    halves=tuple((axis[:-1]+axis[1:])/2 for axis in nodes)
    for component in range(3):
        ecoords=tuple(halves[axis] if axis == component else nodes[axis] for axis in range(3))
        x,y,z=torch.meshgrid(*ecoords,indexing="ij")
        electric_fields.append((2*y-z,3*z+x,-x-y)[component].flatten())
        hcoords=tuple(nodes[axis] if axis == component else halves[axis] for axis in range(3))
        x,y,z=torch.meshgrid(*hcoords,indexing="ij")
        magnetic_fields.append((2*x,3*y,-z)[component].flatten())
    x,y,z=result["surface_points"].unbind(-1)
    expected_e=torch.stack((2*y-z,3*z+x,-x-y),-1)
    expected_h=torch.stack((2*x,3*y,-z),-1)
    basis=result["surface_basis"]
    weight=result["surface_weights"].repeat_interleave(3)
    for name,fields,expected in (("electric",electric_fields,expected_e),("magnetic",magnetic_fields,expected_h)):
        matrix=result["matrix" if name == "electric" else "magnetic_matrix"]
        support=result["support" if name == "electric" else "magnetic_support"]
        actual=torch.sparse.mm(matrix,torch.cat(fields)[support,None])[:,0]
        reference=torch.sparse.mm(basis.T,(weight*expected.flatten())[:,None])[:,0]
        torch.testing.assert_close(actual,reference,rtol=2e-12,atol=2e-16)
        current=torch.linspace(-1.,1.,len(actual),device="cuda",dtype=torch.float64)
        physical_current=torch.sparse.mm(basis,current[:,None])[:,0]
        torch.testing.assert_close(actual@current,(weight*expected.flatten()*physical_current).sum(),rtol=2e-12,atol=2e-16)
    dense_basis=basis.to_dense()
    direct_mass=dense_basis.T@(weight[:,None]*dense_basis)
    torch.testing.assert_close(result["surface_mass"],direct_mass,rtol=2e-12,atol=2e-14)
    assert int(torch.linalg.cholesky_ex(result["surface_mass"]).info) == 0
    electric_only=compile_integrated_traces(vertices,faces,nodes)
    torch.testing.assert_close(electric_only["matrix"].to_dense(),result["matrix"].to_dense(),rtol=2e-12,atol=2e-16)


def test_magnetic_reaction_curl_response_matches_both_native_yee_updates():
    from witwin.maxwell.compiler.power_loss import _component_geometry
    from witwin.maxwell.fdtd.runtime.stepping import update_magnetic_fields, update_electric_fields
    from tests.materials.surface_impedance.test_cut_surface import _simulation
    prepared = _simulation().prepare()
    solver = prepared.solver
    vertices, faces = _mesh().to_mesh()
    nodes = tuple(torch.as_tensor(getattr(solver.scene, axis+"_nodes64"), device="cuda", dtype=torch.float64) for axis in "xyz")
    integrated = compile_integrated_traces(vertices, faces, nodes, include_magnetic=True)
    response = compile_copper_yee_response(integrated, nodes)
    h = torch.sin(torch.arange(len(response["magnetic_support"]), device="cuda", dtype=torch.float64))
    e = torch.cos(torch.arange(len(response["electric_support"]), device="cuda", dtype=torch.float64))
    shapes_e, shapes_h = integrated["shapes"], integrated["magnetic_shapes"]
    size_e, size_h = sum(map(lambda s: s[0]*s[1]*s[2], shapes_e)), sum(map(lambda s: s[0]*s[1]*s[2], shapes_h))
    all_h = h.new_zeros(size_h).index_copy_(0, response["magnetic_support"], h)
    all_e = e.new_zeros(size_e).index_copy_(0, response["electric_support"], e)
    for values, names in ((all_h,("Hx","Hy","Hz")), (all_e,("Ex","Ey","Ez"))):
        offset = 0
        for name in names:
            field = getattr(solver,name)
            field.copy_(values[offset:offset+field.numel()].reshape(field.shape))
            offset += field.numel()
    old_h = tuple(getattr(solver,name).clone() for name in ("Hx","Hy","Hz"))
    for name in ("Hx","Hy","Hz"):
        getattr(solver,name).zero_()
    update_magnetic_fields(solver,solver.Hx,solver.Hy,solver.Hz,solver.Ex,solver.Ey,solver.Ez)
    # Isolate -C E from zero H; subtracting two O(1) float32 fields would
    # contaminate this deliberately small increment with cancellation roundoff.
    magnetic_change = []
    for component,name in enumerate(("Hx","Hy","Hz")):
        widths = tuple(getattr(solver.scene,axis+("_dual64" if dimension==component else "_primal64"))
                       for dimension,axis in enumerate(("dx","dy","dz")))
        volume = (torch.as_tensor(widths[0],device="cuda")[:,None,None]*
                  torch.as_tensor(widths[1],device="cuda")[None,:,None]*
                  torch.as_tensor(widths[2],device="cuda")[None,None,:])
        magnetic_change.append((-getattr(solver,name).double()*
            getattr(solver,"mu_"+name).double()*volume/solver.dt).flatten())
        getattr(solver,name).copy_(old_h[component])
    expected_h = (response["curl_transpose"].T@e)
    actual_h = torch.cat(magnetic_change)[response["magnetic_support"]]
    torch.testing.assert_close(actual_h,expected_h,rtol=5e-5,atol=3e-11)
    for name in ("Ex","Ey","Ez"):
        getattr(solver,name).zero_()
    update_electric_fields(solver,solver.Ex,solver.Ey,solver.Ez,solver.Hx,solver.Hy,solver.Hz,time_value=0.)
    electric_change = []
    for name in ("Ex","Ey","Ez"):
        _,widths = _component_geometry(solver.scene,name)
        volume = (torch.as_tensor(widths[0],device="cuda")[:,None,None]*
                  torch.as_tensor(widths[1],device="cuda")[None,:,None]*
                  torch.as_tensor(widths[2],device="cuda")[None,None,:])
        electric_change.append((getattr(solver,name).double()*getattr(solver,"eps_"+name).double()*volume/solver.dt).flatten())
    actual_e = torch.cat(electric_change)[response["electric_support"]]
    expected_e = response["curl_transpose"]@h
    torch.testing.assert_close(actual_e,expected_e,rtol=2e-6,atol=1e-12)
