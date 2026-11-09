"""Static cell-current fields and distributional continuity diagnostics."""
import torch
from .volume_mesh import TetrahedralVolumeMesh
from .vie_operator import cell_green_potentials


def _current(mesh,current):
    if not isinstance(mesh,TetrahedralVolumeMesh):
        raise TypeError('Static current requires a tetrahedral volume mesh')
    mesh.assert_current()
    if (not isinstance(current,torch.Tensor) or current.device != mesh.vertices.device
            or current.dtype != mesh.vertices.dtype or current.shape != (len(mesh.cells),3)
            or not bool(torch.isfinite(current).all())):
        raise ValueError('Static current requires matching finite real CUDA cell vectors')


def cell_current_continuity(mesh,current):
    """Distributional normal-current jumps, including zero exterior current.

    Cellwise constant currents have zero classical divergence inside each cell.
    Their distributional divergence is the outward-current sum on shared faces
    and outward current at the body boundary. Nodal weak testing alone need not
    make these individual face distributions vanish.
    """
    _current(mesh,current)
    local_faces = mesh.cells.new_tensor([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
    faces = mesh.cells[:,local_faces].reshape(-1,3)
    unique,inverse,counts = torch.unique(faces.sort(-1).values,dim=0,return_inverse=True,return_counts=True)
    triangles = mesh.vertices[faces]
    area_normal = torch.linalg.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])/2
    source = current[:,None,:].expand(-1,4,-1).reshape(-1,3)
    flux = (area_normal*source).sum(-1)
    net_flux = current.new_zeros(len(unique)).index_add(0,inverse,flux)
    areas = current.new_zeros(len(unique)).index_add(0,inverse,area_normal.norm(dim=-1))/counts
    jump = net_flux/areas
    return {'faces':unique,'boundary_mask':counts == 1,'areas_m2':areas,
        'normal_current_jump_A_per_m2':jump,'face_flux_defect_A':net_flux,
        'area_weighted_jump_rms_A_per_m2':((areas*jump.square()).sum()/areas.sum()).sqrt(),
        'physical_qualification':False}


def cell_magnetostatic_current_fields(mesh,current,points):
    """Unit-permeability Biot-Savart H, plus its actual Maxwell residuals.

    H = integral grad(g) cross J, g=1/(4*pi*r). No frequency limit is used.
    Scalar permeability interface corrections must be solved separately. For
    a current with distributional divergence, this field has an Ampere defect
    grad(div(integral g J)); it must not be presented as an exact DC solution.
    """
    _current(mesh,current)
    _,gradient,hessian = cell_green_potentials(mesh,points,0.,order=2)
    gradient,hessian = gradient.real,hessian.real
    magnetic = torch.linalg.cross(gradient,current[None].expand_as(gradient)).sum(1)
    # The Hessian includes its volume contact, so curl(H) can be tested inside.
    ampere_defect = torch.einsum('qtij,tj->qi',hessian,current)
    contact_current = -torch.einsum('qt,ti->qi',hessian.diagonal(dim1=-2,dim2=-1).sum(-1),current)
    derivative = torch.stack([torch.linalg.cross(hessian[:,:,axis,:],current[None].expand_as(gradient)).sum(1)
                              for axis in range(3)],dim=-1)
    return magnetic,{'kernel_version':'static_tetrahedral_biot_savart_contact_v1','frequency_hz':0.,
        'curl_H_A_per_m2':contact_current+ampere_defect,
        'ampere_defect_A_per_m2':ampere_defect,
        'divergence_H_A_per_m2':derivative.diagonal(dim1=-2,dim2=-1).sum(-1),
        'interface_conditions_solved':False,'physical_qualification':False}


def cell_tensor_magnetostatic_current_fields(mesh,current,points,permeability,*,pair_batch=4096):
    """Homogeneous real-SPD magnetic particular field of induced P0 currents.

    For div(mu grad G_mu)=-delta, H=integral (mu grad G_mu) cross J.
    Oriented tetrahedral face integrals retain the interior contact exactly.
    curl H = J + mu Hess(G_mu)*J for distributionally conservative currents;
    the second term is returned as an explicit Ampere defect, not discarded.
    B=mu H has zero divergence. Interfaces and conductive magnetic DC channels
    require separate formulations. No public prescribed source API is added.
    """
    from .static_tensor_green import StaticTensorGreen
    _current(mesh,current)
    if type(pair_batch) is not int or pair_batch<1:
        raise ValueError('Tensor magnetic pair budget must be a positive integer')
    kernel=StaticTensorGreen(permeability)
    if permeability.device!=mesh.vertices.device or permeability.dtype!=mesh.vertices.dtype:
        raise ValueError('Magnetic permeability must match volume device and precision')
    kernel._validate(points,(None,3))
    local=mesh.cells.new_tensor([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
    triangles=mesh.vertices[mesh.cells[:,local]].reshape(-1,3,3)
    normals=torch.linalg.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    normals=normals/normals.norm(dim=-1,keepdim=True)
    conormals=normals@kernel.coefficient.T
    currents=current.repeat_interleave(4,dim=0)
    # Integral grad_x G over a cell equals -integral G n over its boundary.
    density=-torch.linalg.cross(conormals,currents)
    magnetic=points.new_zeros(len(points),3)
    derivative=points.new_zeros(len(points),3,3)
    contact=torch.zeros_like(magnetic);defect=torch.zeros_like(magnetic)
    faces=len(triangles)
    for start in range(0,len(points)*faces,pair_batch):
        flat=torch.arange(start,min(start+pair_batch,len(points)*faces),device=points.device)
        i,j=flat//faces,flat%faces
        potential,gradient=kernel.panel_integrals(points[i],triangles[j])
        magnetic=magnetic.index_add(0,i,potential[:,None]*density[j])
        derivative=derivative.index_add(0,i,density[j,:,None]*gradient[:,None,:])
        contact=contact.index_add(0,i,currents[j]*(conormals[j]*gradient).sum(-1,keepdim=True))
        defect=defect.index_add(0,i,-conormals[j]*(currents[j]*gradient).sum(-1,keepdim=True))
    flux=magnetic@kernel.coefficient.T
    divergence=torch.einsum('ij,qji->q',kernel.coefficient,derivative)
    if not all(bool(torch.isfinite(v).all()) for v in (magnetic,flux,derivative,contact,defect,divergence)):
        raise ValueError('Tensor magnetic current field is not representable')
    return magnetic,flux,dict(kernel_version='spd_tensor_static_cell_current_face_layers_v1',frequency_hz=0.,
        curl_H_A_per_m2=contact+defect,ampere_defect_A_per_m2=defect,contact_current_A_per_m2=contact,
        gradient_H_A_per_m2=derivative,divergence_B_T_per_m=divergence,
        source_distributional_continuity_certified=False,interface_conditions_solved=False,
        pair_batch=pair_batch,physical_qualification=False)
