"""Exact fixed-geometry support and contact checks for named RF terminals.

This is geometry preprocessing. No field or mode solve is performed here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import itertools
import numpy as np
import torch
from witwin.core import Box
from .planar_pec import PlanarPEC


def _cross(a,b):
    return float(a[0]*b[1]-a[1]*b[0])


@dataclass(frozen=True)
class TerminalSupport:
    name: str
    lower: np.ndarray
    upper: np.ndarray
    roundoff: np.ndarray
    dtype: torch.dtype
    normal: int | None
    loops: tuple[np.ndarray, ...]
    copper: bool

    def contains(self, point, tolerance=1e-12):
        point=np.asarray(point,dtype=np.float64)
        if np.any(point < self.lower-tolerance) or np.any(point > self.upper+tolerance):
            return False
        if self.normal is None:
            return True
        indices=[i for i in range(3) if i != self.normal]
        return _inside(point[indices],self.loops,tolerance)

    def intervals(self, axis, point, tolerance=1e-12):
        point=np.asarray(point,dtype=np.float64)
        if any(point[i] < self.lower[i]-tolerance or point[i] > self.upper[i]+tolerance
               for i in range(3) if i != axis):
            return []
        if self.normal is None:
            return [(float(self.lower[axis]),float(self.upper[axis]))]
        if axis == self.normal:
            probe=point.copy();probe[axis]=self.lower[axis]
            return [(float(self.lower[axis]),float(self.lower[axis]))] if self.contains(probe,tolerance) else []
        indices=[i for i in range(3) if i != self.normal]
        varying=indices.index(axis)
        return _line_intervals(self.loops,1-varying,point[indices[1-varying]],tolerance)


def _inside(point, loops, tolerance):
    """Even/odd membership including the actual outer/hole boundary."""
    x,y=point
    parity=False
    for loop in loops:
        for a,b in zip(loop,np.roll(loop,-1,axis=0)):
            edge=b-a
            length=np.linalg.norm(edge)
            if length and abs(_cross(edge,point-a)) <= tolerance*length:
                if np.dot(point-a,point-b) <= tolerance**2:
                    return True
            if (a[1] > y) != (b[1] > y):
                crossing=a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1])
                if x < crossing:
                    parity=not parity
    return parity


def _line_intervals(loops, fixed_axis, coordinate, tolerance):
    varying=1-fixed_axis
    events=[]
    for loop in loops:
        for a,b in zip(loop,np.roll(loop,-1,axis=0)):
            delta=b[fixed_axis]-a[fixed_axis]
            if abs(delta) <= tolerance:
                if abs(coordinate-a[fixed_axis]) <= tolerance:
                    events.extend((a[varying],b[varying]))
            else:
                fraction=(coordinate-a[fixed_axis])/delta
                if -tolerance <= fraction <= 1+tolerance:
                    events.append(a[varying]+fraction*(b[varying]-a[varying]))
    events=sorted(set(float(v) for v in events))
    intervals=[]
    for low,high in zip(events[:-1],events[1:]):
        if high-low <= tolerance:
            continue
        probe=np.zeros(2);probe[fixed_axis]=coordinate;probe[varying]=(low+high)/2
        if _inside(probe,loops,tolerance):
            intervals.append((low,high))
    return intervals


def conductor_support(structure, *, required=False):
    """Describe supported conductor geometry without replacing it by an AABB."""
    material=getattr(structure,'material',None)
    copper=bool(getattr(material,'is_copper_foil',False))
    if not (copper or bool(getattr(material,'is_pec',False))):
        if required:raise ValueError(f"terminal structure {structure.name!r} must be PEC or CopperFoil.")
        return None
    geometry=structure.geometry
    if not isinstance(geometry,(Box,PlanarPEC)):
        raise NotImplementedError(f"Conductor {structure.name!r} requires Box or PlanarPEC geometry for named-terminal topology auditing.")
    for value in (geometry.position,geometry.rotation,getattr(geometry,'size',None)):
        if torch.is_tensor(value) and value.requires_grad:
            raise NotImplementedError('Named terminal support requires fixed conductor geometry.')
    rotation=torch.as_tensor(geometry.rotation).detach().cpu().double()
    if not torch.allclose(rotation,torch.tensor((1.,0.,0.,0.),dtype=torch.float64),rtol=0,atol=1e-12):
        raise ValueError(f"terminal structure {structure.name!r} must be an unrotated axis-aligned Box or planar support.")
    from ..ports import _coordinate_roundoff
    if isinstance(geometry,PlanarPEC):
        normal='xyz'.index(geometry.normal_axis)
        expected=torch.zeros(3,dtype=geometry.position.dtype);expected[normal]=geometry.coordinate
        if not torch.equal(geometry.position.detach().cpu(),expected):
            raise NotImplementedError('Mutated PlanarPEC placement must be redeclared through its polygon and coordinate.')
        transverse=[i for i in range(3) if i != normal]
        loops=tuple(loop.detach().cpu().numpy().copy() for loop in (geometry.vertices,*geometry.holes))
        lower=np.zeros(3);upper=np.zeros(3)
        lower[normal]=upper[normal]=geometry.coordinate
        lower[transverse]=loops[0].min(0);upper[transverse]=loops[0].max(0)
        return TerminalSupport(structure.name,lower,upper,np.zeros(3),torch.float64,normal,loops,copper)
    position=geometry.position.detach().cpu().double().numpy()
    size=geometry.size.detach().cpu().double().numpy()
    if not np.isfinite(position).all() or not np.isfinite(size).all():
        raise ValueError(f"terminal structure {structure.name!r} must have finite Box geometry.")
    normal=None;loops=()
    if copper:
        zero=np.flatnonzero(size == 0)
        if len(zero) != 1 or np.any(np.delete(size,zero[0]) <= 0):
            raise ValueError('CopperFoil terminal requires exactly one zero-normal Box extent.')
        normal=int(zero[0])
    elif np.any(size <= 0):
        raise ValueError(f"terminal structure {structure.name!r} must have positive Box extents; use PlanarPEC for an ideal sheet.")
    lower,upper=position-size/2,position+size/2
    if normal is not None:
        u,v=[i for i in range(3) if i != normal]
        loops=(np.array(((lower[u],lower[v]),(upper[u],lower[v]),(upper[u],upper[v]),(lower[u],upper[v]))),)
    roundoff=np.asarray(_coordinate_roundoff(geometry.position))+.5*np.asarray(_coordinate_roundoff(geometry.size))
    dtype=torch.float64 if torch.float64 in (geometry.position.dtype,geometry.size.dtype) else torch.float32
    return TerminalSupport(structure.name,lower,upper,roundoff,dtype,normal,loops,copper)


def _segment_events(a,b,c,d,tolerance):
    r,s=b-a,d-c
    cross=float(_cross(r,s))
    if abs(cross) <= tolerance*max(np.linalg.norm(r),np.linalg.norm(s),tolerance):return []
    t=float(_cross(c-a,s)/cross);u=float(_cross(c-a,r)/cross)
    return [float((a+t*r)[0])] if 0 <= t <= 1 and 0 <= u <= 1 else []


def _coplanar_contact(a,b,tolerance):
    events=[p[0] for support in (a,b) for loop in support.loops for p in loop]
    edges=lambda support:((p,q) for loop in support.loops for p,q in zip(loop,np.roll(loop,-1,axis=0)))
    for p,q in edges(a):
        for r,s in edges(b):events.extend(_segment_events(p,q,r,s,tolerance))
    events=sorted(set(events))
    for first,second in zip(events[:-1],events[1:]):
        if second-first <= tolerance:continue
        x=(first+second)/2
        ia=_line_intervals(a.loops,0,x,tolerance);ib=_line_intervals(b.loops,0,x,tolerance)
        if any(min(v1,w1)-max(v0,w0) > tolerance for v0,v1 in ia for w0,w1 in ib):return True
    # Adjacent coplanar sheets share a physical seam with nonzero length.
    for p,q in edges(a):
        for r,s in edges(b):
            direction=q-p;length=np.linalg.norm(direction)
            if not length:continue
            if abs(_cross(direction,r-p)) <= tolerance*length and abs(_cross(direction,s-p)) <= tolerance*length:
                axis=int(np.argmax(abs(direction)))
                if min(max(p[axis],q[axis]),max(r[axis],s[axis]))-max(min(p[axis],q[axis]),min(r[axis],s[axis])) > tolerance:return True
    return False


def contact_kind(a,b,tolerance):
    """Return a qualified finite contact, ambiguous point contact, or separation."""
    overlap=np.minimum(a.upper,b.upper)-np.maximum(a.lower,b.lower)
    if np.any(overlap < -tolerance):return None
    if a.normal is None and b.normal is None:
        return 'finite' if np.count_nonzero(overlap > tolerance) >= 2 else 'ambiguous'
    if a.normal is None or b.normal is None:
        sheet,box=(b,a) if a.normal is None else (a,b)
        normal=sheet.normal;transverse=[i for i in range(3) if i != normal]
        lo=np.maximum(sheet.lower,box.lower);hi=np.minimum(sheet.upper,box.upper)
        if all(hi[i]-lo[i] > tolerance for i in transverse):
            u,v=transverse
            rectangle=TerminalSupport(box.name,lo,hi,np.zeros(3),torch.float64,normal,
                (np.array(((lo[u],lo[v]),(hi[u],lo[v]),(hi[u],hi[v]),(lo[u],hi[v]))),),False)
            return 'finite' if _coplanar_contact(sheet,rectangle,tolerance) else None
        # A sheet meeting the side of a solid along a finite seam is valid.
        if sum(hi[i]-lo[i] > tolerance for i in transverse)==1:
            varying=next(i for i in transverse if hi[i]-lo[i] > tolerance)
            point=(lo+hi)/2
            return 'finite' if any(min(high,hi[varying])-max(low,lo[varying]) > tolerance
                                  for low,high in sheet.intervals(varying,point,tolerance)) else None
        return 'ambiguous' if sheet.contains((lo+hi)/2,tolerance) else None
    if a.normal == b.normal:
        if _coplanar_contact(a,b,tolerance):return 'finite'
        return 'ambiguous' if any(_inside(p,b.loops,tolerance) for loop in a.loops for p in loop) or any(_inside(p,a.loops,tolerance) for loop in b.loops for p in loop) else None
    varying=next(i for i in range(3) if i not in (a.normal,b.normal))
    point=np.maximum(a.lower,b.lower).copy()
    point[a.normal]=a.lower[a.normal];point[b.normal]=b.lower[b.normal]
    ia=a.intervals(varying,point,tolerance);ib=b.intervals(varying,point,tolerance)
    if any(min(v1,w1)-max(v0,w0) > tolerance for v0,v1 in ia for w0,w1 in ib):return 'finite'
    return 'ambiguous' if a.contains(point,tolerance) and b.contains(point,tolerance) else None


def audit_components(supports, positive, negative, tolerance):
    parent=list(range(len(supports)));lenient=list(parent);contacts=[]
    def root(parents,i):
        while parents[i]!=i:i=parents[i]
        return i
    for i,a in enumerate(supports):
        for j in range(i):
            kind=contact_kind(a,supports[j],tolerance+max(a.roundoff)+max(supports[j].roundoff))
            if kind is None:continue
            contacts.append({'structures':(supports[j].name,a.name),'kind':kind})
            lenient[root(lenient,i)]=root(lenient,j)
            if kind == 'finite':parent[root(parent,i)]=root(parent,j)
    i=next(i for i,s in enumerate(supports) if s.name==positive)
    j=next(i for i,s in enumerate(supports) if s.name==negative)
    if root(parent,i)==root(parent,j):raise ValueError('terminal surfaces are not distinct facing faces: positive and negative named terminals are shorted by a connected conductor component.')
    if root(lenient,i)==root(lenient,j):raise ValueError('terminal connectivity depends on an unqualified zero-area point/edge contact; declare a finite contact.')
    return {'contacts':contacts,'positive_component':sorted(s.name for k,s in enumerate(supports) if root(parent,k)==root(parent,i)),
            'negative_component':sorted(s.name for k,s in enumerate(supports) if root(parent,k)==root(parent,j))}


def _scene_supports(structures,target_names):
    names=[getattr(s,'name',None) for s in structures]
    output=[]
    for index,structure in enumerate(structures):
        if not bool(getattr(structure,'enabled',True)):continue
        support=conductor_support(structure)
        if support is None:continue
        if support.name not in target_names and (not isinstance(support.name,str) or not support.name or names.count(support.name)>1):
            support=replace(support,name=f"<conductor:{index}:{support.name}>")
        output.append(support)
    return output


def resolve_named_terminals(port,structures,domain_bounds):
    """Resolve actual ray/surface intersections and audit the named return loop."""
    from ..ports import _coordinate_roundoff
    def target(reference,label):
        matches=[s for s in structures if getattr(s,'name',None)==reference.structure_name]
        if not matches:raise ValueError(f"{label} terminal structure {reference.structure_name!r} does not exist.")
        if len(matches)!=1:raise ValueError(f"{label} terminal structure name {reference.structure_name!r} is not unique.")
        if not bool(getattr(matches[0],'enabled',True)):raise ValueError(f"{label} terminal structure {reference.structure_name!r} is disabled.")
        try:return conductor_support(matches[0],required=True)
        except NotImplementedError as error:raise ValueError(str(error)) from error
    positive=target(port.positive_terminal,'positive');negative=target(port.negative_terminal,'negative')
    axis='xyz'.index(port.integration_path.axis)
    transverse=[i for i in range(3) if i!=axis]
    tolerance=1e-12*max(1.,*(abs(float(v)) for bound in domain_bounds for v in bound))
    for support in (positive,negative):
        for i,bound in enumerate(domain_bounds):
            if support.lower[i] < bound[0]-tolerance or support.upper[i] > bound[1]+tolerance:
                raise ValueError(f"terminal structure {support.name!r} lies outside the Scene domain along {'xyz'[i]}.")
    supports=_scene_supports(structures,(positive.name,negative.name))
    audit=audit_components(supports,positive.name,negative.name,tolerance)
    components={positive.name:[q for q in supports if q.name in audit['positive_component']],
                negative.name:[q for q in supports if q.name in audit['negative_component']]}
    lower=np.maximum(positive.lower,negative.lower);upper=np.minimum(positive.upper,negative.upper)
    surface=port._declared_current_surface
    if surface is None:
        if any(upper[i]-lower[i] <= tolerance for i in transverse):
            raise ValueError('terminal transverse footprints do not overlap with finite area; coplanar sheets require explicit current_surface.')
        point=(lower+upper)/2
    else:
        point=surface.position.detach().cpu().double().numpy().copy()
        if abs(point[axis]-port.reference_plane) > tolerance+_coordinate_roundoff(surface.position)[axis]:
            raise ValueError('current_surface must lie on the declared reference_plane.')
    intervals=[]
    for named in (positive,negative):
        for support in components[named.name]:
            intervals.extend((lo,hi,named.name,support.name) for lo,hi in support.intervals(axis,point,tolerance+max(support.roundoff)))
    if any(lo-tolerance <= port.reference_plane <= hi+tolerance for lo,hi,_,_ in intervals):
        raise ValueError('reference_plane must lie strictly between distinct facing terminal surfaces.')
    below=[(hi,name,member) for lo,hi,name,member in intervals if hi < port.reference_plane-tolerance]
    above=[(lo,name,member) for lo,hi,name,member in intervals if lo > port.reference_plane+tolerance]
    if not below or not above:
        raise ValueError('terminal surfaces are not distinct facing faces on the selected path; a polygon hole or slot may exclude the terminal point.')
    before=max(below);after=min(above)
    if before[1]==after[1] or {before[1],after[1]} != {positive.name,negative.name}:
        raise ValueError('terminal surfaces are not distinct facing faces on the selected path.')
    endpoints={before[1]:before[0],after[1]:after[0]}
    contact_members={before[1]:before[2],after[1]:after[2]}
    resolved=[];roundoffs=[]
    scalar_dtype=np.float64 if torch.float64 in (positive.dtype,negative.dtype) else np.float32
    for support in (positive,negative):
        coordinate=point.copy();coordinate[axis]=endpoints[support.name]
        if not any(member.contains(coordinate,tolerance+max(member.roundoff)) for member in components[support.name]):
            raise ValueError(f"terminal path does not contact the actual footprint of {support.name!r}.")
        stored=tuple(scalar_dtype(v) for v in coordinate)
        roundoff=positive.roundoff+negative.roundoff
        roundoff=roundoff+abs(np.asarray(stored,dtype=np.float64)-coordinate)
        resolved.append(stored);roundoffs.append(tuple(roundoff))
    if surface is None:
        position=(lower+upper)/2;position[axis]=port.reference_plane
        size=upper-lower;size[axis]=0.
        surface=Box(position=torch.tensor(position,dtype=torch.float64),size=torch.tensor(size,dtype=torch.float64))
        contour_roundoff=tuple(np.maximum(positive.roundoff,negative.roundoff))
    else:contour_roundoff=(0.,0.,0.)
    aperture=port.voltage_aperture
    if aperture is not None:
        apos=aperture.position.detach().cpu().double().numpy();asize=aperture.size.detach().cpu().double().numpy()
        alo,ahi=apos-asize/2,apos+asize/2
        for support in (positive,negative):
            if not rectangle_component_supported(components[support.name],axis,endpoints[support.name],alo,ahi,tolerance+max(support.roundoff)):
                raise ValueError(f"voltage_aperture leaves the actual terminal footprint or crosses a polygon hole on {support.name!r}.")
    audit.update(positive_target=positive.name,negative_target=negative.name,
                 positive_contact_structure=contact_members[positive.name],negative_contact_structure=contact_members[negative.name],
                 topology='exact axis-aligned Box/planar polygon supports; positive-length sheet seams, finite solid contacts',
                 voltage_aperture=aperture is not None)
    return dict(positive=resolved[0],negative=resolved[1],current_surface=surface,
                positive_roundoff=roundoffs[0],negative_roundoff=roundoffs[1],
                current_bound_roundoff=contour_roundoff,terminal_audit=audit)


def audit_compiled_terminal(scene,port,geometry):
    """Audit every native parallel voltage path, its endpoints and open gap."""
    supports=_scene_supports(scene.structures,(port.positive_terminal.structure_name,port.negative_terminal.structure_name))
    target={s.name:s for s in supports}
    axis='xyz'.index(geometry.axis)
    nodes=[getattr(scene,a+'_nodes64') for a in 'xyz']
    indices=geometry.voltage_indices.detach().cpu().numpy()
    tolerance=1e-12
    grouped={}
    for index in indices:
        key=tuple(int(index[i]) for i in range(3) if i!=axis)
        grouped.setdefault(key,[]).append(index)
    for edges in grouped.values():
        first=min(int(e[axis]) for e in edges);last=max(int(e[axis]) for e in edges)+1
        point=np.array([nodes[i][edges[0][i]] for i in range(3)])
        lo,hi=nodes[axis][first],nodes[axis][last]
        for label,name in ((port.positive,port.positive_terminal.structure_name),(port.negative,port.negative_terminal.structure_name)):
            contact=point.copy();contact[axis]=lo if abs(label[axis]-lo)<abs(label[axis]-hi) else hi
            key='positive_component' if name==port.positive_terminal.structure_name else 'negative_component'
            members=[target[member] for member in port._terminal_audit[key]]
            if not any(support.contains(contact,tolerance+max(support.roundoff)) for support in members):
                raise ValueError(f"native voltage path misses the actual named terminal {name!r}, including its holes/slots.")
        for support in supports:
            for start,end in support.intervals(axis,point,tolerance+max(support.roundoff)):
                if end > lo+tolerance+max(support.roundoff) and start < hi-tolerance-max(support.roundoff):
                    raise ValueError(f"voltage gap intersects conductor {support.name!r}; a named two-terminal feed cannot cross a conductor.")
    # The runtime's two-sided Cu guard still owns joint port/trace elimination.
    # Reject it here too for compile-only callers, rather than declaring a feed
    # whose native E row is an active foil state.
    from ..compiler.copper_foil import compile_copper_faces
    edge_keys={tuple(map(int,index)) for index in indices}
    component='xyz'.index(geometry.voltage_component[-1].lower())
    copper_faces=compile_copper_faces(scene,device='cpu')
    for face in copper_faces:
        if face.component == component and edge_keys.intersection(map(tuple,face.indices.tolist())):
            raise NotImplementedError('TerminalPort voltage row overlaps a split CopperFoil tangential trace; joint port/foil elimination is unsupported.')
    native_audit=_audit_native_connectivity(scene,port,geometry,supports,copper_faces,grouped)
    return {**port._terminal_audit,**native_audit,'native_parallel_path_count':len(grouped),
            'native_voltage_edge_count':len(indices),'gap_audit':'every native parallel path has two real target contacts and an open conductor-free interior'}


def _audit_native_connectivity(scene,port,geometry,supports,copper_faces,grouped):
    """Audit the actual hard-PEC/Cu edge graph after opening only feed rows.

    Sparse graph operations are fixed geometry preprocessing, not a field solve.
    The runtime uses these same PEC fractions. A physically separate pair that
    becomes connected by staircase ownership must be refined, rather than
    repaired by erasing unrelated conductor edges.
    """
    from types import SimpleNamespace
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from ..fdtd.runtime.materials import _pec_edge_open_fractions

    nodes=[np.asarray(getattr(scene,a+'_nodes64')) for a in 'xyz']
    shape=tuple(map(len,nodes));strides=np.array((shape[1]*shape[2],shape[2],1),dtype=np.int64)
    axis='xyz'.index(geometry.axis)
    rows=geometry.voltage_indices.detach().cpu().numpy()
    excluded={tuple(map(int,row)) for row in rows}
    edge_sets=[[] for _ in range(3)]
    fractions=_pec_edge_open_fractions(SimpleNamespace(_compiled_material_model=scene.compile_materials()))
    if fractions is not None:
        for component,name in enumerate(('Ex','Ey','Ez')):
            edge_sets[component].append(torch.nonzero(fractions[name]==0,as_tuple=False).cpu().numpy())
    for face in copper_faces:edge_sets[face.component].append(face.indices.cpu().numpy())
    for normal,kinds in enumerate(scene.boundary._face_kinds):
        for side,kind in enumerate(kinds):
            if kind != 'pec':continue
            for component in range(3):
                if component==normal:continue
                ranges=[np.arange(n-(i==component)) for i,n in enumerate(shape)]
                ranges[normal]=np.array([0 if side==0 else shape[normal]-1])
                edge_sets[component].append(np.stack(np.meshgrid(*ranges,indexing='ij'),axis=-1).reshape(-1,3))
    first=[];second=[]
    for component,parts in enumerate(edge_sets):
        if not parts:continue
        edges=np.unique(np.concatenate(parts),axis=0)
        if component==axis:
            edges=np.asarray([edge for edge in edges if tuple(map(int,edge)) not in excluded],dtype=np.int64).reshape(-1,3)
        if len(edges):
            first.append(edges@strides);second.append(edges@strides+strides[component])
    endpoint_sets=[[],[]]
    for edges in grouped.values():
        low=min(int(e[axis]) for e in edges);high=max(int(e[axis]) for e in edges)+1
        for label,coordinate in enumerate((port.positive,port.negative)):
            index=np.array(edges[0],dtype=np.int64)
            index[axis]=low if abs(coordinate[axis]-nodes[axis][low])<abs(coordinate[axis]-nodes[axis][high]) else high
            endpoint_sets[label].append(int(index@strides))
    # Opposite periodic faces refer to the same electrical node, including a
    # Bloch phase: zero tangential voltage remains a conductor connection.
    all_ids=np.unique(np.concatenate([*first,*second,*map(np.asarray,endpoint_sets)]))
    coordinates=np.stack(np.unravel_index(all_ids,shape),axis=1)
    periodic=[]
    for direction,kinds in enumerate(scene.boundary._face_kinds):
        if kinds[0] not in ('periodic','bloch'):continue
        selected=coordinates[:,direction]==shape[direction]-1
        if np.any(selected):
            periodic.append((all_ids[selected],all_ids[selected]-(shape[direction]-1)*strides[direction]))
    if periodic:
        first.extend(pair[0] for pair in periodic);second.extend(pair[1] for pair in periodic)
        all_ids=np.unique(np.concatenate([all_ids,*[pair[1] for pair in periodic]]))
    if first:
        a=np.searchsorted(all_ids,np.concatenate(first));b=np.searchsorted(all_ids,np.concatenate(second))
        graph=coo_matrix((np.ones(len(a),dtype=np.uint8),(a,b)),shape=(len(all_ids),len(all_ids))).tocsr()
        _,labels=connected_components(graph,directed=False)
    else:labels=np.arange(len(all_ids))
    terminal_labels=[np.unique(labels[np.searchsorted(all_ids,ids)]) for ids in endpoint_sets]
    if any(len(value)!=1 for value in terminal_labels):
        raise ValueError('native parallel feed contacts are disconnected on a named conductor; refine its contact geometry/grid.')
    if terminal_labels[0][0]==terminal_labels[1][0]:
        raise ValueError('native Yee conductor ownership shorts the named terminal gap; refine the geometry/grid instead of deleting conductor edges.')
    coordinates=np.stack(np.unravel_index(all_ids,shape),axis=1)
    points=np.stack([nodes[i][coordinates[:,i]] for i in range(3)],axis=1)
    for name,component_labels in zip((port.positive_terminal.structure_name,port.negative_terminal.structure_name),terminal_labels):
        support=next(s for s in supports if s.name==name)
        candidates=points[labels==component_labels[0]]
        tolerance=1e-12+max(support.roundoff)
        candidates=candidates[np.all((candidates>=support.lower-tolerance)&(candidates<=support.upper+tolerance),axis=1)]
        if not any(support.contains(point,tolerance) for point in candidates):
            raise ValueError(f'native feed contact is disconnected from actual named target {name!r}; refine its physical connection.')
    return {'native_connectivity':'two disjoint conductor components, with every feed contact connected to its named target',
            'native_conductor_edge_count':sum(map(len,first))}


def rectangle_component_supported(supports,axis,coordinate,lower,upper,tolerance):
    """Exact union coverage of a rectangular endpoint patch, including holes."""
    u,v=[i for i in range(3) if i!=axis]
    for values in itertools.product((lower[u],upper[u]),(lower[v],upper[v])):
        point=np.zeros(3);point[axis]=coordinate;point[u],point[v]=values
        if not any(s.contains(point,tolerance) for s in supports):return False
    def covered(intervals,lo,hi):
        end=lo
        for a,b in sorted(intervals):
            if a > end+tolerance:return False
            end=max(end,b)
            if end >= hi-tolerance:return True
        return end >= hi-tolerance
    if upper[u]-lower[u] <= tolerance or upper[v]-lower[v] <= tolerance:
        varying=u if upper[u]-lower[u] > tolerance else v
        point=(lower+upper)/2;point[axis]=coordinate
        return covered([pair for s in supports for pair in s.intervals(varying,point,tolerance)],lower[varying],upper[varying])
    events=[lower[u],upper[u]];loops=[]
    for support in supports:
        if not support.lower[axis]-tolerance <= coordinate <= support.upper[axis]+tolerance:continue
        events.extend((support.lower[u],support.upper[u]))
        if support.normal==axis:loops.extend(support.loops)
    for loop in loops:events.extend(loop[:,0])
    segments=[(p,q) for loop in loops for p,q in zip(loop,np.roll(loop,-1,axis=0))]
    for index,(p,q) in enumerate(segments):
        for r,s in segments[:index]:events.extend(_segment_events(p,q,r,s,tolerance))
    events=sorted(set(float(x) for x in events if lower[u]-tolerance <= x <= upper[u]+tolerance))
    for first,last in zip(events[:-1],events[1:]):
        if last-first <= tolerance:continue
        point=(lower+upper)/2;point[axis]=coordinate;point[u]=(first+last)/2
        if not covered([pair for s in supports for pair in s.intervals(v,point,tolerance)],lower[v],upper[v]):return False
    return True


def wave_terminal_supports(port, structures):
    """Resolve the named terminal and reference supports of a TerminalWavePort.

    Returns ``(terminal_supports, reference_supports)``. Every named structure must
    be a unique, enabled PEC/CopperFoil Box or PlanarPEC that crosses the port plane.
    """
    normal = 'xyz'.index(port.normal_axis)

    def resolve(reference):
        matches = [s for s in structures if getattr(s, 'name', None) == reference.structure_name
                   and bool(getattr(s, 'enabled', True))]
        if len(matches) != 1:
            raise ValueError(f"structure {reference.structure_name!r} must name exactly one enabled conductor.")
        support = conductor_support(matches[0], required=True)
        plane = float(port.reference_plane)
        tolerance = 1e-12 * max(1.0, abs(plane))
        if plane < support.lower[normal] - tolerance or plane > support.upper[normal] + tolerance:
            raise ValueError(f"conductor {reference.structure_name!r} does not cross the port plane.")
        return support

    return (tuple(resolve(ref) for ref in port.terminals), tuple(resolve(ref) for ref in port.reference))


def resolve_wave_terminal_default_path(port, structures):
    """Default voltage path for one signal over one reference conductor.

    The two footprints on the port plane must be separated along exactly one
    tangential axis and overlap with positive length along the other. The path is
    the straight segment from the signal's facing surface to the reference's facing
    surface through the centre of the overlap. Grid snapping happens at compile
    time; anything else is ambiguous and must use explicit voltage integrals.
    """
    (signal,), (reference,) = wave_terminal_supports(port, structures)
    normal = 'xyz'.index(port.normal_axis)
    tangential = [axis for axis in range(3) if axis != normal]
    scale = max(1.0, *(abs(float(v)) for v in (*signal.lower, *signal.upper, *reference.lower, *reference.upper)))
    tolerance = 1e-12 * scale
    separated = [axis for axis in tangential
                 if signal.lower[axis] > reference.upper[axis] + tolerance
                 or reference.lower[axis] > signal.upper[axis] + tolerance]
    overlapping = [axis for axis in tangential
                   if min(signal.upper[axis], reference.upper[axis]) - max(signal.lower[axis], reference.lower[axis]) > tolerance]
    if len(separated) != 1 or len(overlapping) != 1:
        raise ValueError(
            "the signal and reference footprints are not separated along exactly one tangential axis "
            "with an overlap along the other; the default voltage path is ambiguous (for example CPW "
            "or side-by-side conductors). Supply explicit voltage_integrals."
        )
    path_axis, offset_axis = separated[0], overlapping[0]
    above = signal.lower[path_axis] > reference.upper[path_axis]
    low = max(signal.lower[offset_axis], reference.lower[offset_axis])
    high = min(signal.upper[offset_axis], reference.upper[offset_axis])
    return {
        "path_axis": 'xyz'[path_axis],
        "offset_axis": 'xyz'[offset_axis],
        "start": float(signal.lower[path_axis] if above else signal.upper[path_axis]),
        "start_interval": (float(signal.lower[path_axis]), float(signal.upper[path_axis])),
        "end": float(reference.upper[path_axis] if above else reference.lower[path_axis]),
        "end_interval": (float(reference.lower[path_axis]), float(reference.upper[path_axis])),
        "offset": 0.5 * (low + high),
        "offset_interval": (float(low), float(high)),
    }
