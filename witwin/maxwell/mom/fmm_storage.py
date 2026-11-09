"""Shared structural sizes for supported Cartesian M2L representations."""
import math


def shared_displacement_catalog(centers, pairs):
    """Group exact opposite/equal CUDA displacements using host integer keys.

    Numerical geometry stays native; only IEEE bits, orientation flags and
    topology cross to host metadata. Work tensors cover at most1024pairs. Zero
    coordinates are normalized only for key identity. No approximate binning,
    physical partition reduction or coefficient cache is introduced.
    """
    import torch
    if (not isinstance(centers,torch.Tensor) or centers.ndim!=2 or centers.shape[1]!=3
            or centers.dtype!=torch.float64 or centers.device.type!='cuda' or centers.requires_grad):
        raise ValueError('Shared compact catalog requires fixed CUDA float64 centers')
    pairs=tuple(pairs)
    if any(not isinstance(pair,(tuple,list)) or len(pair)!=2
            or any(type(n) is not int or not 0<=n<len(centers) for n in pair) for pair in pairs):
        raise ValueError('Shared compact catalog requires valid physical node pairs')
    owners,bindings,keys=[],[],{}
    for start in range(0,len(pairs),1024):
        aa,bb=torch.tensor(pairs[start:start+1024],device=centers.device,dtype=torch.long).unbind(-1)
        delta=centers[aa]-centers[bb]
        if not bool(torch.isfinite(delta).all() & (delta!=0).any(-1).all()):
            raise ValueError('Shared compact catalog requires finite nonzero displacements')
        axis=(delta!=0).to(torch.int64).argmax(-1)
        flip=delta.gather(1,axis[:,None]).squeeze(-1)<0
        canonical=torch.where(flip[:,None],-delta,delta)
        canonical=torch.where(canonical==0,torch.zeros_like(canonical),canonical)
        records=torch.cat((canonical.contiguous().view(torch.int64),flip[:,None].to(torch.int64)),dim=1)
        for index,record in enumerate(records.cpu().tolist(),start):
            key=tuple(record[:3]);orientation=bool(record[3])
            if key not in keys:
                keys[key]=(len(owners),orientation);owners.append(index)
            owner,original_orientation=keys[key]
            bindings.append((owner,orientation!=original_orientation))
    return tuple(owners),tuple(bindings)


def m2l_storage_bytes(order, real_bytes, representation='dense'):
    if representation not in ('dense', 'compact', 'fft_v1', 'compact_shared_displacement_v1', 'compact_shared_signed_axis_v1', 'radial_series_shared_radius_v1'):
        raise ValueError('M2L table_storage must be dense, compact or fft_v1')
    if type(order) is not int or order < 0 or real_bytes not in (4, 8):
        raise ValueError('M2L storage requires nonnegative order and native real precision')
    if representation == 'fft_v1' and (real_bytes != 8 or order > 12):
        raise ValueError('FFT M2L requires float64 coordinates and order zero through twelve')
    if representation=='compact_shared_displacement_v1' and real_bytes!=8:
        raise ValueError('Shared compact M2L requires native float64 coordinates')
    if representation == 'compact_shared_signed_axis_v1' and (real_bytes != 8 or order > 12):
        raise ValueError('Signed-axis M2L requires native float64 order0..12')
    if representation == 'radial_series_shared_radius_v1':
        if real_bytes != 8:
            raise ValueError('Radial series M2L requires native float64 coordinates')
        return radial_series_m2l_owner_bytes(order)
    count = math.comb(order+3, 3)
    if representation == 'fft_v1':
        return math.comb(2*order+4, 3)*16+32+count*56
    if representation == 'dense':
        return 4*count*count*2*real_bytes
    return math.comb(2*order+4, 3)*2*real_bytes+count*3*8+(2*order+2)*(order+1)*real_bytes


signed_axis_catalog_rule = 'exact_native_signed_axis_permutation_v1'


def signed_axis_displacement_catalog(centers, pairs):
    """Share radial scalar jets under exact coordinate permutations/reflections.

    Owners are original physical displacements, never rounded or projected.
    A binding means actual[i] = owner[permutation[i]] * signs[i]. Only IEEE
    keys, integer permutations and signs cross to host. Each original pair
    remains present, including separate bindings for coincident radial keys.
    """
    import torch
    if (not isinstance(centers,torch.Tensor) or centers.ndim!=2 or centers.shape[1]!=3
            or centers.dtype!=torch.float64 or centers.device.type!='cuda' or centers.requires_grad):
        raise ValueError('Signed-axis catalog requires fixed CUDA float64 centers')
    pairs=tuple(pairs)
    if any(not isinstance(pair,(tuple,list)) or len(pair)!=2
            or any(type(n) is not int or not 0<=n<len(centers) for n in pair) for pair in pairs):
        raise ValueError('Signed-axis catalog requires valid physical node pairs')
    owners,bindings,keys=[],[],{}
    for start in range(0,len(pairs),1024):
        aa,bb=torch.tensor(pairs[start:start+1024],device=centers.device,dtype=torch.long).unbind(-1)
        delta=centers[aa]-centers[bb]
        if not bool(torch.isfinite(delta).all() & (delta!=0).any(-1).all()):
            raise ValueError('Signed-axis catalog requires finite nonzero displacements')
        absolute,axes=torch.sort(delta.abs(),dim=-1,stable=True)
        signs=torch.where(delta<0,-torch.ones_like(axes),torch.ones_like(axes))
        records=torch.cat((absolute.contiguous().view(torch.int64),axes,signs),dim=-1)
        for index,record in enumerate(records.cpu().tolist(),start):
            key=tuple(record[:3]); order=tuple(record[3:6]); signs=tuple(record[6:])
            if key not in keys:
                keys[key]=(len(owners),order,signs);owners.append(index)
            owner,owner_order,owner_signs=keys[key]
            inverse=tuple(order.index(axis) for axis in range(3))
            permutation=tuple(owner_order[inverse[axis]] for axis in range(3))
            relative_signs=tuple(signs[axis]*owner_signs[permutation[axis]] for axis in range(3))
            bindings.append((owner,permutation,relative_signs))
    return tuple(owners),tuple(bindings)


def radial_displacement_catalog(displacements):
    """Group only identical native Helmholtz radius bits, retaining every vector.

    Radius evaluation is the original CUDA vector norm, with no rounded norm,
    approximate binning, orientation transform or physical partition change.
    Only native IEEE radius keys and original integer positions cross to host;
    validation/work tensors cover at most1024 physical vectors per chunk.
    """
    import torch
    if (not isinstance(displacements,torch.Tensor) or displacements.device.type!='cuda'
            or displacements.dtype!=torch.float64 or displacements.ndim!=2
            or displacements.shape[1]!=3 or displacements.requires_grad):
        raise ValueError('Radial catalog requires fixed native CUDA float64 displacements')
    return _radial_series_catalog_blocks((start,displacements[start:start+1024])
        for start in range(0,len(displacements),1024))


def radial_series_m2l_owner_bytes(order):
    """Native radius, original univariate series and original compact layout."""
    if type(order) is not int or not 0<=order<=12:
        raise ValueError('Radial series M2L requires native order0..12')
    return (8+16*(2*order+2)+24*math.comb(order+3,3)
            +8*(2*order+2)*(order+1))


radial_series_catalog_rule = 'exact_original_native_radius_bits_with_original_physical_vectors_v1'


def _radial_series_catalog_blocks(blocks):
    import torch
    owners,bindings,keys=[],[],{}
    for start,delta in blocks:
        radius=torch.linalg.vector_norm(delta,dim=-1)
        if not bool(torch.isfinite(delta).all() & torch.isfinite(radius).all() & (radius>0).all()):
            raise ValueError('Radial catalog requires finite positive original radii')
        for index,key in enumerate(radius.contiguous().view(torch.int64).cpu().tolist(),start):
            if key not in keys:
                keys[key]=len(owners);owners.append(index)
            bindings.append(keys[key])
    return tuple(owners),tuple(bindings)


def radial_series_pair_catalog(centers,pairs):
    """Exact radius keys for physical node pairs, with at most1024 deltas."""
    import torch
    if (not isinstance(centers,torch.Tensor) or centers.ndim!=2 or centers.shape[1]!=3
            or centers.dtype!=torch.float64 or centers.device.type!='cuda' or centers.requires_grad):
        raise ValueError('Radial pair catalog requires fixed CUDA float64 centers')
    pairs=tuple(pairs)
    if any(not isinstance(pair,(tuple,list)) or len(pair)!=2
            or any(type(n) is not int or not 0<=n<len(centers) for n in pair) for pair in pairs):
        raise ValueError('Radial pair catalog requires valid physical node pairs')
    def blocks():
        for start in range(0,len(pairs),1024):
            aa,bb=torch.tensor(pairs[start:start+1024],device=centers.device,dtype=torch.long).unbind(-1)
            yield start,centers[aa]-centers[bb]
    return _radial_series_catalog_blocks(blocks())


def radial_series_pair_storage_bytes(centers,pairs,order):
    pairs=tuple(pairs)
    return len(radial_series_pair_catalog(centers,pairs)[0])*radial_series_m2l_owner_bytes(order)+24*len(pairs)
