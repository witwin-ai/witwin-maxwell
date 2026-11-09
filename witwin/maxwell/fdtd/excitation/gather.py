"""Prepare immutable CSR source routes; CUDA alone evaluates and adds deltas."""
from __future__ import annotations

import math

import torch


# Bound temporary routing storage before allocating entry/image expansions.
MAX_GATHER_CONTRIBUTIONS = 8_000_000


def build_source_gather(offsets, codes=None, *, shape=None, wrap=0):
    """Return (rows, targets[component, offset], entries[index, image flags]).

    Input entries must already follow patch ordinal, then C-order patch index.
    Stable target sorting preserves that order, then increasing image subset.
    Offsets may include a batch-lane base; images always stay in that lane.
    Flags contain the subset in bits 0..2 and former atomic-boundary bit 3.
    """
    if wrap < 0 or wrap > 7 or (wrap and (shape is None or len(shape) != 3)):
        raise ValueError("Invalid source gather wrap geometry.")
    images = [subset for subset in range(8) if not subset & ~wrap]
    count = offsets.numel()
    if count * len(images) > MAX_GATHER_CONTRIBUTIONS:
        raise ValueError("Source gather exceeds the bounded prepare-time routing budget.")
    offsets = offsets.to(dtype=torch.int64).reshape(-1)
    codes = torch.zeros_like(offsets) if codes is None else codes.to(dtype=torch.int64).reshape(-1)
    if codes.numel() != count:
        raise ValueError("Source gather component and offset lengths differ.")
    indices = torch.arange(count, device=offsets.device, dtype=torch.int64)
    subsets = torch.tensor(images, device=offsets.device, dtype=torch.int64)
    targets = offsets[:, None].expand(-1, len(images)).clone()
    valid = torch.ones_like(targets, dtype=torch.bool)
    boundary_entry = torch.zeros_like(offsets, dtype=torch.bool)
    if wrap:
        nx, ny, nz = map(int, shape)
        if min(nx, ny, nz) < 1 or math.prod(shape) > torch.iinfo(torch.int64).max:
            raise ValueError("Invalid source gather field shape.")
        local = offsets % (nx * ny * nz)
        coords = (local // (ny * nz), (local // nz) % ny, local % nz)
        for axis, (coord, size, stride) in enumerate(zip(coords, (nx, ny, nz), (ny*nz, nz, 1))):
            selected = (subsets & (1 << axis)) != 0
            boundary = (coord == 0) | (coord == size - 1)
            if wrap & (1 << axis):
                boundary_entry |= boundary
            valid &= ~selected[None, :] | boundary[:, None]
            paired = torch.where(coord == 0, size - 1, 0)
            targets += (paired - coord)[:, None] * stride * selected[None, :]
    flags = subsets[None, :] | (boundary_entry[:, None].to(torch.int64) << 3)
    entries = torch.stack((indices[:, None].expand_as(targets), flags), -1)[valid]
    components = codes[:, None].expand_as(targets)[valid]
    targets = targets[valid]
    # Two stable sorts avoid overflowing a packed component/offset key.
    order = torch.argsort(targets, stable=True)
    order = order[torch.argsort(components[order], stable=True)]
    targets = torch.stack((components[order], targets[order]), -1)
    entries = entries[order].contiguous()
    starts = torch.ones(targets.shape[0], device=offsets.device, dtype=torch.bool)
    starts[1:] = (targets[1:] != targets[:-1]).any(dim=1)
    rows = torch.cat((torch.nonzero(starts).flatten(), offsets.new_tensor([entries.shape[0]])))
    return rows.contiguous(), targets[starts].contiguous(), entries


def build_patch_gather(field_shape, patch_shape, offsets, *, device, wrap=0):
    """Clip a rectangular patch while retaining its original entry indices."""
    count = math.prod(patch_shape)
    if count * (1 << int(wrap).bit_count()) > MAX_GATHER_CONTRIBUTIONS:
        raise ValueError("Source gather exceeds the bounded prepare-time routing budget.")
    q = torch.arange(count, device=device, dtype=torch.int64)
    i = q // (patch_shape[1] * patch_shape[2]) + offsets[0]
    j = (q // patch_shape[2]) % patch_shape[1] + offsets[1]
    k = q % patch_shape[2] + offsets[2]
    nx, ny, nz = field_shape
    valid = (i >= 0) & (i < nx) & (j >= 0) & (j < ny) & (k >= 0) & (k < nz)
    rows, targets, entries = build_source_gather(((i*ny+j)*nz+k)[valid], shape=field_shape, wrap=wrap)
    entries[:, 0] = q[valid][entries[:, 0]]
    return rows, targets, entries
