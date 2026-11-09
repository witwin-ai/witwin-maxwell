"""Hash padded coefficient keys without a dense six-column workspace."""
import torch

from .build import build_extension
from .dispatch import UNCHANGED


def compress(tensors, shape):
    if len(shape) != 3 or any(t.ndim != 3 or t.dtype != torch.float32
                             or t.device.type != 'cuda' or t.device != tensors[0].device
                             or not t.is_contiguous() or t.requires_grad for t in tensors):
        return UNCHANGED
    extension = build_extension()
    slots, inverse, overflow = extension.hash(tensors, shape)
    # A bounded open-addressing table deliberately retains unique for overflow.
    if overflow.item():
        return UNCHANGED
    occupied = (slots >= 0).nonzero()[:, 0]
    rows = extension.gather(tensors, shape, slots[occupied].contiguous())
    order = torch.arange(len(rows), device=rows.device)
    for column in reversed(range(6)):
        order = order[torch.argsort(rows[order, column], stable=True)]
    table = rows[order].view(torch.float32)
    remap = torch.empty_like(slots)
    remap[occupied[order]] = torch.arange(len(rows), device=rows.device, dtype=torch.int32)
    return extension.remap(inverse, remap, len(rows) > 65535).reshape(shape), table
