"""Border-padded trilinear sampling with an ordered CUDA texture pullback."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _trilinear_border_plan(texture, query):
    # Query ownership releases device plans with the compiled autograd graph.
    # Inference tensors have no mutation counter, so they cannot cache a plan.
    cacheable = not query.requires_grad and not query.is_inference()
    key = (id(query), query._version, query.shape, query.dtype, query.device,
           texture.shape, torch.cuda.current_stream(query.device).cuda_stream) if cacheable else None
    cached = getattr(query, "_maxwell_trilinear_plan", None) if cacheable else None
    if cached is not None and cached[0] == key:
        return cached[1]
    from ..fdtd.cuda.backend import get_compiled_extension

    voxels = torch.empty((query.numel() // 3, 8), device=query.device, dtype=torch.int64)
    weights = torch.empty_like(voxels, dtype=query.dtype)
    get_compiled_extension().trilinear_border_entries(
        query.contiguous().view(-1, 3), *texture.shape, voxels, weights)
    # Entries start in query order; a stable voxel sort preserves that order
    # within every row. Masked corners sort after the last voxel.
    ordered_voxels, order = voxels.flatten().sort(stable=True)
    rows = torch.searchsorted(ordered_voxels, torch.arange(
        math.prod(texture.shape) + 1, device=query.device, dtype=torch.int64))
    plan = (rows, order // 8, weights.flatten()[order])
    if cacheable:
        query._maxwell_trilinear_plan = key, plan
    return plan


class _TrilinearBorder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, texture, query):
        from ..fdtd.cuda.backend import get_compiled_extension

        texture_c = texture.contiguous()
        query_c = query.contiguous()
        out = texture.new_empty(query.shape[:-1])
        get_compiled_extension().trilinear_border_forward(
            texture_c, query_c.view(-1, 3), out)
        # The ordered texture pullback plan is built lazily in backward, only when a
        # texture gradient is requested. It is built from the saved query (the caller's
        # tensor when contiguous, so its cache survives across calls); autograd's saved
        # tensor version check rejects in-place edits between forward and backward.
        ctx.save_for_backward(texture_c, query_c)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if torch.is_grad_enabled():
            raise RuntimeError("Trilinear border sampling does not support double backward")
        from ..fdtd.cuda.backend import get_compiled_extension

        texture, query = ctx.saved_tensors
        ops = get_compiled_extension()
        grad_out = grad_out.contiguous()
        grad_texture = grad_query = None
        if ctx.needs_input_grad[0]:
            rows, indices, weights = _trilinear_border_plan(texture, query)
            grad_texture = torch.empty_like(texture)
            ops.trilinear_border_backward(rows, indices, weights, grad_out, grad_texture)
        if ctx.needs_input_grad[1]:
            grad_query = torch.empty_like(query)
            ops.trilinear_border_grid_backward(
                texture, query.view(-1, 3), grad_out, grad_query)
        return grad_texture, grad_query


def _sample_trilinear_border(texture, query):
    """Sample a (z,y,x) texture at normalized (...,3) (x,y,z) queries.

    The output has shape ``query.shape[:-1]``. CPU uses the deterministic ATen
    oracle; CUDA uses native float32/float64 kernels; the texture pullback uses a query-owned
    CSR plan built on the first backward. Non-finite queries follow grid_sample.
    Box position/size can require gradients, so both inputs are differentiable.
    """
    if texture.device.type == "cpu":
        return F.grid_sample(
            texture[None, None], query.reshape(1, 1, 1, -1, 3),
            mode="bilinear", padding_mode="border", align_corners=False,
        ).reshape(query.shape[:-1])
    if texture.dtype not in (torch.float32, torch.float64) or query.dtype != texture.dtype:
        raise TypeError("CUDA trilinear border sampling requires matching float32 or float64 inputs")
    return _TrilinearBorder.apply(texture, query)
