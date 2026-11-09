"""CUDA tensor-product interpolation on owned physical support boxes."""
import math
import torch


class InterpolationBox:
    """Chebyshev-Lobatto cardinal functions, including collapsed box axes."""
    def __init__(self, lower, upper, order=4):
        if type(order) is not int or order < 1:
            raise ValueError('Interpolation order must be a positive node count')
        if (not isinstance(lower, torch.Tensor) or not isinstance(upper, torch.Tensor)
                or lower.shape != (3,) or upper.shape != (3,) or lower.device.type != 'cuda'
                or upper.device != lower.device or lower.dtype not in (torch.float32, torch.float64)
                or upper.dtype != lower.dtype or lower.requires_grad or upper.requires_grad
                or not bool(torch.isfinite(lower).all() & torch.isfinite(upper).all() & (upper >= lower).all())):
            raise ValueError('Interpolation boxes require ordered finite fixed real CUDA bounds')
        self.lower, self.upper, self.order = lower.clone(), upper.clone(), order
        self.device, self.dtype = lower.device, lower.dtype
        self.center = lower*.5+upper*.5
        self.half_width = (upper-lower)*.5
        if not bool(torch.isfinite(self.half_width).all()):
            raise ValueError('Interpolation box extent exceeds native precision')
        coordinates, weights = [], []
        for axis in range(3):
            count = order if bool(self.half_width[axis] > 0) else 1
            if count == 1:
                coordinates.append(torch.zeros(1, device=self.device, dtype=self.dtype))
                weights.append(torch.ones(1, device=self.device, dtype=self.dtype))
            else:
                j = torch.arange(count, device=self.device, dtype=self.dtype)
                coordinates.append(torch.cos(math.pi*j/(count-1)))
                w = torch.where(j.to(torch.int64) % 2 == 0, 1., -1.).to(self.dtype)
                w[0] *= .5
                w[-1] *= .5
                weights.append(w)
        self.coordinates, self.weights = tuple(coordinates), tuple(weights)
        grid = torch.stack(torch.meshgrid(*coordinates, indexing='ij'), dim=-1).reshape(-1, 3)
        self.nodes = self.center+grid*self.half_width
        self._layout = (self.order, self.device, self.dtype)
        self._versions = tuple((id(t), t._version) for t in self.tensors())

    def tensors(self):
        return (self.lower, self.upper, self.center, self.half_width, self.nodes, *self.coordinates, *self.weights)

    def assert_current(self):
        if ((self.order, self.device, self.dtype) != self._layout
                or tuple((id(t), t._version) for t in self.tensors()) != self._versions):
            raise ValueError('Interpolation box changed')

    def validate_points(self, points):
        self.assert_current()
        if (not isinstance(points, torch.Tensor) or points.ndim < 1 or points.shape[-1] != 3
                or points.device != self.device or points.dtype != self.dtype or points.requires_grad
                or not bool(torch.isfinite(points).all())):
            raise ValueError('Interpolation points require matching fixed real CUDA coordinates')
        slack = 32*torch.finfo(self.dtype).eps*torch.maximum(self.lower.abs(), self.upper.abs()).clamp_min(torch.finfo(self.dtype).tiny)
        if not bool(((points >= self.lower-slack) & (points <= self.upper+slack)).all()):
            raise ValueError('Interpolation points exceed the full support box')

    def evaluate(self, points):
        self.validate_points(points)
        values = []
        for axis, (nodes, weights) in enumerate(zip(self.coordinates, self.weights)):
            if len(nodes) == 1:
                values.append(torch.ones((*points.shape[:-1], 1), device=self.device, dtype=self.dtype))
                continue
            normalized = (points[..., axis]-self.center[axis])/self.half_width[axis]
            delta = normalized[..., None]-nodes
            hits = delta == 0
            terms = weights/torch.where(hits, torch.ones_like(delta), delta)
            cardinal = terms/terms.sum(-1, keepdim=True)
            cardinal = torch.where(hits.any(-1, keepdim=True), hits.to(self.dtype), cardinal)
            values.append(cardinal)
        result = (values[0][..., :, None, None]*values[1][..., None, :, None]*values[2][..., None, None, :]).flatten(-3)
        if not bool(torch.isfinite(result).all()):
            raise ValueError('Cardinal interpolation overflowed native precision')
        return result

    def transfer_to(self, child):
        """Return L_parent(child_nodes): child cardinal values times this transfer."""
        self.assert_current()
        child.assert_current()
        self.validate_points(torch.stack((child.lower, child.upper)))
        return self.evaluate(child.nodes)
