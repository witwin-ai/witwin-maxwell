def assert_current(self):
    if self._state() != self._binding:
        raise ValueError('Directional far actions or admitted partition metadata changed')
    self.test.assert_current()
    self.source.assert_current()
    for projection in (*self.targets, *self.origins):
        projection.assert_current()
    for coupling in self.couplings:
        coupling.assert_current()

def _action(self, value, *, adjoint):
    from .fmm_fft import _rhs_chunks
    self.assert_current()
    ni, no = (self.shape[0], self.shape[1]) if adjoint else (self.shape[1], self.shape[0])
    if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != ni or (value.device != self.device) or (value.dtype != self.dtype):
        raise ValueError('Directional far action requires matching physical complex128 currents')
    result = value.new_empty((math.prod(value.shape[:-1]), no))
    inputs, outputs = (self.targets, self.origins) if adjoint else (self.origins, self.targets)
    for start, chunk in _rhs_chunks(value, 1, self.rhs_batch):
        currents = torch.stack((chunk[..., :ni // 2], chunk[..., ni // 2:]), -2)
        moments = {id(p): p.receiver_upward(currents) if adjoint else p.source_upward(currents) for p in inputs}
        locals_by_owner = {id(p): [value.new_zeros((len(chunk), 2, count, 4)) for count in p.node_counts] for p in outputs}
        for coupling, rectangle in zip(self.couplings, self.rectangles, strict=True):
            target, origin = coupling._rwg_context
            a, b = rectangle[-2:]
            if adjoint:
                translated = coupling.adjoint(moments[id(target)][a])
                locals_by_owner[id(origin)][b] += translated
            else:
                translated = coupling.apply(moments[id(origin)][b])
                locals_by_owner[id(target)][a] += translated
            del translated
        batch = value.new_zeros((len(chunk), 2, no // 2))
        for projection in outputs:
            locals_ = tuple(locals_by_owner[id(projection)])
            batch += projection.source_downward(locals_) if adjoint else projection.receiver_downward(locals_)
            del locals_
        if not bool(torch.isfinite(batch).all()):
            raise ValueError('Directional far physical action exceeds native precision')
        result[start:start + len(chunk)] = torch.cat((batch[:, 0], batch[:, 1]), -1)
        del currents, moments, locals_by_owner, batch
    self.assert_current()
    return result.reshape(*value.shape[:-1], no)
