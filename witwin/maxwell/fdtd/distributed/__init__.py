"""Internal single-node distributed FDTD runtime.

Only :class:`~witwin.maxwell.fdtd_parallel.FDTDParallelConfig` is public.  The
objects in this package deliberately remain implementation details so transport
handles and mutable rank-local state never become part of the Result contract.
"""

from .solver import DistributedFDTD

__all__ = ["DistributedFDTD"]
