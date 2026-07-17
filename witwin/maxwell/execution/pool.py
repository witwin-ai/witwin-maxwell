from __future__ import annotations

import threading
from dataclasses import dataclass

import torch


def _normalize_pool_device(value, *, name: str, require_cuda: bool) -> str:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise TypeError(f"{name} must be a device, got {value!r}.") from exc
    if require_cuda:
        if device.type != "cuda":
            raise ValueError(f"{name} must be a CUDA device, got {device}.")
        if device.index is None:
            raise ValueError(f"{name} must include an explicit CUDA index, got {device}.")
    return str(device)


@dataclass(frozen=True)
class DeviceCapability:
    """Immutable snapshot of a pool device used for memory-aware placement."""

    device: str
    free_bytes: int | None
    total_bytes: int | None


class DeviceLease:
    """A single, exclusive-for-its-slot claim on a pool device.

    The lease is the only object that may run work on ``device``; the executor
    must return it to the pool so the slot is reused deterministically.
    """

    __slots__ = ("device", "_pool", "_released")

    def __init__(self, device: str, pool: "DevicePool"):
        self.device = device
        self._pool = pool
        self._released = False

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._pool._release(self.device)

    def __enter__(self) -> "DeviceLease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class DevicePool:
    """Deterministic device discovery and leasing for ensemble execution.

    Leasing hands out devices in a fixed pool order: a request always takes the
    first device (in declaration order) with a free slot, blocking until one is
    available. ``per_device_concurrency`` caps how many leases a single device
    may hold at once (default one large solver per GPU, per the plan). The pool
    also records the peak number of concurrent leases per device so tests can
    assert a device is never over-subscribed beyond its capacity.
    """

    def __init__(
        self,
        devices,
        *,
        per_device_concurrency: int = 1,
        require_cuda: bool = True,
        capabilities: dict[str, DeviceCapability] | None = None,
    ):
        if isinstance(devices, (str, bytes, torch.device)):
            raise TypeError("devices must be an ordered iterable of devices.")
        normalized = tuple(
            _normalize_pool_device(value, name=f"devices[{i}]", require_cuda=require_cuda)
            for i, value in enumerate(devices)
        )
        if not normalized:
            raise ValueError("DevicePool requires at least one device.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("DevicePool devices must be unique.")
        if isinstance(per_device_concurrency, bool) or not isinstance(per_device_concurrency, int):
            raise TypeError("per_device_concurrency must be an integer.")
        if per_device_concurrency <= 0:
            raise ValueError("per_device_concurrency must be > 0.")

        self._devices = normalized
        self._per_device_concurrency = per_device_concurrency
        self._capabilities = dict(capabilities or {})
        self._condition = threading.Condition()
        self._active = {device: 0 for device in normalized}
        self._peak = {device: 0 for device in normalized}

    @classmethod
    def discover(cls, devices, *, per_device_concurrency: int = 1) -> "DevicePool":
        """Build a CUDA pool and snapshot each device's free/total memory."""

        pool = cls(devices, per_device_concurrency=per_device_concurrency, require_cuda=True)
        capabilities = {}
        for device in pool._devices:
            torch_device = torch.device(device)
            try:
                free, total = torch.cuda.mem_get_info(torch_device)
                capabilities[device] = DeviceCapability(device, int(free), int(total))
            except (RuntimeError, AssertionError):
                capabilities[device] = DeviceCapability(device, None, None)
        pool._capabilities = capabilities
        return pool

    @property
    def devices(self) -> tuple[str, ...]:
        return self._devices

    @property
    def per_device_concurrency(self) -> int:
        return self._per_device_concurrency

    def capability(self, device: str) -> DeviceCapability | None:
        return self._capabilities.get(str(device))

    def peak_concurrency(self, device: str) -> int:
        with self._condition:
            return self._peak[str(device)]

    def _has_free_slot(self, device: str) -> bool:
        return self._active[device] < self._per_device_concurrency

    def lease(self, *, estimated_bytes: int | None = None) -> DeviceLease:
        """Lease the first free device in pool order, blocking until available.

        When ``estimated_bytes`` is provided and the device snapshot has a free
        figure, a device whose snapshot cannot hold the estimate is skipped in
        favor of one that can; if no snapshot fits, the first free device is
        still leased so the run's own preflight raises a structured CAPACITY
        failure rather than the pool silently deadlocking.
        """

        with self._condition:
            while True:
                free_devices = [d for d in self._devices if self._has_free_slot(d)]
                if free_devices:
                    chosen = self._select(free_devices, estimated_bytes)
                    self._active[chosen] += 1
                    self._peak[chosen] = max(self._peak[chosen], self._active[chosen])
                    return DeviceLease(chosen, self)
                self._condition.wait()

    def _select(self, free_devices, estimated_bytes: int | None) -> str:
        if estimated_bytes is not None:
            for device in free_devices:
                capability = self._capabilities.get(device)
                if capability is None or capability.free_bytes is None:
                    continue
                if estimated_bytes <= capability.free_bytes:
                    return device
        return free_devices[0]

    def _release(self, device: str) -> None:
        with self._condition:
            if self._active[device] <= 0:
                raise RuntimeError(f"DevicePool double-release for {device!r}.")
            self._active[device] -= 1
            self._condition.notify_all()


__all__ = ["DeviceCapability", "DeviceLease", "DevicePool"]
