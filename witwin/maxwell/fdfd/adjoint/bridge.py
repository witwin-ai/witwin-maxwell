"""Differentiable FDFD via the adjoint method.

For the linear system A(eps) x = b with real loss L(x), the adjoint solve
A^T lam = conj(dL/dx*) yields material gradients without differentiating
through the solver iterations. The permittivity enters A only on the
diagonal (k0^2 * eps_face), so

    dL/deps_face = -2 k0^2 Re[lam * x]   (per Yee face)

which is pulled back to trainable scene inputs through the shared material
compiler graph, exactly like the FDTD gradient bridge.
"""

from __future__ import annotations

import cupy as cp
import torch

from ...adjoint_inputs import material_dependent_inputs, scene_trainable_material_tensors
from ...fdtd.material_pullback import pullback_material_input_gradients
from ...result import Result

# The FDFD matrix uses relative permittivity directly, so the pullback's
# absolute-permittivity scaling is the identity.
_RELATIVE_EPS_SCALE = 1.0


def _unsupported_fdfd_adjoint_medium(scene):
    for structure in getattr(scene, "structures", ()):
        material = getattr(structure, "material", None)
        if material is None:
            continue
        if getattr(material, "is_anisotropic", False):
            return "FDFD adjoint does not support anisotropic media yet."
        if getattr(material, "is_dispersive", False):
            return "FDFD adjoint does not support dispersive media yet."
    return None


class _FDFDGradientBridge:
    def __init__(self, simulation):
        self.simulation = simulation
        self.base_scene = simulation.scene
        self.material_inputs = self._resolve_material_inputs()
        if not self.material_inputs:
            raise NotImplementedError(
                "FDFD backward currently supports trainable scene inputs that contribute to "
                "prepared-scene material tensors."
            )
        self.solver = None
        self.solver_stats = None

    def _material_graph_scene(self):
        if self.simulation.scene_module is not None:
            return self.simulation.scene_module.to_scene()
        self.simulation._refresh_scene()
        return self.simulation.scene

    def _candidate_material_inputs(self) -> tuple[torch.Tensor, ...]:
        if self.simulation.scene_module is not None:
            return tuple(
                parameter
                for parameter in self.simulation.scene_module.parameters()
                if parameter.requires_grad
            )
        return scene_trainable_material_tensors(self.base_scene)

    def _resolve_material_inputs(self) -> tuple[torch.Tensor, ...]:
        scene = self._material_graph_scene()
        return material_dependent_inputs(scene, self._candidate_material_inputs())

    def forward(self, _material_inputs):
        solver = self.simulation._build_fdfd_solver()
        unsupported = _unsupported_fdfd_adjoint_medium(solver.scene)
        if unsupported is not None:
            raise NotImplementedError(unsupported)

        solver_cfg = self.simulation.config.solver
        solver.solve(
            max_iter=solver_cfg.max_iter,
            tol=solver_cfg.tol,
            restart=solver_cfg.restart,
        )
        if solver.E_field is None:
            raise RuntimeError("FDFD solve did not produce any field data.")

        self.solver = solver
        self.solver_stats = {
            "solver": {
                "type": solver_cfg.solver_type,
                "max_iter": solver_cfg.max_iter,
                "tol": solver_cfg.tol,
                "restart": solver_cfg.restart,
                "preconditioner": solver_cfg.preconditioner,
            },
            "converged": getattr(solver, "converged", None),
            "solver_info": getattr(solver, "solver_info", None),
            "final_residual": getattr(solver, "final_residual", None),
        }
        return solver.E_field

    def backward(self, material_inputs, grad_outputs):
        solver = self.solver
        scene = solver.scene
        solver_cfg = self.simulation.config.solver

        resolved = tuple(
            torch.zeros_like(field) if grad is None else grad
            for field, grad in zip(solver.E_field, grad_outputs)
        )
        # For real loss L, PyTorch hands us g = 2 dL/dx* (real-pair gradient
        # packed as complex). Solving A^T lam = conj(g) gives lam = 2 A^-T
        # dL/dx, and dL/deps_face = -2 k0^2 Re[(lam/2) x] = -k0^2 Re[lam x].
        grad_x = torch.cat([grad.reshape(-1) for grad in resolved]).to(torch.complex64)
        rhs = cp.from_dlpack(grad_x.conj().resolve_conj().contiguous()).astype(cp.complex64, copy=False)
        lam = solver.solve_adjoint(
            rhs,
            max_iter=solver_cfg.max_iter,
            tol=solver_cfg.tol,
            restart=solver_cfg.restart,
        )

        x = cp.from_dlpack(
            torch.cat([field.reshape(-1) for field in solver.E_field]).contiguous()
        )
        grad_eps_flat = (-solver.k0**2) * (lam * x).real.astype(cp.float32, copy=False)

        N_ex, N_ey = scene.N_ex, scene.N_ey
        grad_eps_ex = torch.from_dlpack(
            grad_eps_flat[:N_ex].reshape(scene.Nx_ex, scene.Ny_ex, scene.Nz_ex)
        )
        grad_eps_ey = torch.from_dlpack(
            grad_eps_flat[N_ex:N_ex + N_ey].reshape(scene.Nx_ey, scene.Ny_ey, scene.Nz_ey)
        )
        grad_eps_ez = torch.from_dlpack(
            grad_eps_flat[N_ex + N_ey:].reshape(scene.Nx_ez, scene.Ny_ez, scene.Nz_ez)
        )

        with torch.enable_grad():
            graph_scene = self._material_graph_scene()
            return pullback_material_input_gradients(
                graph_scene,
                inputs=material_inputs,
                grad_eps_ex=grad_eps_ex,
                grad_eps_ey=grad_eps_ey,
                grad_eps_ez=grad_eps_ez,
                eps0=_RELATIVE_EPS_SCALE,
            )


class _FDFDMaterialGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bridge, *material_inputs):
        ctx.bridge = bridge
        ctx.material_inputs = tuple(material_inputs)
        return bridge.forward(tuple(material_inputs))

    @staticmethod
    def backward(ctx, *grad_outputs):
        gradients = ctx.bridge.backward(ctx.material_inputs, grad_outputs)
        return (None, *gradients)


def run_fdfd_with_gradient_bridge(simulation) -> Result:
    bridge = _FDFDGradientBridge(simulation)
    Ex, Ey, Ez = _FDFDMaterialGradientFunction.apply(bridge, *bridge.material_inputs)
    return Result(
        method="fdfd",
        scene=simulation.scene,
        prepared_scene=bridge.solver.scene,
        frequency=simulation.frequency,
        frequencies=simulation.frequencies,
        solver=bridge.solver,
        fields={"EX": Ex, "EY": Ey, "EZ": Ez},
        metadata=simulation.metadata,
        solver_stats=bridge.solver_stats,
    )
