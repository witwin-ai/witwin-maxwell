"""Differentiable FDFD via the adjoint method.

For the linear system A(eps) x = b with real loss L(x), the adjoint solve
A^T lam = conj(dL/dx*) yields material gradients without differentiating
through the solver iterations. The permittivity enters A only on the
diagonal (k0^2 * eps_face), so

    g_eps_face = -k0^2 conj(lam * x)   (PyTorch complex cotangent)

which is pulled back through the same compiled Yee-edge material graph used
by the FDFD matrix assembly.
"""

from __future__ import annotations

import cupy as cp
import torch

from ...adjoint_inputs import material_dependent_inputs, scene_trainable_material_tensors
from ...result import Result
from ...scene import prepare_scene
from ..materials import compile_fdfd_materials


def _unsupported_fdfd_adjoint_medium(scene):
    for structure in getattr(scene, "structures", ()):
        material = getattr(structure, "material", None)
        if material is None:
            continue
        if getattr(material, "is_medium2d", False):
            return "FDFD adjoint does not support 2D sheet (Medium2D) media yet."
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
        from ..magnetic import require_fixed_permeability
        with torch.enable_grad():
            graph_scene = prepare_scene(self._material_graph_scene())
            _, permeability = compile_fdfd_materials(graph_scene, self.simulation.frequency, edges=True)
            require_fixed_permeability(permeability,_material_inputs)
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
                "precision": solver_cfg.precision,
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
        grad_x = torch.cat([grad.reshape(-1) for grad in resolved]).to(solver.E_field[0].dtype)
        rhs = cp.from_dlpack(grad_x.conj().resolve_conj().contiguous())
        lam = solver.solve_adjoint(
            rhs,
            max_iter=solver_cfg.max_iter,
            tol=solver_cfg.tol,
            restart=solver_cfg.restart,
        )

        x = cp.from_dlpack(
            torch.cat([field.reshape(-1) for field in solver.E_field]).contiguous()
        )
        # PyTorch's complex cotangent contracts as Re(conj(g) * d_eps).
        # Retaining its imaginary part also covers geometry that moves a lossy
        # material, whose complex node permittivity changes in both components.
        grad_eps_flat = (-solver.k0**2) * (lam * x).conj()

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
            prepared = prepare_scene(graph_scene)
            try:
                components, _ = compile_fdfd_materials(prepared, solver.frequency, edges=True)
                edges = tuple(components[axis] for axis in 'xyz')
                pairs = [
                    (edge, (gradient if edge.is_complex() else gradient.real).to(edge))
                    for edge, gradient in zip(edges, (grad_eps_ex, grad_eps_ey, grad_eps_ez))
                    if edge.requires_grad
                ]
                gradients = torch.autograd.grad(
                    [edge for edge, _ in pairs], material_inputs,
                    grad_outputs=[gradient for _, gradient in pairs], allow_unused=True,
                ) if pairs else (None,) * len(material_inputs)
                return tuple(
                    torch.zeros_like(value) if gradient is None else gradient.to(value)
                    for value, gradient in zip(material_inputs, gradients)
                )
            finally:
                prepared.release_meshgrid()


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
        monitors=simulation._fdfd_projection_monitors(bridge.solver, (Ex, Ey, Ez)),
        metadata=simulation.metadata,
        solver_stats=bridge.solver_stats,
    )
