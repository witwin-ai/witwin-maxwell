"""Shared scalar PMCHWT coupling for vector/divergence expansion channels."""
import torch


def scalar_moment_action(green, gradient, medium, value, *, adjoint=False):
    wave = medium.k
    e, m = 1j*medium.omega*medium.epsilon, -1j*medium.omega*medium.mu
    if adjoint:
        green, gradient, wave = green.mH, -gradient.transpose(-3, -2).conj(), complex(wave).conjugate()
        e, m = m.conjugate(), e.conjugate()
    vector, charge = value[..., :3], value[..., 3]
    potential = torch.matmul(green.unsqueeze(-3), vector)
    electric = torch.cat((potential, (-(charge.unsqueeze(-2)@green.mT.unsqueeze(-3)).squeeze(-2)/(wave*wave))[..., None]), dim=-1)
    curl = torch.stack(((vector[..., 2].unsqueeze(-2)@gradient[..., 1].mT.unsqueeze(-3)).squeeze(-2)-(vector[..., 1].unsqueeze(-2)@gradient[..., 2].mT.unsqueeze(-3)).squeeze(-2),
                        (vector[..., 0].unsqueeze(-2)@gradient[..., 2].mT.unsqueeze(-3)).squeeze(-2)-(vector[..., 2].unsqueeze(-2)@gradient[..., 0].mT.unsqueeze(-3)).squeeze(-2),
                        (vector[..., 1].unsqueeze(-2)@gradient[..., 0].mT.unsqueeze(-3)).squeeze(-2)-(vector[..., 0].unsqueeze(-2)@gradient[..., 1].mT.unsqueeze(-3)).squeeze(-2)), dim=-1)
    magnetic = torch.cat((curl, torch.zeros_like(curl[..., :1])), dim=-1)
    return torch.stack((magnetic[..., 0, :, :]+e*electric[..., 1, :, :],
                        m*electric[..., 0, :, :]+magnetic[..., 1, :, :]), dim=-3)
