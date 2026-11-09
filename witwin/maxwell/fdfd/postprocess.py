import numpy as np
import torch


def get_field_raw(solver):
    if solver.E_field is None:
        return None
    ex, ey, ez = solver.E_field
    return {
        "Ex": ex.cpu().numpy(),
        "Ey": ey.cpu().numpy(),
        "Ez": ez.cpu().numpy(),
    }


def interpolate_yee_to_center(solver):
    ex, ey, ez = solver.E_field
    if ex.shape == ey.shape == ez.shape:
        field_data = ex
    else:
        ex_int = 0.5 * (ex[:, :-1, :] + ex[:, 1:, :])
        ex_int = 0.5 * (ex_int[:, :, :-1] + ex_int[:, :, 1:])
        ey_int = 0.5 * (ey[:-1, :, :] + ey[1:, :, :])
        ey_int = 0.5 * (ey_int[:, :, :-1] + ey_int[:, :, 1:])
        ez_temp = 0.5 * (ez[:-1, :, :] + ez[1:, :, :])
        ez_int = 0.5 * (ez_temp[:, :-1, :] + ez_temp[:, 1:, :])
        field_data = torch.sqrt(torch.abs(ex_int) ** 2 + torch.abs(ey_int) ** 2 + torch.abs(ez_int) ** 2)

    x_c = solver.scene.x_half.cpu().numpy()
    y_c = solver.scene.y_half.cpu().numpy()
    z_c = solver.scene.z_half.cpu().numpy()
    return field_data.cpu().numpy(), x_c, y_c, z_c
