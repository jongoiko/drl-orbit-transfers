import os
import numpy as np
import pykep as pk


def get_earth_radius(
    gravity_model_file_path=os.path.join(
        os.path.dirname(__file__), "envs/gravity/egm96.txt"
    )
):
    earth_radius = pk.util.load_gravity_model(gravity_model_file_path)[0]
    return earth_radius


def rsw_to_eci(rsw, position, velocity):
    rsw_x, rsw_y, rsw_z = rsw
    rsw_basis_z = np.cross(position, velocity)
    rsw_basis_z /= np.linalg.norm(rsw_basis_z)
    rsw_basis_y = np.cross(rsw_basis_z, position)
    rsw_basis_y /= np.linalg.norm(rsw_basis_y)
    eci = (
        rsw_x * position / np.linalg.norm(position)
        + rsw_y * rsw_basis_y
        + rsw_z * rsw_basis_z
    )
    return eci
