import matplotlib.pyplot as plt
import numpy as np
import pykep as pk
import pandas as pd
import argparse
from math import radians
from utils import get_earth_radius


def equal_axis_size_3d(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    ranges = np.abs(limits[:, 1] - limits[:, 0])
    midpoints = limits.mean(axis=1)
    plot_radius = ranges.max() / 2
    ax.set_xlim3d([midpoints[0] - plot_radius, midpoints[0] + plot_radius])
    ax.set_ylim3d([midpoints[1] - plot_radius, midpoints[1] + plot_radius])
    ax.set_zlim3d([midpoints[2] - plot_radius, midpoints[2] + plot_radius])


def plot_orbit(orbit, color, ax, label):
    epoch = pk.epoch(0)
    planet = pk.planet.keplerian(epoch, orbit, 1, 1, 1, 1)
    return pk.orbit_plots.plot_planet(
        planet, axes=ax, color=color, s=0, legend=(False, label)
    )


def plot_earth(ax):
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    earth_wireframe = get_earth_radius() * np.array(
        [np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)]
    )
    return ax.plot_wireframe(*earth_wireframe, color="gray", alpha=0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trajectory_file_path", type=str, help="path to trajectory JSON file"
    )
    parser.add_argument(
        "experiment_file_path", type=str, help="path to experiment CSV file"
    )
    args = parser.parse_args()
    trajectory_data = pd.read_json(args.trajectory_file_path)
    traj_x, traj_y, traj_z = [
        trajectory_data.position.apply(lambda pos: pos[i]) for i in range(3)
    ]
    exp = pd.read_csv(args.experiment_file_path).set_index("param")
    initial_orbit = np.array(
        [
            exp.loc["a_km"]["initial"] * 1000.0,
            exp.loc["e"]["initial"],
            radians(exp.loc["i_deg"]["initial"]),
            radians(exp.loc["raan_deg"]["initial"]),
            radians(exp.loc["peri_deg"]["initial"]),
            0.0,
        ]
    )
    goal_orbit = np.array(
        [
            exp.loc["a_km"]["goal"] * 1000.0,
            exp.loc["e"]["goal"],
            radians(exp.loc["i_deg"]["goal"]),
            radians(exp.loc["raan_deg"]["goal"]),
            radians(exp.loc["peri_deg"]["goal"]),
            0.0,
        ]
    )

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax = plot_orbit(initial_orbit, color="tab:blue", label="Initial orbit", ax=ax)
    ax = plot_orbit(goal_orbit, color="tab:orange", label="Target orbit", ax=ax)
    plot_earth(ax)

    ax.plot(
        traj_x, traj_y, traj_z, label="Spacecraft trajectory", c="#5c92c4", alpha=0.4
    )
    ax.legend()

    ax.set_box_aspect([1.0, 1.0, 1.0])
    equal_axis_size_3d(ax)

    plt.show()


if __name__ == "__main__":
    main()
