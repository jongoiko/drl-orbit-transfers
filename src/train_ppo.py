#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
from math import radians
from envs.earth_orbit import EarthOrbitMatchingEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo import PPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_file_path", type=str, help="path to experiment CSV file"
    )
    parser.add_argument(
        "--tb_log_dir", default=None, help="TensorBoard logging directory"
    )
    parser.add_argument(
        "--traj_log_dir", default=None, help="trajectory logging directory"
    )
    args = parser.parse_args()
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
    weights = np.array(
        [
            exp.loc["a_km"]["weight"],
            exp.loc["e"]["weight"],
            exp.loc["i_deg"]["weight"],
            exp.loc["raan_deg"]["weight"],
            exp.loc["peri_deg"]["weight"],
        ]
    )
    make_env = lambda: Monitor(
        EarthOrbitMatchingEnv(
            trajectory_log_dir=args.traj_log_dir,
            initial_orbit=initial_orbit,
            target_orbit=goal_orbit,
            integrator_time_step=60.0,
            isp=3300.0,
            max_thrust=10.0,
            initial_mass=1200.0,
            max_episode_length=20 * 24 * 3600.0,
            weights=weights,
        )
    )
    env = make_vec_env(make_env, n_envs=32, seed=42)
    ppo = PPO(
        "MlpPolicy",
        env,
        n_steps=32 * 2048,
        batch_size=32 * 2048,
        learning_rate=0.1e-4,
        n_epochs=5,
        ent_coef=0.01,
        gamma=0.99,
        verbose=1,
        tensorboard_log=args.tb_log_dir,
    )
    ppo.learn(total_timesteps=1e8)


if __name__ == "__main__":
    main()
